from time import sleep
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import warnings

from environment import *
from buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_size, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 256)
        self.fc_2 = nn.Linear(256, 1024)
        self.fc_out = nn.Linear(1024, action_size)

        self.reset_parameters()

        self.lr = q_lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # TODO: dropout, elu & attention

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        q = F.relu(self.fc_1(x))
        q = F.relu(self.fc_2(q))
        q = self.fc_out(q)
        return q

DEFAULT_INIT_EPSILON = 0.95
DEFAULT_END_EPSILON = 0.15  # 保持适度探索
DEFUALT_EPSILON_DECAY = 0.9995  # 大幅降低衰减速度：200ep后epsilon≈0.86，500ep后≈0.72

# TODO parameterize default values
class DQNAgent:
    def __init__(self, env: Environment = Environment(), random_seed=None):
        self.env           = env
        self.state_dim     = self.env.state_dim  # 37维
        self.action_size   = self.env.act_dim    # 最大动作空间
        self.lr            = 1e-3                       # 5e-4, 1e-3
        self.gamma         = GAMMA
        self.epsilon       = DEFAULT_INIT_EPSILON       # 0.95
        self.epsilon_decay = DEFUALT_EPSILON_DECAY      # TODO not sure 0.99 slow/fast enough, or should I choose another decay scheme?
        self.epsilon_min   = DEFAULT_END_EPSILON
        self.targ_update_n = 10
        self.test_every    = 100
        self.memory        = ReplayBuffer(20000, 128, device, self.env)      # NOTE perhaps too large?

        # 创建独立的随机数生成器，确保每个Agent使用不同的随机状态
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()

        self.Q        = QNetwork(self.state_dim, self.action_size, self.lr).to(device)
        self.Q_target = QNetwork(self.state_dim, self.action_size, self.lr).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state: State, use_softmax=True, temperature=1.0):
        """
        选择动作，支持多种探索策略：
        - epsilon-greedy：随机探索
        - softmax：基于Q值概率分布选择（更平滑的探索）
        - UCB bonus：未探索动作获得额外奖励
        """
        # 使用Agent的独立随机数生成器
        random_number = self.rng.rand()
        maxQ_action_count = 0
        
        # 获取动作掩码
        action_mask = state.get_action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            # 如果没有有效动作，使用get_action_idx_limits作为fallback
            ai_low, ai_high = state.get_action_idx_limits()
            return self.rng.randint(ai_low, ai_high + 1), None, 0
        
        # 工艺阶段使用更高的探索率，强制探索工艺参数
        is_process_phase = state.get_episode_count() >= COMP_EPISODE_LEN
        if is_process_phase:
            # 工艺阶段：使用更高的探索率（至少0.6），确保充分探索工艺参数
            effective_epsilon = max(self.epsilon, 0.6)
            # 工艺阶段使用更高的温度，增加随机性
            effective_temperature = max(temperature, 2.0)
        else:
            # 成分阶段：允许更低的探索率，但不能太低
            effective_epsilon = max(self.epsilon, 0.25)
            effective_temperature = temperature
        
        if effective_epsilon < random_number:
            # 利用阶段：使用softmax或贪婪选择
            if use_softmax and len(valid_actions) > 1:
                action = self.choose_action_softmax(state, action_mask, effective_temperature)
            else:
                action = self.choose_action_greedy(state, action_mask)
            maxQ_action_count = 1
        else:
            # 探索阶段：随机选择，但给予较少访问的动作更高概率（UCB思想）
            if hasattr(self, 'action_visit_counts'):
                visit_counts = np.array([self.action_visit_counts.get(a, 0) for a in valid_actions])
                # UCB bonus：1 / sqrt(visit_count + 1)
                ucb_bonus = 1.0 / np.sqrt(visit_counts + 1)
                probs = ucb_bonus / ucb_bonus.sum()
                action = self.rng.choice(valid_actions, p=probs)
            else:
                action = self.rng.choice(valid_actions)
        
        # 记录动作访问次数（用于UCB探索）
        if not hasattr(self, 'action_visit_counts'):
            self.action_visit_counts = {}
        self.action_visit_counts[action] = self.action_visit_counts.get(action, 0) + 1

        return action, None, maxQ_action_count
    
    def choose_action_softmax(self, state: State, action_mask=None, temperature=1.0) -> int:
        """基于Q值的softmax概率分布选择动作，实现更平滑的探索"""
        qs, ai_low, ai_high = self.get_state_qs(state)
        
        if action_mask is None:
            action_mask = state.get_action_mask()
        
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return ai_low
        
        # 获取有效动作的Q值
        valid_qs = qs[valid_actions]
        
        # 数值稳定的softmax计算
        valid_qs = valid_qs - np.max(valid_qs)  # 防止溢出
        exp_qs = np.exp(valid_qs / temperature)
        probs = exp_qs / (exp_qs.sum() + 1e-10)
        
        # 根据概率分布选择动作
        action_idx = self.rng.choice(len(valid_actions), p=probs)
        return valid_actions[action_idx]
    
    def choose_action_greedy(self, state: State, action_mask=None) -> int:
        qs, ai_low, ai_high = self.get_state_qs(state)
        
        # 应用动作掩码
        if action_mask is None:
            action_mask = state.get_action_mask()
        
        # 将无效动作的Q值设为-inf
        masked_qs = qs.copy()
        masked_qs[~action_mask] = -np.inf
        
        # 在有效动作中选择Q值最大的
        action = np.argmax(masked_qs)
        return action
    
    def get_state_qs(self, state: State) -> Tuple[np.ndarray, int, int]:
        with torch.no_grad():
            self.Q.eval()
            state_tensor = torch.tensor(state.repr()).float().unsqueeze(0).to(device)
            qs = self.Q(state_tensor).detach().cpu().numpy().flatten()
            self.Q.train()
        act_idx_low, act_idx_high = state.get_action_idx_limits()
        return qs, act_idx_low, act_idx_high

    def train_agent(self, ep):
        ''' ep for target update '''
        s_batch, a_batch, r_batch, s_prime_batch, done_batch, next_act_idx_bounds = self.memory.sample()
        a_batch = a_batch.type(torch.int64)
        s_batch = s_batch.to(device)
        a_batch = a_batch.to(device)
        r_batch = r_batch.to(device)
        s_prime_batch = s_prime_batch.to(device)
        done_batch = done_batch.to(device)

        with torch.no_grad():
            # 使用主网络选择动作，目标网络评估
            Q_prime_qs_list = self.Q(s_prime_batch)
            # 应用动作掩码（通过next_act_idx_bounds）
            Q_prime_actions = []
            for Q_prime_qs, (ai_low, ai_high) in zip(Q_prime_qs_list, next_act_idx_bounds):
                # 将Q值移到CPU进行处理，避免CUDA内存访问问题
                Q_prime_qs_cpu = Q_prime_qs.detach().cpu()
                
                # 确保索引在有效范围内
                q_size = Q_prime_qs_cpu.shape[0]
                ai_low = max(0, min(int(ai_low), q_size - 1))
                ai_high = max(ai_low, min(int(ai_high), q_size - 1))
                
                # 只在有效动作范围内选择
                if ai_high >= ai_low:
                    # 确保切片不会越界
                    try:
                        q_slice = Q_prime_qs_cpu[ai_low: ai_high + 1]
                        if q_slice.numel() > 0:
                            action_in_range = q_slice.argmax().item() + ai_low
                        else:
                            action_in_range = ai_low  # fallback
                    except (IndexError, RuntimeError) as e:
                        # 如果出现错误，使用安全的fallback
                        action_in_range = ai_low
                else:
                    action_in_range = ai_low  # fallback
                Q_prime_actions.append(action_in_range)
            Q_prime_actions = torch.tensor(Q_prime_actions, dtype=torch.int64).reshape(-1, 1).to(device)
            Q_target_next = self.Q_target(s_prime_batch).gather(1, Q_prime_actions)
            Q_targets = r_batch + self.gamma * (1 - done_batch) * Q_target_next

        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.mse_loss(Q_a, Q_targets)     # NOTE: or smooth_l1_loss?
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()

        ''' Important to clip the grads to avoid exploding gradients '''
        for param in self.Q.parameters():
            param.grad.data.clamp_(min = -1., max = 1.)
        
        self.Q.optimizer.step()

        if ep % self.targ_update_n == 0: 
            for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                param_target.data.copy_(param.data)

    def save_model(self, path):
        torch.save({
            'Q_state_dict': self.Q.state_dict(),
            'Q_target_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.Q.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
        self.Q.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def test_agent(agent: DQNAgent, func):
    state = agent.env.reset()
    done = False
    while not done:
        action = agent.choose_action_greedy(state)
        state_prime, _, done = agent.env.step(state, action)
        state = state_prime

    return func(state.get_composition())

# TODO check
def propose_candidates_to_exp(agent: DQNAgent, ei_agent: DQNAgent, ei_act_prob: float, candidates_n: int):
    '''
        Propose candidates_n xs to do experiment.
    '''
    assert 0. <= ei_act_prob <= 1.

    try_counter, try_max_n = 0, 500
    prop_x_key_set = set()
    prop_x_a_key_set = set()
    while len(prop_x_key_set) < candidates_n:
        state = agent.env.reset()
        done = False
        while not done:
            action_found = False
            inner_loop_count = 0
            max_inner_loops = 100  # 防止内部循环死循环
            while not action_found and inner_loop_count < max_inner_loops:
                inner_loop_count += 1
                # TODO check modify returned val of get_state_qs()
                # 使用agent的随机状态
                if agent.rng.rand() < ei_act_prob:
                    qs, ai_low, ai_high = ei_agent.get_state_qs(state)
                else:
                    qs, ai_low, ai_high = agent.get_state_qs(state)
                sorted_idxs = np.argsort(qs[ai_low: ai_high + 1])[::-1] + ai_low    # sort a slice
                found_unused_action = False
                for action in sorted_idxs:
                    tmp_key = State.encode_key(list(state.get_composition) + [state.get_episode_count(), action])
                    if tmp_key not in prop_x_a_key_set:
                        found_unused_action = True
                        prop_x_a_key_set.add(tmp_key)
                        action_found = True
                        break
                if not found_unused_action:
                    # 如果所有动作都被使用过，随机选择一个
                    ai_low, ai_high = state.get_action_idx_limits()
                    if ai_high >= ai_low:
                        action = agent.rng.randint(ai_low, ai_high + 1)
                        action_found = True
            if not action_found:
                # 如果仍然找不到动作，使用随机选择作为fallback
                ai_low, ai_high = state.get_action_idx_limits()
                if ai_high >= ai_low:
                    action = agent.rng.randint(ai_low, ai_high + 1)
                else:
                    action = 0  # 最后的fallback
            state_prime, _, done = agent.env.step(state, action)
            state = state_prime
        new_x = state.get_composition
        new_x_key = State.encode_key(new_x)
        if (not agent.env.check_collided(new_x)) and (new_x_key not in prop_x_key_set):
            prop_x_key_set.add(new_x_key)
        
        try_counter += 1
        if try_counter >= try_max_n: 
            raise Exception(f'Tried {try_counter} times to propose candidates. Reached maximum allowed attempts.')
    return [State.decode_key(_x_key) for _x_key in prop_x_key_set]

def train_one_ep(agent: DQNAgent, env: Environment, EP: int):
    ''' train <agent> on <env>, with EP_ID of <EP> '''
    _ = state = env.reset()
    done = False
    max_steps = MAX_EPISODE_LEN + 10  # 添加安全上限，防止死循环
    step_count = 0
    
    # 调试：记录动作序列
    action_sequence = []
    episode_reward = 0.0

    while not done:
        step_count += 1
        if step_count > max_steps:
            print(f"警告: Episode {EP} 超过最大步数限制 ({max_steps})，强制结束")
            break
        
        action, _, _ = agent.choose_action(state)
        action_sequence.append(action)
        state_prime, reward, done = env.step(state, action)
        
        # 只在成分阶段应用动作多样性惩罚，工艺阶段不需要
        # 降低惩罚力度，避免过度干扰学习
        if len(action_sequence) >= 10 and state.get_episode_count() < COMP_EPISODE_LEN:
            recent_actions = action_sequence[-10:]
            unique_recent = len(set(recent_actions))
            if unique_recent < 3:  # 提高阈值，从5降到3，只惩罚极端重复
                diversity_penalty = -0.05 * (3 - unique_recent)  # 大幅降低惩罚，从-0.2降到-0.05
                reward += diversity_penalty
        
        episode_reward = reward  # 只在episode结束时reward非零
        agent.memory.add(state, action, reward, state_prime, done)
        state = state_prime
            
        '''
            Train agent with a batch of buffered sars' samples.
            优化：每4个step训练一次，而不是每个step都训练，大幅提升训练速度
            EP to determine if time to target update.
        '''
        # 只有当缓冲区有足够样本时才训练
        if len(agent.memory) > agent.memory.batch_size:
            # 每4个step训练一次，而不是每个step都训练
            # 这样可以大幅提升训练速度（约4倍），同时保持足够的学习频率
            if step_count % 4 == 0:
                agent.train_agent(EP)

    # Episode结束时再训练一次，确保学习到episode结束时的经验
    if len(agent.memory) > agent.memory.batch_size:
        agent.train_agent(EP)
    
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)
    
    # 定期重置探索率，避免过早陷入局部最优
    # 每100个episode检查一次，如果BSF没有改进则增加探索率
    if EP > 0 and EP % 100 == 0:
        if hasattr(env, '_last_bsf_at_reset'):
            current_bsf = env.best_score
            if current_bsf <= env._last_bsf_at_reset * 1.01:  # 改进小于1%
                # 重置探索率到较高水平
                old_epsilon = agent.epsilon
                agent.epsilon = min(0.7, agent.epsilon * 1.5)
                print(f"  [RESET] Episode {EP}: BSF停滞，重置epsilon从{old_epsilon:.3f}到{agent.epsilon:.3f}")
            env._last_bsf_at_reset = current_bsf
        else:
            env._last_bsf_at_reset = env.best_score
    
    # 调试输出（每100个episode输出一次）
    if EP % 100 == 0 and EP > 0:
        unique_actions = len(set(action_sequence))
        comp_actions = [a for a in action_sequence[:COMP_EPISODE_LEN] if a < len(ALL_ACTIONS)]
        proc_actions = action_sequence[COMP_EPISODE_LEN:]
        
        # 检查成分动作多样性
        if len(comp_actions) > 0:
            comp_unique = len(set(comp_actions))
            comp_info = f"comp_actions: unique={comp_unique}/{len(comp_actions)}"
        else:
            comp_info = "comp_actions: none"
        
        # 检查工艺动作多样性
        if len(proc_actions) > 0:
            proc_unique = len(set(proc_actions))
            proc_info = f"proc_actions: unique={proc_unique}/{len(proc_actions)}"
        else:
            proc_info = "proc_actions: none"
        
        print(f"  [DEBUG Episode {EP}] reward={episode_reward:.6f}, total_steps={step_count}, "
              f"unique_actions={unique_actions}/{len(action_sequence)}, {comp_info}, {proc_info}")
        
        # 如果动作多样性太低，发出警告
        if unique_actions < len(action_sequence) * 0.3:
            print(f"  [WARNING] Episode {EP}: 动作多样性过低！只有{unique_actions}/{len(action_sequence)}个唯一动作")
    
    # After finishing one episode, update multi-objective weights in the environment
    # 降低更新频率：每20个episode更新一次，而不是每个episode
    if EP % 20 == 0:
        try:
            if hasattr(env, 'update_mo_weights'):
                env.update_mo_weights(method='improvement', window=50)
        except Exception as e:
            print(f"Warning: update_mo_weights failed: {e}")

def retry_on_error(max_retries = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    retries += 1
            raise Exception(f"Function {func.__name__} failed after {max_retries} retries.")
        return wrapper
    return decorator

def collect_random(env, dataset, num_samples=200, random_seed=None):
    """
    收集随机样本，使用独立的随机数生成器确保可重复性
    
    Args:
        env: Environment对象
        dataset: ReplayBuffer对象
        num_samples: 要收集的样本数量
        random_seed: 随机种子（可选）
    """
    # 创建独立的随机数生成器
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()
    
    state = env.reset()
    for _ in range(num_samples):
        # 根据当前阶段选择动作
        if state.get_episode_count() < COMP_EPISODE_LEN:
            # 成分阶段：使用generate_random_action返回的动作值
            # 传入随机状态以确保可重复性
            action = state.generate_random_action(rng)
            action_idx = ALL_ACTIONS.index(action)
        else:
            # 工艺阶段：使用动作掩码确保选择有效动作
            action_mask = state.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # 从有效动作中随机选择，确保工艺参数被充分探索
                action_idx = rng.choice(valid_actions)
            else:
                # Fallback：使用get_action_idx_limits
                ai_low, ai_high = state.get_action_idx_limits()
                if ai_high >= ai_low:
                    action_idx = rng.randint(ai_low, ai_high + 1)
                else:
                    action_idx = 0
        
        next_state, reward, done = env.step(state, action_idx)
        dataset.add(state, action_idx, reward, next_state, done)  # 存储action_idx
        state = next_state
        if done:
            state = env.reset()
    print(f'Collected {num_samples} random samples')