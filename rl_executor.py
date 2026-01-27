import random
import warnings
import pandas as pd
from datetime import datetime

import joblib
import numpy as np

from buffer import ReplayBuffer
from environment import (
    COMPOSITION_ROUNDUP_DIGITS, Environment, ALL_ACTIONS_COUNT, MAX_EPISODE_LEN, 
    COMP_MULTIPLIER, COMP_EPISODE_LEN, ALL_ACTIONS, calculate_phase_scalar
)
from rl_dqn_agents import (
    DQNAgent, 
    collect_random, propose_candidates_to_exp, train_one_ep, retry_on_error,
    DEFAULT_INIT_EPSILON,
    device,
)
from model_env import ID, COMP, PROC_BOOL, PROC_SCALAR, PHASE_SCALAR, PROP
import torch
import os
N_JOBS=5

if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.95/N_JOBS)
    except:
        pass

def save_results_to_csv(env: Environment, seed: int):
    """
    将搜索完成后的最佳结果保存到CSV文件，格式与Ti_Alloy_Dataset.xlsx相同
    
    Args:
        env: Environment对象，包含最佳结果
        seed: 随机种子
    """
    try:
        best_x = env.get_best_x()
        best_prop = env.get_best_prop()
        
        if best_x is None or best_prop is None:
            print(f"警告: 未找到最佳结果，跳过CSV保存")
            return
        
        comp = np.array(best_x['comp'])
        proc_bool = np.array(best_x['proc_bool'])
        proc_scalar = np.array(best_x['proc_scalar'])
        phase = np.array(best_x['phase'])
        
        comp_percent = comp * COMP_MULTIPLIER
        
        data_dict = {}
        
        data_dict['id'] = int(10000 + seed)
        data_dict['Activated'] = 1
        
        for i, elem_name in enumerate(COMP):
            data_dict[elem_name] = round(comp_percent[i], 3)
        
        for i, proc_name in enumerate(PROC_BOOL):
            data_dict[proc_name] = int(proc_bool[i]) if i < len(proc_bool) else 0
        
        for i, proc_name in enumerate(PROC_SCALAR):
            value = proc_scalar[i] if i < len(proc_scalar) else 0.0
            if proc_name in ['Def_Temp', 'HT1_Temp', 'HT2_Temp']:
                data_dict[proc_name] = round(value, 1)
            elif proc_name in ['HT1_Time', 'HT2_Time']:
                data_dict[proc_name] = round(value, 3)
            else:
                data_dict[proc_name] = round(value, 1)
        
        
        for i, phase_name in enumerate(PHASE_SCALAR):
            data_dict[phase_name] = round(phase[i], 2) if i < len(phase) else 0.0
        
        prop_array = np.array(best_prop) if isinstance(best_prop, (list, np.ndarray)) else best_prop
        for i, prop_name in enumerate(PROP):
            if i < len(prop_array):
                value = prop_array[i]
                if prop_name in ['YM']:
                    data_dict[prop_name] = round(value, 1)
                elif prop_name in ['YS', 'UTS']:
                    data_dict[prop_name] = round(value, 1)
                elif prop_name in ['El']:
                    data_dict[prop_name] = round(value, 1)
                elif prop_name in ['HV']:
                    data_dict[prop_name] = round(value, 1)
                else:
                    data_dict[prop_name] = round(value, 1)
            else:
                data_dict[prop_name] = None
        
        df = pd.DataFrame([data_dict])       
        column_order = (ID + COMP + PROC_BOOL + PROC_SCALAR+ PHASE_SCALAR + PROP)
        
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'results/rl_best_result_seed{seed}_{timestamp}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        import traceback
        traceback.print_exc()

def rl_dqn_two_stage(init_N = 20,
                     seed = 0,
                     comp_train_ep_n = 50,
                     proc_train_ep_n = 50,
                     ):
    '''
    两阶段训练：先优化成分，再优化工艺参数
    阶段1：训练成分Agent，找到好的成分
    阶段2：固定成分，训练工艺Agent优化工艺参数
    '''
    # 设置随机种子，确保每个进程使用独立的随机状态
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()
    
    print(f"\n{'='*60}")
    print(f"阶段1：优化成分（{comp_train_ep_n} episodes）")
    print(f"{'='*60}")
    
    comp_agent = DQNAgent(env, random_seed=seed)  # 传递seed
    comp_agent.memory = ReplayBuffer(3000, 256, device, env)
    comp_agent.epsilon = DEFAULT_INIT_EPSILON
    
    collect_random(env, comp_agent.memory, 2000, random_seed=seed)  # 传递seed
    
    for ep in range(comp_train_ep_n + 1):
        state = env.reset()
        done = False
        step_count = 0
        max_steps = COMP_EPISODE_LEN + 5
        
        while not done and state.get_episode_count() < COMP_EPISODE_LEN:
            step_count += 1
            if step_count > max_steps:
                break
            
            action, _, _ = comp_agent.choose_action(state)
            state_prime, reward, done = env.step(state, action)
            
            if state_prime.get_episode_count() >= COMP_EPISODE_LEN:
                while not state_prime.done():
                    action_mask = state_prime.get_action_mask()
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        ai_low, ai_high = state_prime.get_action_idx_limits()
                        action = np.random.randint(ai_low, ai_high + 1)
                    state_prime, reward, done = env.step(state_prime, action)
                
                comp = state_prime.get_composition()
                proc_bool = state_prime.get_proc_bool()
                proc_scalar = state_prime.get_proc_scalar()
                phase = calculate_phase_scalar(comp)
                reward = env.func_with_proc(comp, proc_bool, proc_scalar, phase)
                done = True
            
            comp_agent.memory.add(state, action, reward, state_prime, done)
            state = state_prime
            comp_agent.train_agent(ep)
        
        if comp_agent.epsilon > comp_agent.epsilon_min:
            comp_agent.epsilon *= comp_agent.epsilon_decay
            comp_agent.epsilon = max(comp_agent.epsilon, comp_agent.epsilon_min)
        
        if ep % 10 == 0:
            print(f"阶段1 Episode {ep}: best_score={env.best_score:.4f}")
    
    best_comp = env.get_best_x()['comp'] if env.get_best_x() else None
    print(f"\n阶段1完成，最佳成分分数: {env.best_score:.4f}")
    
    print(f"\n{'='*60}")
    print(f"阶段2：优化工艺参数（固定成分，{proc_train_ep_n} episodes）")
    print(f"{'='*60}")
    
    proc_agent = DQNAgent(env, random_seed=seed + 10000)
    proc_agent.memory = ReplayBuffer(3000, 256, device, env)
    proc_agent.epsilon = 0.8
    
    proc_rng = np.random.RandomState(seed + 20000)
    for _ in range(1000):
        state = env.reset()
        while state.get_episode_count() < COMP_EPISODE_LEN:
            if best_comp is not None:
                comp = state.get_composition()
                for i in range(min(len(comp), len(best_comp))):
                    comp[i] = best_comp[i]
            action = state.generate_random_action(proc_rng)
            action_idx = ALL_ACTIONS.index(action)
            state, _, _ = env.step(state, action_idx)
        
        while not state.done():
            action_mask = state.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = proc_rng.choice(valid_actions)
            else:
                ai_low, ai_high = state.get_action_idx_limits()
                action = proc_rng.randint(ai_low, ai_high + 1)
            
            state_prime, reward, done = env.step(state, action)
            if done:
                comp = state_prime.get_composition()
                proc_bool = state_prime.get_proc_bool()
                proc_scalar = state_prime.get_proc_scalar()
                phase = calculate_phase_scalar(comp)
                reward = env.func_with_proc(comp, proc_bool, proc_scalar, phase)
            
            proc_agent.memory.add(state, action, reward, state_prime, done)
            state = state_prime
            if done:
                break
    
    for ep in range(proc_train_ep_n + 1):
        state = env.reset()
        if best_comp is not None:
            comp_rng = np.random.RandomState(seed + 30000 + ep)
            while state.get_episode_count() < COMP_EPISODE_LEN:
                comp = state.get_composition()
                for i in range(min(len(comp), len(best_comp))):
                    comp[i] = best_comp[i]
                action = state.generate_random_action(comp_rng)
                action_idx = ALL_ACTIONS.index(action)
                state, _, _ = env.step(state, action_idx)
        
        done = False
        step_count = 0
        max_steps = MAX_EPISODE_LEN + 10
        
        while not done:
            step_count += 1
            if step_count > max_steps:
                break
            
            action, _, _ = proc_agent.choose_action(state)
            state_prime, reward, done = env.step(state, action)
            
            if done:
                comp = state_prime.get_composition()
                proc_bool = state_prime.get_proc_bool()
                proc_scalar = state_prime.get_proc_scalar()
                phase = calculate_phase_scalar(comp)
                reward = env.func_with_proc(comp, proc_bool, proc_scalar, phase)
            
            proc_agent.memory.add(state, action, reward, state_prime, done)
            state = state_prime
            proc_agent.train_agent(ep)
        
        if proc_agent.epsilon > proc_agent.epsilon_min:
            proc_agent.epsilon *= proc_agent.epsilon_decay
            proc_agent.epsilon = max(proc_agent.epsilon, 0.2)  # 工艺阶段保持最小0.2的探索率
        
        if ep % 10 == 0:
            print(f"阶段2 Episode {ep}: best_score={env.best_score:.4f}, "
                  f"HT1_Temp={env.get_best_x()['proc_scalar'][2]:.1f} "
                  f"HT2_Temp={env.get_best_x()['proc_scalar'][4]:.1f}")
    
    save_results_to_csv(env, seed)
    print(f"\n两阶段训练完成，最终最佳分数: {env.best_score:.4f}")

def rl_dqn_serial(init_N = 20,
                  seed = 0,
                  train_ep_n = 300,  # 增加训练episode数量
                  ):
    '''
        Train one DQN agent using on-the-fly rewards.
        改进版本：增加探索、改进奖励、更好的调试输出
    '''
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()

    agent = DQNAgent(env, random_seed=seed)

    TRAIN_START_MIN_MEMORY = 5000
    
    
    agent.memory = ReplayBuffer(10000, 256, device, env)
    agent.epsilon = DEFAULT_INIT_EPSILON
    collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY, random_seed=seed)
    print(f"初始经验缓冲区大小: {len(agent.memory)}")

    traj = []
    
    best_score_history = []
    no_improvement_count = 0
    last_best_score = -float('inf')
    
    ''' train ONE agent using on-the-fly rewards '''
    for ep in range(train_ep_n + 1):
        train_one_ep(agent, env, ep)
        
        current_best = env.best_score
        best_score_history.append(current_best)
        
        if len(best_score_history) > 30:
            recent_best = max(best_score_history[-30:])
            if current_best <= recent_best * 1.001:
                no_improvement_count += 1
                if no_improvement_count > 30:
                    old_epsilon = agent.epsilon
                    agent.epsilon = min(0.7, agent.epsilon * 1.3)
                    no_improvement_count = 0
                    if abs(old_epsilon - agent.epsilon) > 0.01:
                        print(f"  [STAGNATION] Episode {ep}: 性能停滞{no_improvement_count}次，"
                              f"增加epsilon: {old_epsilon:.3f} -> {agent.epsilon:.3f}")
            else:
                no_improvement_count = 0
        
        if ep % 10 == 0 or ep<=10:
            bsf = round(env.best_score, COMPOSITION_ROUNDUP_DIGITS)
            _tmp_res = [ep, bsf]
            traj.append(_tmp_res)
            
            epsilon_info = f"eps={agent.epsilon:.3f}"
            
            best_x = env.get_best_x()
            proc_info = ""
            comp_info = ""
            if best_x:
                if 'proc_scalar' in best_x:
                    proc_scalar = best_x['proc_scalar']
                    proc_bool = best_x.get('proc_bool', np.zeros(8))
                    is_wrought = "W" if proc_bool[1] > 0.5 else "N"
                    ht1 = "HT1" if proc_scalar[2] > 0 else "---"
                    ht2 = "HT2" if proc_scalar[4] > 0 else "---"
                    proc_info = f"{is_wrought}/{ht1}/{ht2}"
                if 'comp' in best_x:
                    comp = np.array(best_x['comp'])
                    non_zero = np.sum(comp > 0.01)
                    top_elems = np.argsort(comp)[-3:][::-1]
                    comp_info = f"elem={non_zero}"
            
            unique_designs = len(env.surrogate_buffer) if hasattr(env, 'surrogate_buffer') else 0
            
            prop_str = ""
            if env.get_best_prop() is not None:
                props = env.get_best_prop()
                # [YM, YS, UTS, El, HV]
                prop_str = f"YS={props[1]:.0f} UTS={props[2]:.0f} El={props[3]:.1f}"
            
            print(f"Ep {ep:4d} | BSF={bsf:.4f} | {epsilon_info} | {proc_info:12s} | "
                  f"{comp_info:8s} | {prop_str:25s} | unique={unique_designs:4d} | ")
  
    bsf_list = []
    for item in env.surrogate_buffer_list:
        if isinstance(item, dict):
            reward = env.func_with_proc(
                item['comp'], 
                item['proc_bool'], 
                item['proc_scalar'], 
                item['phase']
            ) * 2.0
            bsf_list.append(reward)
        else:
            bsf_list.append(env.func(item) * 2.0)

    bsf_list = np.maximum.accumulate(bsf_list).tolist()
    
    print(f"总探索设计数: {len(env.surrogate_buffer)}")
    unique_bsf = len(set([round(x, 6) for x in bsf_list]))
    print(f"BSF唯一值数量: {unique_bsf}/{len(bsf_list)} ({100*unique_bsf/len(bsf_list):.1f}%)")
    
    txt_filename = f'results/rl_single_agent_{seed}.txt'
    np.savetxt(txt_filename, bsf_list, fmt='%.6f')
  
    _props = env.get_best_prop()
    if _props is not None:
        print(f"\n预测性能:")
        prop_names = ['YM(GPa)', 'YS(MPa)', 'UTS(MPa)', 'El(%)', 'HV']
        for name, val in zip(prop_names, _props):
            print(f"  {name}: {val:.1f}")
    
    save_results_to_csv(env, seed)

if __name__ == '__main__':
    TRAIN_MODE = 'single'
    base_seed = random.randint(0, 10000)
    seeds = [base_seed + i * 1000 for i in range(N_JOBS)]
    
    if TRAIN_MODE == 'two_stage':
        joblib.Parallel(n_jobs = N_JOBS)(
            joblib.delayed(rl_dqn_two_stage)(
                comp_train_ep_n = 80,
                proc_train_ep_n = 80,
                seed = sd
            ) for sd in seeds)
    else:
        joblib.Parallel(n_jobs = N_JOBS)(
            joblib.delayed(rl_dqn_serial)(
                train_ep_n = 300, seed = sd
            ) for sd in seeds)