import random
import uuid
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
# 为每个进程设置不同的CUDA设备（如果有多个GPU）或使用CPU
# 如果只有一个GPU，减少并行数量或使用CPU
if torch.cuda.is_available():
    # 尝试设置内存限制，但可能不够稳定
    try:
        torch.cuda.set_per_process_memory_fraction(0.8/N_JOBS)  # 降低内存使用
    except:
        pass
    # 如果CUDA设备数量少于job数量，建议减少并行数或使用CPU
    if torch.cuda.device_count() < N_JOBS:
        print(f"警告: CUDA设备数量({torch.cuda.device_count()})少于并行job数量({N_JOBS})，可能导致CUDA错误")
        print("建议: 减少N_JOBS或使用CPU")
else:
    print("未检测到CUDA，将使用CPU")

def save_results_to_csv(env: Environment, id_str: str, seed: int):
    """
    将搜索完成后的最佳结果保存到CSV文件，格式与Ti_Alloy_Dataset.xlsx相同
    
    Args:
        env: Environment对象，包含最佳结果
        id_str: 唯一标识符
        seed: 随机种子
    """
    try:
        best_x = env.get_best_x()
        best_prop = env.get_best_prop()
        
        if best_x is None or best_prop is None:
            print(f"警告: 未找到最佳结果，跳过CSV保存")
            return
        
        # 提取数据
        comp = np.array(best_x['comp'])  # 归一化格式 (0-1)
        proc_bool = np.array(best_x['proc_bool'])  # 8维布尔数组
        proc_scalar = np.array(best_x['proc_scalar'])  # 6维标量数组
        phase = np.array(best_x['phase'])  # 3维金相信息
        
        # 将成分从归一化格式转换为百分比格式
        comp_percent = comp * COMP_MULTIPLIER  # 转换为百分比 (0-100)
        
        # 构建数据字典
        data_dict = {}
        
        # ID列
        data_dict['id'] = int(10000 + seed)  # 使用种子生成ID
        data_dict['Activated'] = 1
        
        # 成分列 (COMP) - 按COMP顺序
        for i, elem_name in enumerate(COMP):
            data_dict[elem_name] = round(comp_percent[i], 3)
        
        # 工艺布尔参数 (PROC_BOOL)
        # proc_bool格式: [Is_Not_Wrought, Is_Wrought, HT1_Quench, HT1_Air, HT1_Furnace, HT2_Quench, HT2_Air, HT2_Furnace]
        for i, proc_name in enumerate(PROC_BOOL):
            data_dict[proc_name] = int(proc_bool[i]) if i < len(proc_bool) else 0
        
        # 工艺标量参数 (PROC_SCALAR)
        # proc_scalar格式: [Def_Temp, Def_Strain, HT1_Temp, HT1_Time, HT2_Temp, HT2_Time]
        for i, proc_name in enumerate(PROC_SCALAR):
            value = proc_scalar[i] if i < len(proc_scalar) else 0.0
            # 如果是温度或时间，保留小数；如果是应变，保留1位小数
            if proc_name in ['Def_Temp', 'HT1_Temp', 'HT2_Temp']:
                data_dict[proc_name] = round(value, 1)
            elif proc_name in ['HT1_Time', 'HT2_Time']:
                data_dict[proc_name] = round(value, 3)
            else:  # Def_Strain
                data_dict[proc_name] = round(value, 1)
        
        # 额外的变形类型列（Excel中有但State中没有，设置为默认值）
        data_dict['Def_Type_Casting'] = 0
        data_dict['Def_Type_Rolling'] = 0
        data_dict['Def_Type_Forging'] = 0
        data_dict['Def_Type_Extrusion'] = 0
        
        # 额外的相信息列（Excel中有但State中没有，设置为默认值）
        data_dict['Phase_Info'] = 0
        data_dict['Phase_Alpha'] = 0
        data_dict['Phase_Beta'] = 0
        data_dict['Phase_Alpha_Prime'] = 0
        data_dict['Phase_Alpha_Double_Prime'] = 0
        data_dict['Phase_Omega'] = 0
        data_dict['Phase_Intermetallic'] = 0
        
        # 金相信息 (PHASE_SCALAR)
        for i, phase_name in enumerate(PHASE_SCALAR):
            data_dict[phase_name] = round(phase[i], 2) if i < len(phase) else 0.0
        
        # 性能参数 (PROP)
        # best_prop格式: [YM, YS, UTS, El, HV]
        prop_array = np.array(best_prop) if isinstance(best_prop, (list, np.ndarray)) else best_prop
        for i, prop_name in enumerate(PROP):
            if i < len(prop_array):
                value = prop_array[i]
                # 根据属性类型设置精度
                if prop_name in ['YM']:
                    data_dict[prop_name] = round(value, 1)  # GPa
                elif prop_name in ['YS', 'UTS']:
                    data_dict[prop_name] = round(value, 1)  # MPa
                elif prop_name in ['El']:
                    data_dict[prop_name] = round(value, 1)  # %
                elif prop_name in ['HV']:
                    data_dict[prop_name] = round(value, 1)  # HV
                else:
                    data_dict[prop_name] = round(value, 1)
            else:
                data_dict[prop_name] = None
        
        # 创建DataFrame
        df = pd.DataFrame([data_dict])
        
        # 确保列顺序与Ti_Alloy_Dataset.xlsx相同
        # 完整顺序: ID + COMP + PROC_BOOL + PROC_SCALAR + 额外列 + PHASE_SCALAR + PROP + 其他
        extra_proc_cols = ['Def_Type_Casting', 'Def_Type_Rolling', 'Def_Type_Forging', 'Def_Type_Extrusion']
        extra_phase_cols = ['Phase_Info', 'Phase_Alpha', 'Phase_Beta', 'Phase_Alpha_Prime', 
                            'Phase_Alpha_Double_Prime', 'Phase_Omega', 'Phase_Intermetallic']
        other_cols = ['doi', 'Reference']
        
        column_order = (ID + COMP + PROC_BOOL + PROC_SCALAR + extra_proc_cols + 
                       extra_phase_cols + PHASE_SCALAR + PROP + other_cols)
        
        # 只保留存在的列
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # 添加缺失的其他列（如果不存在）
        if 'doi' not in df.columns:
            df['doi'] = None
        if 'Reference' not in df.columns:
            df['Reference'] = f'RL_DQN_Search_{id_str}'
        
        # 保存到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'results/rl_best_result_{id_str}_seed{seed}_{timestamp}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"结果已保存到: {csv_filename}")
        print(f"最佳性能: {dict(zip(PROP, [float(round(x, 1)) for x in prop_array]))}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        import traceback
        traceback.print_exc()

def rl_dqn_two_stage(init_N = 20,
                     seed = 0,
                     comp_train_ep_n = 50,  # 成分阶段训练episode数
                     proc_train_ep_n = 50,  # 工艺阶段训练episode数
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
    
    id_str = str(uuid.uuid4())[:8]
    
    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()
    
    # ========== 阶段1：优化成分 ==========
    print(f"\n{'='*60}")
    print(f"阶段1：优化成分（{comp_train_ep_n} episodes）")
    print(f"{'='*60}")
    
    comp_agent = DQNAgent(env, random_seed=seed)  # 传递seed
    comp_agent.memory = ReplayBuffer(3000, 256, device, env)
    comp_agent.epsilon = DEFAULT_INIT_EPSILON
    
    # 收集随机样本（只包含成分阶段）
    collect_random(env, comp_agent.memory, 2000, random_seed=seed)  # 传递seed
    
    for ep in range(comp_train_ep_n + 1):
        state = env.reset()
        done = False
        step_count = 0
        max_steps = COMP_EPISODE_LEN + 5
        
        # 只训练到成分阶段结束
        while not done and state.get_episode_count() < COMP_EPISODE_LEN:
            step_count += 1
            if step_count > max_steps:
                break
            
            action, _, _ = comp_agent.choose_action(state)
            state_prime, reward, done = env.step(state, action)
            
            # 如果进入工艺阶段，使用默认工艺参数并结束episode
            if state_prime.get_episode_count() >= COMP_EPISODE_LEN:
                # 快速完成工艺阶段，使用默认值
                while not state_prime.done():
                    # 使用随机动作快速完成
                    action_mask = state_prime.get_action_mask()
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        ai_low, ai_high = state_prime.get_action_idx_limits()
                        action = np.random.randint(ai_low, ai_high + 1)
                    state_prime, reward, done = env.step(state_prime, action)
                
                # 计算最终reward
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
    
    # 保存阶段1的最佳成分
    best_comp = env.get_best_x()['comp'] if env.get_best_x() else None
    print(f"\n阶段1完成，最佳成分分数: {env.best_score:.4f}")
    
    # ========== 阶段2：优化工艺参数（固定成分） ==========
    print(f"\n{'='*60}")
    print(f"阶段2：优化工艺参数（固定成分，{proc_train_ep_n} episodes）")
    print(f"{'='*60}")
    
    proc_agent = DQNAgent(env, random_seed=seed + 10000)  # 使用不同的seed确保独立性
    proc_agent.memory = ReplayBuffer(3000, 256, device, env)
    proc_agent.epsilon = 0.8  # 工艺阶段使用更高的初始探索率
    
    # 收集工艺阶段的随机样本
    proc_rng = np.random.RandomState(seed + 20000)  # 使用独立的随机状态
    for _ in range(1000):
        state = env.reset()
        # 快速设置到工艺阶段开始
        while state.get_episode_count() < COMP_EPISODE_LEN:
            if best_comp is not None:
                # 使用最佳成分
                comp = state.get_composition()
                for i in range(min(len(comp), len(best_comp))):
                    comp[i] = best_comp[i]
            action = state.generate_random_action(proc_rng)  # 使用独立的随机状态
            action_idx = ALL_ACTIONS.index(action)
            state, _, _ = env.step(state, action_idx)
        
        # 收集工艺阶段的样本
        while not state.done():
            action_mask = state.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = proc_rng.choice(valid_actions)  # 使用独立的随机状态
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
        # 快速设置到工艺阶段，使用最佳成分
        if best_comp is not None:
            comp_rng = np.random.RandomState(seed + 30000 + ep)  # 每个episode使用不同的随机状态
            while state.get_episode_count() < COMP_EPISODE_LEN:
                comp = state.get_composition()
                for i in range(min(len(comp), len(best_comp))):
                    comp[i] = best_comp[i]
                action = state.generate_random_action(comp_rng)  # 使用独立的随机状态
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
    
    # 保存结果
    save_results_to_csv(env, id_str, seed)
    print(f"\n两阶段训练完成，最终最佳分数: {env.best_score:.4f}")

def rl_dqn_serial(init_N = 20,
                  seed = 0,
                  train_ep_n = 300,  # 增加训练episode数量
                  ):
    '''
        Train one DQN agent using on-the-fly rewards.
        改进版本：增加探索、改进奖励、更好的调试输出
    '''
    # 设置随机种子，确保每个进程使用独立的随机状态
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    id_str = str(uuid.uuid4())[:8]
    
    print(f"\n{'='*60}")
    print(f"开始训练 DQN Agent (seed={seed}, id={id_str})")
    print(f"训练参数: init_N={init_N}, train_ep_n={train_ep_n}")
    print(f"{'='*60}\n")

    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)

    ''' 
        Clear internal buffers.
        The internal buffers are intially built for GPR rewards.
    '''
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()

    agent = DQNAgent(env, random_seed=seed)  # 传递seed给Agent

    TRAIN_START_MIN_MEMORY = 5000  # 增加初始经验数量
    
    ''' 
        收集更多初始随机经验，确保经验缓冲区包含足够多样的样本
    '''
    agent.memory = ReplayBuffer(10000, 256, device, env)  # 增大缓冲区
    agent.epsilon = DEFAULT_INIT_EPSILON
    print(f"收集初始随机经验 ({TRAIN_START_MIN_MEMORY} samples)...")
    collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY, random_seed=seed)  # 传递seed
    print(f"初始经验收集完成，缓冲区大小: {len(agent.memory)}")

    traj = []
    
    # 性能监控：跟踪最佳分数历史，用于检测停滞
    best_score_history = []
    no_improvement_count = 0
    last_best_score = -float('inf')
    
    # 记录训练开始时间
    import time
    start_time = time.time()

    ''' train ONE agent using on-the-fly rewards '''
    print(f"\n开始训练循环 ({train_ep_n} episodes)...")
    for ep in range(train_ep_n + 1):
        train_one_ep(agent, env, ep)
        
        # 性能监控：检测是否停滞
        current_best = env.best_score
        best_score_history.append(current_best)
        
        # 如果连续30个episode没有改进（降低阈值），增加探索率
        if len(best_score_history) > 30:
            recent_best = max(best_score_history[-30:])
            if current_best <= recent_best * 1.001:  # 允许0.1%的波动
                no_improvement_count += 1
                if no_improvement_count > 30:
                    # 增加探索率，帮助跳出局部最优
                    old_epsilon = agent.epsilon
                    agent.epsilon = min(0.7, agent.epsilon * 1.3)  # 更激进的增加
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
            
            # 计算训练进度和速度
            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eta = (train_ep_n - ep) / eps_per_sec if eps_per_sec > 0 else 0
            
            # 收集统计信息
            epsilon_info = f"eps={agent.epsilon:.3f}"
            
            # 打印工艺参数信息，监控工艺探索
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
            
            # 统计唯一设计数量
            unique_designs = len(env.surrogate_buffer) if hasattr(env, 'surrogate_buffer') else 0
            
            # 获取性能预测
            prop_str = ""
            if env.get_best_prop() is not None:
                props = env.get_best_prop()
                # [YM, YS, UTS, El, HV]
                prop_str = f"YS={props[1]:.0f} UTS={props[2]:.0f} El={props[3]:.1f}"
            
            print(f"Ep {ep:4d} | BSF={bsf:.4f} | {epsilon_info} | {proc_info:12s} | "
                  f"{comp_info:8s} | {prop_str:25s} | unique={unique_designs:4d} | "
                  f"speed={eps_per_sec:.1f}ep/s ETA={eta/60:.1f}min")

    # 训练完成，输出总结
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}")
    print(f"总训练时间: {total_time/60:.1f} 分钟")
    print(f"平均速度: {(train_ep_n+1)/total_time:.1f} episodes/秒")
    
    # 从buffer_list中提取成分计算reward（用于历史记录）
    bsf_list = []
    for item in env.surrogate_buffer_list:
        if isinstance(item, dict):
            # 使用完整参数计算reward（乘以2.0以匹配新的奖励缩放）
            reward = env.func_with_proc(
                item['comp'], 
                item['proc_bool'], 
                item['proc_scalar'], 
                item['phase']
            ) * 2.0  # 匹配step函数中的reward_scale
            bsf_list.append(reward)
        else:
            bsf_list.append(env.func(item) * 2.0)

    # 计算累积最大值（Best So Far）
    bsf_list = np.maximum.accumulate(bsf_list).tolist()
    
    # 统计分析
    print(f"\n【训练统计】")
    print(f"总探索设计数: {len(env.surrogate_buffer)}")
    unique_bsf = len(set([round(x, 6) for x in bsf_list]))
    print(f"BSF唯一值数量: {unique_bsf}/{len(bsf_list)} ({100*unique_bsf/len(bsf_list):.1f}%)")
    
    if unique_bsf == 1:
        print(f"[WARNING] 所有episode的BSF值完全相同: {bsf_list[0]:.6f}")
    elif unique_bsf < len(bsf_list) * 0.1:
        print(f"[WARNING] BSF值多样性很低!")

    joblib.dump(bsf_list, f'results/rl_single_agent_direct_R-{id_str}.pkl')
    
    # 输出最佳设计详情
    print(f"\n【最佳设计】")
    best_x = env.get_best_x()
    if best_x:
        print(f"成分 (非零元素):")
        comp = np.array(best_x['comp'])
        for i, (elem, val) in enumerate(zip(COMP, comp)):
            if val > 0.001:
                print(f"  {elem}: {val*100:.2f}%")
        
        print(f"\n工艺参数:")
        proc_bool = best_x['proc_bool']
        proc_scalar = best_x['proc_scalar']
        print(f"  Is_Wrought: {proc_bool[1] > 0.5}")
        print(f"  Def_Temp: {proc_scalar[0]:.1f}°C, Def_Strain: {proc_scalar[1]:.1f}%")
        print(f"  HT1: Temp={proc_scalar[2]:.1f}°C, Time={proc_scalar[3]:.2f}h")
        print(f"  HT2: Temp={proc_scalar[4]:.1f}°C, Time={proc_scalar[5]:.2f}h")
    
    _props = env.get_best_prop()
    if _props is not None:
        print(f"\n预测性能:")
        prop_names = ['YM(GPa)', 'YS(MPa)', 'UTS(MPa)', 'El(%)', 'HV']
        for name, val in zip(prop_names, _props):
            print(f"  {name}: {val:.1f}")
    
    # 保存结果到CSV文件
    save_results_to_csv(env, id_str, seed)
    
    print(f"\n结果已保存: results/rl_single_agent_direct_R-{id_str}.pkl")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    # 选择训练模式：'single' 单agent训练, 'two_stage' 两阶段训练
    TRAIN_MODE = 'single'  # 可以改为 'two_stage' 使用两阶段训练
    
    # 为每个job设置不同的种子，确保搜索空间不同
    # 使用job索引作为基础种子，避免重复
    base_seed = random.randint(0, 10000)  # 随机基础种子
    seeds = [base_seed + i * 1000 for i in range(N_JOBS)]  # 每个job间隔1000，确保不重复
    
    print(f"\n{'='*60}")
    print(f"强化学习钛合金设计优化")
    print(f"{'='*60}")
    print(f"训练模式: {TRAIN_MODE}")
    print(f"并行jobs数: {N_JOBS}")
    print(f"使用种子: {seeds}")
    print(f"\n改进内容:")
    print(f"  - epsilon衰减: 0.9995 (原0.998，更慢的衰减)")
    print(f"  - 最小探索率: 成分阶段0.25, 工艺阶段0.6")
    print(f"  - 奖励缩放: 2x (增加区分度)")
    print(f"  - 工艺多样性奖励: HT1最高+0.17, HT2最高+0.20")
    print(f"  - 新设计发现奖励: 0.15~0.25")
    print(f"  - UCB探索 + Softmax动作选择")
    print(f"  - 定期探索率重置机制")
    print(f"{'='*60}\n")
    
    if TRAIN_MODE == 'two_stage':
        # 两阶段训练：先优化成分，再优化工艺
        joblib.Parallel(n_jobs = N_JOBS)(
            joblib.delayed(rl_dqn_two_stage)(
                comp_train_ep_n = 80,   # 增加成分阶段训练
                proc_train_ep_n = 80,   # 增加工艺阶段训练
                seed = sd
            ) for sd in seeds)
    else:
        # 单agent训练（改进版）
        joblib.Parallel(n_jobs = N_JOBS)(
            joblib.delayed(rl_dqn_serial)(
                train_ep_n = 300, seed = sd  # 增加训练episode数
            ) for sd in seeds)