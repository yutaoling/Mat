"""
快速测试脚本：验证强化学习训练修复是否有效
运行方式: python test_fixes.py
"""
import sys
import os
# 添加父目录到路径，以便导入主模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import random

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print("=" * 60)
print("测试修复后的强化学习训练")
print("=" * 60)

# 测试1: 导入检查
print("\n[测试1] 导入模块...")
try:
    from environment import Environment, State, COMP_EPISODE_LEN, ALL_ACTIONS
    from rl_dqn_agents import DQNAgent, DEFAULT_INIT_EPSILON, DEFUALT_EPSILON_DECAY, DEFAULT_END_EPSILON
    from buffer import ReplayBuffer
    print("  [OK] 所有模块导入成功")
except Exception as e:
    print(f"  [FAIL] 导入失败: {e}")
    exit(1)

# 测试2: 检查新的epsilon参数
print("\n[测试2] 检查探索参数...")
print(f"  INIT_EPSILON = {DEFAULT_INIT_EPSILON}")
print(f"  END_EPSILON = {DEFAULT_END_EPSILON}")
print(f"  EPSILON_DECAY = {DEFUALT_EPSILON_DECAY}")

# 计算200个episode后的epsilon
eps_after_200 = DEFAULT_INIT_EPSILON * (DEFUALT_EPSILON_DECAY ** 200)
eps_after_500 = DEFAULT_INIT_EPSILON * (DEFUALT_EPSILON_DECAY ** 500)
print(f"  200 episodes后 epsilon ≈ {eps_after_200:.4f}")
print(f"  500 episodes后 epsilon ≈ {eps_after_500:.4f}")

if eps_after_200 > 0.7:
    print("  [OK] epsilon衰减速度合理")
else:
    print("  [WARN] epsilon衰减可能仍然太快")

# 测试3: 创建环境和Agent
print("\n[测试3] 创建环境和Agent...")
try:
    env = Environment(init_N=5, enable_ei=False, random_seed=seed)
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()
    
    agent = DQNAgent(env, random_seed=seed)
    print(f"  [OK] 环境创建成功")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作空间: {env.act_dim}")
except Exception as e:
    print(f"  [FAIL] 创建失败: {e}")
    exit(1)

# 测试4: 检查新的动作选择方法
print("\n[测试4] 测试动作选择...")
try:
    state = env.reset()
    
    # 测试choose_action（包含softmax和UCB）
    actions_chosen = []
    for _ in range(20):
        action, _, _ = agent.choose_action(state)
        actions_chosen.append(action)
    
    unique_actions = len(set(actions_chosen))
    print(f"  20次动作选择中唯一动作数: {unique_actions}")
    
    # 检查是否有UCB计数
    if hasattr(agent, 'action_visit_counts'):
        print(f"  [OK] UCB访问计数已启用 (记录了 {len(agent.action_visit_counts)} 个动作)")
    else:
        print("  [FAIL] UCB访问计数未启用")
    
    # 测试softmax选择
    action_softmax = agent.choose_action_softmax(state, temperature=1.0)
    print(f"  [OK] Softmax动作选择正常工作")
    
except Exception as e:
    print(f"  [FAIL] 动作选择测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 测试一个完整的episode
print("\n[测试5] 运行一个完整episode...")
try:
    state = env.reset()
    total_reward = 0
    step_count = 0
    done = False
    
    while not done and step_count < 40:
        action, _, _ = agent.choose_action(state)
        next_state, reward, done = env.step(state, action)
        total_reward += reward
        state = next_state
        step_count += 1
    
    print(f"  Episode完成: steps={step_count}, total_reward={total_reward:.4f}")
    print(f"  最终状态: done={done}")
    
    # 检查是否有工艺参数
    if done:
        comp = state.get_composition()
        proc_bool = state.get_proc_bool()
        proc_scalar = state.get_proc_scalar()
        
        non_zero_elem = np.sum(np.array(comp) > 0.01)
        has_ht1 = proc_scalar[2] > 0
        has_ht2 = proc_scalar[4] > 0
        
        print(f"  成分: {non_zero_elem} 个非零元素")
        print(f"  工艺: HT1={'是' if has_ht1 else '否'}, HT2={'是' if has_ht2 else '否'}")
    
    print("  [OK] Episode运行成功")
    
except Exception as e:
    print(f"  [FAIL] Episode运行失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 检查奖励统计
print("\n[测试6] 检查奖励计算...")
try:
    # 运行多个episode收集奖励
    rewards = []
    for _ in range(10):
        state = env.reset()
        done = False
        step_count = 0
        while not done and step_count < 40:
            action, _, _ = agent.choose_action(state)
            next_state, reward, done = env.step(state, action)
            if reward != 0:
                rewards.append(reward)
            state = next_state
            step_count += 1
    
    if rewards:
        unique_rewards = len(set([round(r, 6) for r in rewards]))
        print(f"  收集到 {len(rewards)} 个非零奖励")
        print(f"  唯一奖励值: {unique_rewards}")
        print(f"  奖励范围: [{min(rewards):.4f}, {max(rewards):.4f}]")
        print(f"  奖励均值: {np.mean(rewards):.4f}")
        print(f"  奖励标准差: {np.std(rewards):.4f}")
        
        if unique_rewards > 1:
            print("  [OK] 奖励具有多样性")
        else:
            print("  [FAIL] 奖励缺乏多样性")
    else:
        print("  [WARN] 没有收集到非零奖励")
        
except Exception as e:
    print(f"  [FAIL] 奖励测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 检查环境统计
print("\n[测试7] 检查环境统计...")
if hasattr(env, '_reward_stats'):
    stats = env._reward_stats
    print(f"  总奖励计算次数: {stats['count']}")
    print(f"  唯一设计数: {len(stats['unique_designs'])}")
    if stats['count'] > 0:
        mean = stats['sum'] / stats['count']
        print(f"  平均奖励: {mean:.4f}")
    print("  [OK] 奖励统计正常工作")
else:
    print("  奖励统计尚未初始化（需要更多episode）")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

# 总结
print("\n【修复总结】")
print("1. epsilon衰减: 0.998 → 0.9995 (更慢的衰减)")
print("2. 最小探索率: 成分阶段0.25, 工艺阶段0.6")
print("3. 奖励缩放: 2x (增加区分度)")
print("4. 添加工艺多样性奖励 (HT1/HT2)")
print("5. 添加新设计发现奖励 (0.15~0.25)")
print("6. 添加UCB探索 + Softmax动作选择")
print("7. 添加定期探索率重置机制")

print("\n【建议】")
print("运行完整训练: python rl_executor.py")
print("使用更长的训练时间 (300+ episodes) 以获得更好的结果")
