"""
工艺参数常量定义
包含工艺参数的候选值列表和归一化参数
"""
import numpy as np

# ==================== 工艺参数候选值列表 ====================

# 变形温度（step >= 25°C）
DEF_TEMP_CANDIDATES = [
    0,      # 不变形
    600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150
]
DEF_TEMP_CANDIDATES = np.array(DEF_TEMP_CANDIDATES, dtype=np.float32)
DEF_TEMP_MAX = 1150.0
DEF_TEMP_COUNT = len(DEF_TEMP_CANDIDATES)

# 变形量（step >= 10%）
DEF_STRAIN_CANDIDATES = [
    0.0,    # 不变形
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0
]
DEF_STRAIN_CANDIDATES = np.array(DEF_STRAIN_CANDIDATES, dtype=np.float32)
DEF_STRAIN_MAX = 99.0
DEF_STRAIN_COUNT = len(DEF_STRAIN_CANDIDATES)

# HT1温度（step >= 25°C）
HT1_TEMP_CANDIDATES = [
    0,      # 不进行HT1
    400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 
    725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 
    1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400
]
HT1_TEMP_CANDIDATES = np.array(HT1_TEMP_CANDIDATES, dtype=np.float32)
HT1_TEMP_MAX = 1400.0
HT1_TEMP_COUNT = len(HT1_TEMP_CANDIDATES)

# HT1时间（小时）
HT1_TIME_CANDIDATES = [
    0.0,    # 不进行HT1
    0.017, 0.05, 0.083, 0.167, 0.25, 0.33, 0.5, 0.58, 
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 24.0, 30.0
]
HT1_TIME_CANDIDATES = np.array(HT1_TIME_CANDIDATES, dtype=np.float32)
HT1_TIME_MAX = 30.0
HT1_TIME_COUNT = len(HT1_TIME_CANDIDATES)

# HT2温度（step >= 25°C）
HT2_TEMP_CANDIDATES = [
    0,      # 不进行HT2
    200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 
    525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
]
HT2_TEMP_CANDIDATES = np.array(HT2_TEMP_CANDIDATES, dtype=np.float32)
HT2_TEMP_MAX = 1000.0
HT2_TEMP_COUNT = len(HT2_TEMP_CANDIDATES)

# HT2时间（小时）
HT2_TIME_CANDIDATES = [
    0.0,    # 不进行HT2
    0.167, 0.33, 0.5, 1.0, 1.5, 1.67, 2.0, 3.0, 4.0, 5.0, 6.0, 
    7.0, 8.0, 8.5, 9.0, 10.0, 12.0, 15.0, 18.0, 24.0, 30.0, 36.0, 48.0, 72.0
]
HT2_TIME_CANDIDATES = np.array(HT2_TIME_CANDIDATES, dtype=np.float32)
HT2_TIME_MAX = 72.0
HT2_TIME_COUNT = len(HT2_TIME_CANDIDATES)

# 冷却方式
COOLING_METHODS = ['Quench', 'Air', 'Furnace']
COOLING_METHOD_COUNT = len(COOLING_METHODS)

# ==================== 动作空间定义 ====================

# 工艺阶段各步骤的动作空间大小
PROC_ACTIONS = {
    'init_state': 2,              # 初始状态选择
    'deform_decision': 2,         # 变形决策
    'def_temp': DEF_TEMP_COUNT,   # Def_Temp
    'def_strain': DEF_STRAIN_COUNT,  # Def_Strain
    'ht1_decision': 2,           # HT1决策
    'ht1_temp': HT1_TEMP_COUNT,   # HT1_Temp
    'ht1_time': HT1_TIME_COUNT,   # HT1_Time
    'ht1_cooling': COOLING_METHOD_COUNT,  # HT1冷却方式
    'ht2_decision': 2,            # HT2决策
    'ht2_temp': HT2_TEMP_COUNT,   # HT2_Temp
    'ht2_time': HT2_TIME_COUNT,   # HT2_Time
    'ht2_cooling': COOLING_METHOD_COUNT,  # HT2冷却方式
}

# 最大动作空间（用于Q网络）
MAX_PROC_ACTION_SPACE = max(PROC_ACTIONS.values())  # 41

# ==================== 归一化参数 ====================

# Episode长度
EPISODE_COUNT_MAX = 30.0

# 金相信息归一化范围
PHASE_NORMALIZATION = {
    'mo_eq_min': -100.0,
    'mo_eq_max': 100.0,
    'al_eq_min': 0.0,
    'al_eq_max': 50.0,
    'beta_transform_T_min': 700.0,
    'beta_transform_T_max': 1200.0,
}

# ==================== 辅助函数 ====================

def get_proc_action_size(episode_count):
    """
    根据episode_count返回当前步骤的动作空间大小
    
    Args:
        episode_count: 当前步骤计数 (0-29)
    
    Returns:
        动作空间大小
    """
    if episode_count < 18:
        # 成分阶段：使用原有的动作空间
        from environment import ALL_ACTIONS_COUNT
        return ALL_ACTIONS_COUNT
    elif episode_count == 18:
        return PROC_ACTIONS['init_state']
    elif episode_count == 19:
        return PROC_ACTIONS['deform_decision']
    elif episode_count == 20:
        return PROC_ACTIONS['def_temp']
    elif episode_count == 21:
        return PROC_ACTIONS['def_strain']
    elif episode_count == 22:
        return PROC_ACTIONS['ht1_decision']
    elif episode_count == 23:
        return PROC_ACTIONS['ht1_temp']
    elif episode_count == 24:
        return PROC_ACTIONS['ht1_time']
    elif episode_count == 25:
        return PROC_ACTIONS['ht1_cooling']
    elif episode_count == 26:
        return PROC_ACTIONS['ht2_decision']
    elif episode_count == 27:
        return PROC_ACTIONS['ht2_temp']
    elif episode_count == 28:
        return PROC_ACTIONS['ht2_time']
    elif episode_count == 29:
        return PROC_ACTIONS['ht2_cooling']
    else:
        return 0  # Episode结束
