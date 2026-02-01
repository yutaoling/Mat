'''
    RL environment with a GPR as an internally maintained surrogate.
    The immediate rewards are from the surrogate. Only after the RL
    proposition will the surrogate be updated to capture the current
    relationship between the experimented xs and ys.
'''
from __future__ import annotations
from collections import namedtuple
import math
import random
from typing import List
from copy import deepcopy
import uuid
import warnings
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
import torch

from surroagate_train import get_model, device, N_PROP
from surrogate_model import N_PROC_BOOL, N_PROC_SCALAR, N_PHASE_SCALAR
from rl_proc_constants import (
    DEF_TEMP_CANDIDATES, DEF_TEMP_MAX, DEF_STRAIN_CANDIDATES, DEF_STRAIN_MAX,
    HT1_TEMP_CANDIDATES, HT1_TEMP_MAX, HT1_TIME_CANDIDATES, HT1_TIME_MAX,
    HT2_TEMP_CANDIDATES, HT2_TEMP_MAX, HT2_TIME_CANDIDATES, HT2_TIME_MAX,
    COOLING_METHODS, MAX_PROC_ACTION_SPACE, EPISODE_COUNT_MAX, PHASE_NORMALIZATION
)

''' botorch GPR part, if needed '''
# import torch
# import gpytorch
# from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.utils.sampling import manual_seed
# from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement

# Genaral data structure definition
Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))
TrainingIndicator = namedtuple('TrainingIndicator', ('epoch', 'loss', 'total_q'))
CompositionLimit = namedtuple('CompositionLimit', ('min_bound', 'max_bound'))

OUTPUT_BLOCK_SIZE = 100

''' class: State '''
# state.py
ActionType = float
REPLAY_MEMORY_PATH = 'replay_memory_buffer.pk'
MEMORY_CAPACITY = 3000
RESUME_MEMORY_BUFFER = False
COMPOSITION_INTERVAL = 0.001
COMPOSITION_ROUNDUP_DIGITS = 4
MANDATORY_MIN = 0.001

''' compositional bounds, NOTE USER PARAMETERS '''
# 按照 model_env.py 中 COMP 列表的顺序：Ti, Al, V, Cr, Mn, Fe, Cu, Zr, Nb, Mo, Sn, Hf, Ta, W, Si, C, N, O, Sc
TI_MIN, TI_MAX = 0.6, 1.0
AL_MIN, AL_MAX = 0., 0.1
V_MIN, V_MAX = 0., 0.1
CR_MIN, CR_MAX = 0., 0.15
MN_MIN, MN_MAX = 0., 0.0
FE_MIN, FE_MAX = 0., 0.1
CU_MIN, CU_MAX = 0., 0.0
ZR_MIN, ZR_MAX = 0., 0.50
NB_MIN, NB_MAX = 0., 0.35
MO_MIN, MO_MAX = 0., 0.20
SN_MIN, SN_MAX = 0., 0.15
HF_MIN, HF_MAX = 0., 0.0
TA_MIN, TA_MAX = 0., 0.5
W_MIN, W_MAX = 0., 0.0
SI_MIN, SI_MAX = 0., 0.0
C_MIN, C_MAX = 0., 0.0
N_MIN, N_MAX = 0., 0.0
O_MIN, O_MAX = 0., 0.0
SC_MIN, SC_MAX = 0., 0.0

COMP_LIMITS =(
    CompositionLimit(TI_MIN, TI_MAX),
    CompositionLimit(AL_MIN, AL_MAX),
    CompositionLimit(V_MIN, V_MAX),
    CompositionLimit(CR_MIN, CR_MAX),
    CompositionLimit(MN_MIN, MN_MAX),
    CompositionLimit(FE_MIN, FE_MAX),
    CompositionLimit(CU_MIN, CU_MAX),
    CompositionLimit(ZR_MIN, ZR_MAX),
    CompositionLimit(NB_MIN, NB_MAX),
    CompositionLimit(MO_MIN, MO_MAX),
    CompositionLimit(SN_MIN, SN_MAX),
    CompositionLimit(HF_MIN, HF_MAX),
    CompositionLimit(TA_MIN, TA_MAX),
    CompositionLimit(W_MIN, W_MAX),
    CompositionLimit(SI_MIN, SI_MAX),
    CompositionLimit(C_MIN, C_MAX),
    CompositionLimit(N_MIN, N_MAX),
    CompositionLimit(O_MIN, O_MAX),
    CompositionLimit(SC_MIN, SC_MAX),
)

COMP_MIN_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).min_bound
COMP_MAX_LIMITS = CompositionLimit(*zip(*COMP_LIMITS)).max_bound
# After computing COMP_MIN_LIMITS and COMP_MAX_LIMITS
NEW_COMP_MIN_LIMITS = []
NEW_COMP_MAX_LIMITS = []

for lo, hi in zip(COMP_MIN_LIMITS, COMP_MAX_LIMITS):
    if hi < MANDATORY_MIN:
        # This element can ONLY be 0
        NEW_COMP_MIN_LIMITS.append(0.0)
        NEW_COMP_MAX_LIMITS.append(0.0)
    elif lo>MANDATORY_MIN:
        NEW_COMP_MIN_LIMITS.append(lo)
        NEW_COMP_MAX_LIMITS.append(hi)
    elif lo>0.0:
        NEW_COMP_MIN_LIMITS.append(MANDATORY_MIN)
        NEW_COMP_MAX_LIMITS.append(hi)
    else:
        NEW_COMP_MIN_LIMITS.append(0.0)
        NEW_COMP_MAX_LIMITS.append(hi)


COMP_MIN_LIMITS = NEW_COMP_MIN_LIMITS
COMP_MAX_LIMITS = NEW_COMP_MAX_LIMITS


COMP_MULTIPLIER = 100.          # sum=1 -> sum=100.

ELEM_N = len(COMP_LIMITS)
COMP_EPISODE_LEN = ELEM_N - 1  # 成分阶段长度
MAX_EPISODE_LEN = 30  # 最大episode长度（成分18 + 工艺12）

EPSILON_START = 0.9
EPSILON_DECAY_COEF = 10000
EPSILON_END = 0.1
LEARNING_RATE = 1e-3            # Modification needed!
RL_TRAINING_EPOCHS = 1000       # Modification needed!
DEFAULT_LOG_INTERVAL = 1000     # terminal log every this epochs
RL_SAMPLE_BATCH_SIZE = 128
GAMMA = 0.80                    # TODO original 0.8
TARGET_UPDATE_PERIOD = 10
DQL_AGENT_PATH = 'dql_agent.pt'
DQL_TRAINING_INDICATOR_PATH = 'rl_agent_training_indicators.pk'
# composition tuning limits


COMP_LOW_BOUND_INT = round(min(COMP_MIN_LIMITS) / COMPOSITION_INTERVAL)
COMP_HIGH_BOUND_INT = round(max(COMP_MAX_LIMITS) / COMPOSITION_INTERVAL)
COMP_LOW_BOUND = COMP_LOW_BOUND_INT * COMPOSITION_INTERVAL
COMP_HIGH_BOUND = COMP_HIGH_BOUND_INT * COMPOSITION_INTERVAL
# action definition
ALL_ACTIONS = [0.0]
mandatory_idx = int(MANDATORY_MIN / COMPOSITION_INTERVAL)
for x in range(mandatory_idx, COMP_HIGH_BOUND_INT+1):
    ALL_ACTIONS.append(round(x * COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS))
ALL_ACTIONS = sorted(set(ALL_ACTIONS))
#ALL_ACTIONS=[round(x * COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS) \
#                for x in range(COMP_LOW_BOUND_INT, COMP_HIGH_BOUND_INT + 1)]
#ALL_ACTIONS.append(0.0)
ALL_ACTIONS_COUNT = len(ALL_ACTIONS)
ACTIONS_TO_INDEX_DICT = dict(zip(ALL_ACTIONS, range(ALL_ACTIONS_COUNT)))

ROUND_DIGIT = 5
STATE_DELIMETER_CHAR = '*'

# TODO move get_ground_truth_func and get_mo_ground_truth_func -> utils.py
def get_ground_truth_func(model_path = 'model\\model.pth', data_path = 'model\\data.pth'):
    '''
        Return the func that maps a composition -> a mechanical property (UTS / YS).
        保持向后兼容，仅使用成分（工艺参数设为默认值）
    '''
    from model_env import FCNN_Model
    # 抑制 sklearn 版本不匹配警告
    with warnings.catch_warnings():
        from sklearn.base import InconsistentVersionWarning
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        model, d, scalers = get_model(model=FCNN_Model(), default_model_pth=model_path, default_data_pth=data_path, resume=True, train=False)
    model.eval()    # NOTE important

    # 检查scalers数量
    if len(scalers) < 5:
        raise ValueError(f'scalers数量不足: 期望至少5个，实际{len(scalers)}个')
    
    comp_scaler = scalers[0]
    proc_bool_scaler = scalers[1]
    proc_scalar_scaler = scalers[2]
    phase_scalar_scaler = scalers[3]
    prop_scaler = scalers[4]
    elem_ft = d[-1]
    
    # 检查comp_scaler期望的特征数
    if hasattr(comp_scaler, 'n_features_in_'):
        if comp_scaler.n_features_in_ != ELEM_N:
            raise ValueError(f'成分scaler维度不匹配: 期望{ELEM_N}维，实际{comp_scaler.n_features_in_}维。'
                           f'这可能是因为使用了旧版本的模型文件。请使用新的模型文件或重新训练模型。')

    def _func(x):
        ''' maps a composition (a list with the sumation of 1.) to its mechanical property prediction '''
        x = np.array(x, dtype=np.float32)
        # 确保成分是19维
        if len(x) != ELEM_N:
            raise ValueError(f'成分维度错误: 期望{ELEM_N}维，实际{len(x)}维。成分: {x}')
        
        # 转换为百分比格式（如果输入是归一化的）
        x = (x * COMP_MULTIPLIER).round(ROUND_DIGIT)
        _comp = x.reshape(1, -1)
        
        # 确保维度匹配
        if hasattr(comp_scaler, 'n_features_in_'):
            expected_dim = comp_scaler.n_features_in_
            if _comp.shape[1] != expected_dim:
                # 如果维度不匹配，尝试修复
                if _comp.shape[1] == expected_dim + 1:
                    # 可能是多了一维，取前expected_dim维
                    _comp = _comp[:, :expected_dim]
                elif _comp.shape[1] == expected_dim - 1:
                    # 可能是少了一维，填充0
                    _comp = np.pad(_comp, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
                else:
                    raise ValueError(f'成分维度不匹配: scaler期望{expected_dim}维，实际{_comp.shape[1]}维')
        _comp = comp_scaler.transform(_comp)

        # 使用默认工艺参数（全0）
        _proc_bool = np.zeros((1, N_PROC_BOOL), dtype=np.float32)
        _proc_scalar = np.zeros((1, N_PROC_SCALAR), dtype=np.float32)
        # calculate_phase_scalar需要归一化格式（0-1），但x已经是百分比格式，需要转换回去
        x_normalized = x / COMP_MULTIPLIER  # 转换回归一化格式
        _phase_scalar = calculate_phase_scalar(x_normalized).reshape(1, -1)
        
        _proc_bool = proc_bool_scaler.transform(_proc_bool)
        _proc_scalar = proc_scalar_scaler.transform(_proc_scalar)
        _phase_scalar = phase_scalar_scaler.transform(_phase_scalar)

        # 使用实际的成分维度（经过scaler处理后的维度）
        comp_dim = _comp.shape[1]
        _comp = torch.tensor(_comp, dtype=torch.float32).reshape(1, 1, comp_dim, 1).to(device)
        elem_t = torch.tensor(elem_ft, dtype=torch.float32).reshape(1, 1, *(elem_ft.shape)).to(device)
        _proc_bool = torch.tensor(_proc_bool, dtype=torch.float32).reshape(1, 1, N_PROC_BOOL, 1).to(device)
        _proc_scalar = torch.tensor(_proc_scalar, dtype=torch.float32).reshape(1, 1, N_PROC_SCALAR, 1).to(device)
        _phase_scalar = torch.tensor(_phase_scalar, dtype=torch.float32).reshape(1, 1, N_PHASE_SCALAR, 1).to(device)

        _prop = model(_comp, elem_t, _proc_bool, _proc_scalar, _phase_scalar).detach().cpu().numpy()
        _prop = prop_scaler.inverse_transform(_prop)

        return _prop[0]  # 确保返回单个标量值而不是数组
    
    return _func

def get_ground_truth_func_with_proc(model_path = 'models/surrogate/model_multi.pth', data_path = 'models/surrogate/data_multi.pth'):
    '''
        Return the func that maps (composition, proc_bool, proc_scalar, phase_scalar) -> mechanical property.
        支持完整工艺参数输入
    '''
    from model_env import FCNN_Model
    # 抑制 sklearn 版本不匹配警告
    with warnings.catch_warnings():
        from sklearn.base import InconsistentVersionWarning
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        model, d, scalers = get_model(model=FCNN_Model(), default_model_pth=model_path, default_data_pth=data_path, resume=True, train=False)
    model.eval()

    # 检查scalers数量
    if len(scalers) < 5:
        raise ValueError(f'scalers数量不足: 期望至少5个，实际{len(scalers)}个')
    
    comp_scaler = scalers[0]
    proc_bool_scaler = scalers[1]
    proc_scalar_scaler = scalers[2]
    phase_scalar_scaler = scalers[3]
    prop_scaler = scalers[4]
    elem_ft = d[-1]
    
    # 检查comp_scaler期望的特征数
    if hasattr(comp_scaler, 'n_features_in_'):
        if comp_scaler.n_features_in_ != ELEM_N:
            raise ValueError(f'成分scaler维度不匹配: 期望{ELEM_N}维，实际{comp_scaler.n_features_in_}维。'
                           f'这可能是因为使用了旧版本的模型文件。请使用新的模型文件或重新训练模型。')

    def _func(comp, proc_bool, proc_scalar, phase_scalar):
        ''' 
        maps (composition, proc_bool, proc_scalar, phase_scalar) -> mechanical property prediction
        comp: list, 成分（归一化，和为1）
        proc_bool: np.array, 工艺布尔参数 (N_PROC_BOOL,)
        proc_scalar: np.array, 工艺标量参数 (N_PROC_SCALAR,)
        phase_scalar: np.array, 金相信息 (N_PHASE_SCALAR,)
        '''
        comp = np.array(comp, dtype=np.float32)
        # 确保成分是19维
        if len(comp) != ELEM_N:
            raise ValueError(f'成分维度错误: 期望{ELEM_N}维，实际{len(comp)}维。成分: {comp}')
        
        # 转换为百分比格式（如果输入是归一化的）
        comp = (comp * COMP_MULTIPLIER).round(ROUND_DIGIT)
        
        _comp = comp.reshape(1, -1)
        # 确保维度匹配
        if hasattr(comp_scaler, 'n_features_in_'):
            expected_dim = comp_scaler.n_features_in_
            if _comp.shape[1] != expected_dim:
                # 如果维度不匹配，尝试修复
                if _comp.shape[1] == expected_dim + 1:
                    # 可能是多了一维，取前expected_dim维
                    _comp = _comp[:, :expected_dim]
                elif _comp.shape[1] == expected_dim - 1:
                    # 可能是少了一维，填充0
                    _comp = np.pad(_comp, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
                else:
                    raise ValueError(f'成分维度不匹配: scaler期望{expected_dim}维，实际{_comp.shape[1]}维')
        _comp = comp_scaler.transform(_comp)
        
        _proc_bool = proc_bool.reshape(1, -1)
        _proc_scalar = proc_scalar.reshape(1, -1)
        _phase_scalar = phase_scalar.reshape(1, -1)
        
        _proc_bool = proc_bool_scaler.transform(_proc_bool)
        _proc_scalar = proc_scalar_scaler.transform(_proc_scalar)
        _phase_scalar = phase_scalar_scaler.transform(_phase_scalar)

        # 使用实际的成分维度（经过scaler处理后的维度）
        comp_dim = _comp.shape[1]
        _comp = torch.tensor(_comp, dtype=torch.float32).reshape(1, 1, comp_dim, 1).to(device)
        elem_t = torch.tensor(elem_ft, dtype=torch.float32).reshape(1, 1, *(elem_ft.shape)).to(device)
        _proc_bool = torch.tensor(_proc_bool, dtype=torch.float32).reshape(1, 1, N_PROC_BOOL, 1).to(device)
        _proc_scalar = torch.tensor(_proc_scalar, dtype=torch.float32).reshape(1, 1, N_PROC_SCALAR, 1).to(device)
        _phase_scalar = torch.tensor(_phase_scalar, dtype=torch.float32).reshape(1, 1, N_PHASE_SCALAR, 1).to(device)

        _prop = model(_comp, elem_t, _proc_bool, _proc_scalar, _phase_scalar).detach().cpu().numpy()
        _prop = prop_scaler.inverse_transform(_prop)

        return _prop[0]
    
    return _func

def calculate_phase_scalar(composition):
    """
    根据成分计算金相信息
    成分单位：原子分数（at fraction），范围0-1，需要转换为百分比
    
    COMP列表索引：['Ti', 'Al', 'V', 'Cr', 'Mn', 'Fe', 'Cu', 'Zr', 'Nb', 'Mo', 
                   'Sn', 'Hf', 'Ta', 'W', 'Si', 'C', 'N', 'O', 'Sc']
    索引：         0     1     2    3     4     5     6     7     8     9     10    11    12    13   14    15   16   17   18
    """
    comp = np.array(composition) * 100.0  # 转换为百分比
    
    # Mo当量 = 1.0*Mo + 0.67*V + 0.44*W + 0.28*Nb + 0.22*Ta + 
    #          2.9*Fe + 1.6*Cr + 1.7*Mn - 1.0*Al
    # 注意：COMP中没有Ni和Co，相关项为0
    mo_eq = (1.0 * comp[9] +      # Mo (索引9)
             0.67 * comp[2] +     # V (索引2)
             0.44 * comp[13] +    # W (索引13)
             0.28 * comp[8] +     # Nb (索引8)
             0.22 * comp[12] +    # Ta (索引12)
             2.9 * comp[5] +      # Fe (索引5)
             1.6 * comp[3] +      # Cr (索引3)
             1.7 * comp[4] -      # Mn (索引4)
             1.0 * comp[1])      # Al (索引1)
    
    # Al当量 = 1.0*Al + Zr/6 + Sn/3 + 10*(O + N)
    al_eq = (1.0 * comp[1] +           # Al (索引1)
             comp[7] / 6.0 +           # Zr (索引7)
             comp[10] / 3.0 +          # Sn (索引10)
             10.0 * (comp[17] + comp[16]))  # O (索引17) + N (索引16)
    
    # beta转变温度 = 882 + 2.1*Al - 9.5*Mo + 4.2*Sn - 6.9*Zr - 
    #                11.8*V - 12.1*Cr - 15.4*Fe + 23.3*Si + 123.0*O
    beta_transform_T = (882.0 +
                        2.1 * comp[1] -      # Al
                        9.5 * comp[9] +      # Mo
                        4.2 * comp[10] -     # Sn
                        6.9 * comp[7] -       # Zr
                        11.8 * comp[2] -     # V
                        12.1 * comp[3] -     # Cr
                        15.4 * comp[5] +     # Fe
                        23.3 * comp[14] +    # Si
                        123.0 * comp[17])    # O
    
    return np.array([mo_eq, al_eq, beta_transform_T], dtype=np.float32)

def get_mo_ground_truth_func():
    '''
        Build a linearly compounded multi-objective (MO) 'property' predicting function.
        支持完整工艺参数输入
    '''
    _func = get_ground_truth_func_with_proc('models/surrogate/model_multi.pth', 'models/surrogate/data_multi.pth')

    ''' local optimal maximums for YS, UTS, and ELongation '''
    _mo_scale = np.array([
        70,
        813,
        932,
        13.5,
        273,
    ])

    def _mo_raw_func(comp, proc_bool, proc_scalar, phase_scalar):
        """
            返回原始多目标向量（未加权）的预测值，形如 (n_props,)
        """
        return np.array(_func(comp, proc_bool, proc_scalar, phase_scalar))

    # 返回原始向量函数以及用于归一化的尺度向量
    return _mo_raw_func, _mo_scale

class State:
    def __init__(self, if_init: bool = False, 
                 previous_state: State = None, 
                 action: ActionType = None, 
                 action_idx: int = None,  # 新增：动作索引（用于工艺阶段）
                 episode_len = COMP_EPISODE_LEN):
        if if_init:
            # 初始化成分
            self.__composition = [0.] * ELEM_N
            self.__episode_count = -1
            self.__max_episode_len = MAX_EPISODE_LEN
            
            # 初始化工艺参数
            self.__proc_bool = np.zeros(N_PROC_BOOL, dtype=np.float32)  # 8维
            self.__proc_scalar = np.zeros(N_PROC_SCALAR, dtype=np.float32)  # 6维
            
            # 工艺状态标志
            self.__is_wrought = 0  # 0或1
            self.__deform_decision = 0  # 0=不变形, 1=变形
            self.__ht1_decision = 0  # 0=不进行, 1=进行
            self.__ht2_decision = 0  # 0=不进行, 1=进行
        else:
            # 复制前一个状态
            self.__composition = deepcopy(previous_state.get_composition())
            self.__episode_count = previous_state.get_episode_count() + 1
            self.__max_episode_len = MAX_EPISODE_LEN
            
            # 复制工艺参数
            self.__proc_bool = deepcopy(previous_state.get_proc_bool())
            self.__proc_scalar = deepcopy(previous_state.get_proc_scalar())
            self.__is_wrought = previous_state.get_is_wrought()
            self.__deform_decision = previous_state.get_deform_decision()
            self.__ht1_decision = previous_state.get_ht1_decision()
            self.__ht2_decision = previous_state.get_ht2_decision()
            
            # 根据episode_count决定是成分阶段还是工艺阶段
            if self.__episode_count < COMP_EPISODE_LEN:
                # 成分阶段：沿用原有逻辑
                sub_idx = self.__episode_count
                _min = max(COMP_MIN_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]) - sum(COMP_MAX_LIMITS[sub_idx + 1:]))
                _max = min(COMP_MAX_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]))
                _min, _max = round(_min, COMPOSITION_ROUNDUP_DIGITS), round(_max, COMPOSITION_ROUNDUP_DIGITS)
                action = min(max(action, _min), _max)
                self.__composition[sub_idx] = action

                if sub_idx == COMP_EPISODE_LEN - 1:
                    last = 1. - sum(self.__composition[:-1])
                    if 0 < last < MANDATORY_MIN:
                        last = 0.0
                    self.__composition[-1] = round(last, COMPOSITION_ROUNDUP_DIGITS)

                # round up compositions
                for idx in range(len(self.__composition)):
                    self.__composition[idx] = round(self.__composition[idx], COMPOSITION_ROUNDUP_DIGITS)
            else:
                # 工艺阶段：根据action_idx更新工艺参数
                self._update_proc_params(action_idx)

    def get_episode_count(self) -> int:
        return self.__episode_count
    
    def get_composition(self):
        return self.__composition
    
    def get_proc_bool(self):
        return self.__proc_bool.copy()
    
    def get_proc_scalar(self):
        return self.__proc_scalar.copy()
    
    def get_is_wrought(self):
        return self.__is_wrought
    
    def get_deform_decision(self):
        return self.__deform_decision
    
    def get_ht1_decision(self):
        return self.__ht1_decision
    
    def get_ht2_decision(self):
        return self.__ht2_decision
    
    def _update_proc_params(self, action_idx):
        """根据episode_count和action_idx更新工艺参数"""
        ep = self.__episode_count
        
        if ep == 18:  # 初始状态选择
            if action_idx == 0:
                self.__proc_bool[0] = 1.0  # Is_Not_Wrought
                self.__proc_bool[1] = 0.0  # Is_Wrought
                self.__is_wrought = 0
            else:
                self.__proc_bool[0] = 0.0
                self.__proc_bool[1] = 1.0
                self.__is_wrought = 1
        
        elif ep == 19:  # 变形决策
            if self.__is_wrought == 1:
                self.__deform_decision = action_idx
                if action_idx == 0:  # 不变形
                    self.__proc_scalar[0] = 0.0  # Def_Temp
                    self.__proc_scalar[1] = 0.0  # Def_Strain
        
        elif ep == 20:  # Def_Temp
            if self.__is_wrought == 1 and self.__deform_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(DEF_TEMP_CANDIDATES):
                    self.__proc_scalar[0] = DEF_TEMP_CANDIDATES[action_idx]
                else:
                    # 如果超出范围，使用最后一个候选值
                    self.__proc_scalar[0] = DEF_TEMP_CANDIDATES[-1]
        
        elif ep == 21:  # Def_Strain
            if self.__is_wrought == 1 and self.__deform_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(DEF_STRAIN_CANDIDATES):
                    self.__proc_scalar[1] = DEF_STRAIN_CANDIDATES[action_idx]
                else:
                    # 如果超出范围，使用最后一个候选值
                    self.__proc_scalar[1] = DEF_STRAIN_CANDIDATES[-1]
        
        elif ep == 22:  # HT1决策
            self.__ht1_decision = action_idx
            if action_idx == 0:  # 不进行HT1
                self.__proc_scalar[2] = 0.0  # HT1_Temp
                self.__proc_scalar[3] = 0.0  # HT1_Time
                self.__proc_bool[2:5] = 0.0  # HT1冷却方式
        
        elif ep == 23:  # HT1_Temp
            if self.__ht1_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(HT1_TEMP_CANDIDATES):
                    self.__proc_scalar[2] = HT1_TEMP_CANDIDATES[action_idx]
                else:
                    self.__proc_scalar[2] = HT1_TEMP_CANDIDATES[-1]
        
        elif ep == 24:  # HT1_Time
            if self.__ht1_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(HT1_TIME_CANDIDATES):
                    self.__proc_scalar[3] = HT1_TIME_CANDIDATES[action_idx]
                else:
                    self.__proc_scalar[3] = HT1_TIME_CANDIDATES[-1]
        
        elif ep == 25:  # HT1_冷却方式
            if self.__ht1_decision == 1:
                self.__proc_bool[2:5] = 0.0
                # 确保action_idx在有效范围内（0-2对应3种冷却方式）
                if 0 <= action_idx <= 2:
                    self.__proc_bool[2 + action_idx] = 1.0
                else:
                    self.__proc_bool[2] = 1.0  # 默认使用第一种冷却方式
        
        elif ep == 26:  # HT2决策
            if self.__ht1_decision == 1:  # 只有HT1完成后才能进行HT2
                self.__ht2_decision = action_idx
                if action_idx == 0:  # 不进行HT2
                    self.__proc_scalar[4] = 0.0  # HT2_Temp
                    self.__proc_scalar[5] = 0.0  # HT2_Time
                    self.__proc_bool[5:8] = 0.0  # HT2冷却方式
        
        elif ep == 27:  # HT2_Temp
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(HT2_TEMP_CANDIDATES):
                    self.__proc_scalar[4] = HT2_TEMP_CANDIDATES[action_idx]
                else:
                    self.__proc_scalar[4] = HT2_TEMP_CANDIDATES[-1]
        
        elif ep == 28:  # HT2_Time
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                # 确保action_idx在有效范围内
                if action_idx < len(HT2_TIME_CANDIDATES):
                    self.__proc_scalar[5] = HT2_TIME_CANDIDATES[action_idx]
                else:
                    self.__proc_scalar[5] = HT2_TIME_CANDIDATES[-1]
        
        elif ep == 29:  # HT2_冷却方式
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_bool[5:8] = 0.0
                # 确保action_idx在有效范围内（0-2对应3种冷却方式）
                if 0 <= action_idx <= 2:
                    self.__proc_bool[5 + action_idx] = 1.0
                else:
                    self.__proc_bool[5] = 1.0  # 默认使用第一种冷却方式

    def repr(self):
        """
        返回归一化后的状态向量（37维）
        [成分(19) + episode_count(1) + 工艺布尔(8) + 工艺标量(6) + 金相(3)]
        """
        # 1. 成分部分（19维）：保持不变
        state = deepcopy(self.__composition)
        
        # 2. episode_count（1维）：归一化
        state.append(self.__episode_count / EPISODE_COUNT_MAX)
        
        # 3. 工艺布尔参数（8维）：保持不变
        state.extend(self.__proc_bool.tolist())
        
        # 4. 工艺标量参数（6维）：对数归一化（温度、时间）或线性归一化（strain）
        # Def_Temp: 对数归一化
        if self.__proc_scalar[0] > 0:
            state.append(np.log(self.__proc_scalar[0] + 1) / np.log(DEF_TEMP_MAX + 1))
        else:
            state.append(0.0)
        
        # Def_Strain: 线性归一化
        state.append(self.__proc_scalar[1] / DEF_STRAIN_MAX)
        
        # HT1_Temp: 对数归一化
        if self.__proc_scalar[2] > 0:
            state.append(np.log(self.__proc_scalar[2] + 1) / np.log(HT1_TEMP_MAX + 1))
        else:
            state.append(0.0)
        
        # HT1_Time: 对数归一化
        if self.__proc_scalar[3] > 0:
            state.append(np.log(self.__proc_scalar[3] + 1) / np.log(HT1_TIME_MAX + 1))
        else:
            state.append(0.0)
        
        # HT2_Temp: 对数归一化
        if self.__proc_scalar[4] > 0:
            state.append(np.log(self.__proc_scalar[4] + 1) / np.log(HT2_TEMP_MAX + 1))
        else:
            state.append(0.0)
        
        # HT2_Time: 对数归一化
        if self.__proc_scalar[5] > 0:
            state.append(np.log(self.__proc_scalar[5] + 1) / np.log(HT2_TIME_MAX + 1))
        else:
            state.append(0.0)
        
        # 5. 金相信息（3维）：Min-Max归一化
        phase_scalar = calculate_phase_scalar(self.__composition)
        mo_eq_norm = (phase_scalar[0] - PHASE_NORMALIZATION['mo_eq_min']) / \
                     (PHASE_NORMALIZATION['mo_eq_max'] - PHASE_NORMALIZATION['mo_eq_min'])
        al_eq_norm = (phase_scalar[1] - PHASE_NORMALIZATION['al_eq_min']) / \
                     (PHASE_NORMALIZATION['al_eq_max'] - PHASE_NORMALIZATION['al_eq_min'])
        beta_T_norm = (phase_scalar[2] - PHASE_NORMALIZATION['beta_transform_T_min']) / \
                      (PHASE_NORMALIZATION['beta_transform_T_max'] - PHASE_NORMALIZATION['beta_transform_T_min'])
        
        # 裁剪到[0,1]
        mo_eq_norm = np.clip(mo_eq_norm, 0.0, 1.0)
        al_eq_norm = np.clip(al_eq_norm, 0.0, 1.0)
        beta_T_norm = np.clip(beta_T_norm, 0.0, 1.0)
        
        state.extend([mo_eq_norm, al_eq_norm, beta_T_norm])
        
        return np.array(state, dtype=np.float32)

    def done(self):
        """判断episode是否结束"""
        ep = self.__episode_count
        
        # 安全检查：如果超过最大episode长度，强制结束
        if ep >= self.__max_episode_len:
            return True
        
        # 成分阶段：未完成
        if ep < COMP_EPISODE_LEN - 1:
            return False
        
        # 成分阶段最后一步：还需要进入工艺阶段
        if ep == COMP_EPISODE_LEN - 1:
            return False
        
        # 工艺阶段开始
        # 步骤18：初始状态选择 - 未完成
        if ep == 18:
            return False
        
        # 步骤19：变形决策
        if ep == 19:
            if self.__is_wrought == 1:
                return False  # 需要决定是否变形
            else:
                return False  # Is_Wrought=0，跳过变形，继续HT1决策
        
        # 步骤20-21：变形参数（仅当Is_Wrought=1且选择变形时）
        if self.__is_wrought == 1:
            if ep == 20:  # Def_Temp
                return False
            if ep == 21:  # Def_Strain
                return False  # 变形完成，继续HT1
        
        # 步骤22：HT1决策
        if ep == 22:
            return False
        
        # 步骤23-25：HT1参数（仅当选择进行HT1时）
        if self.__ht1_decision == 1:
            if ep == 23:  # HT1_Temp
                return False
            if ep == 24:  # HT1_Time
                return False
            if ep == 25:  # HT1_冷却方式
                return False  # HT1完成，继续HT2
        
        # 如果选择不进行HT1，episode在22步结束（但需要先完成HT1决策）
        # 注意：这里ep==22时，ht1_decision已经设置，但还需要检查
        if ep == 22 and self.__ht1_decision == 0:
            return True
        
        # 步骤26：HT2决策（仅当HT1完成时）
        if ep == 26:
            if self.__ht1_decision == 1:
                return False  # 需要决定是否进行HT2
            else:
                return True  # 没有HT1，episode结束
        
        # 步骤27-29：HT2参数（仅当选择进行HT2时）
        if self.__ht1_decision == 1 and self.__ht2_decision == 1:
            if ep == 27:  # HT2_Temp
                return False
            if ep == 28:  # HT2_Time
                return False
            if ep == 29:  # HT2_冷却方式
                return True  # HT2完成，episode结束
        
        # 如果选择不进行HT2，episode在26步结束
        if self.__ht1_decision == 1 and self.__ht2_decision == 0 and ep == 26:
            return True
        
        # 其他情况：未完成
        return False

    def get_action_idx_limits(self):
        '''
            Get action limits according to current state.
            成分阶段：返回(composition_min_idx, composition_max_idx)
            工艺阶段：返回(0, action_space_size-1)
        '''
        if self.__episode_count < COMP_EPISODE_LEN:
            # 成分阶段：沿用原有逻辑
            # 注意：episode_count 从 0 开始，对应元素索引也是从 0 开始
            elem_index = self.__episode_count
            assert elem_index < ELEM_N, f'elem_index: {elem_index}, ELEM_N: {ELEM_N}, COMP_EPISODE_LEN: {COMP_EPISODE_LEN}'
            _min = max(COMP_MIN_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]) - sum(COMP_MAX_LIMITS[elem_index + 1:]))
            _max = min(COMP_MAX_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]) - sum(COMP_MIN_LIMITS[elem_index + 1:]))

            _min, _max = round(_min, COMPOSITION_ROUNDUP_DIGITS), round(_max, COMPOSITION_ROUNDUP_DIGITS)
            if _max < MANDATORY_MIN:
                _max = 0.0
                _min = 0.0
            elif 0 < _min < MANDATORY_MIN:
                _min = MANDATORY_MIN

            comp_min_idx = ACTIONS_TO_INDEX_DICT[round(_min, COMPOSITION_ROUNDUP_DIGITS)]
            comp_max_idx = ACTIONS_TO_INDEX_DICT[round(_max, COMPOSITION_ROUNDUP_DIGITS)]
            return comp_min_idx, comp_max_idx
        else:
            # 工艺阶段：返回动作掩码对应的范围
            mask = self.get_action_mask()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                return valid_actions[0], valid_actions[-1]
            else:
                return 0, 0
    
    def get_action_mask(self):
        """
        返回动作掩码（布尔数组），True表示允许，False表示禁止
        用于工艺阶段屏蔽无效动作
        """
        from rl_proc_constants import MAX_PROC_ACTION_SPACE
        
        # 成分阶段：使用最大动作空间，通过get_action_idx_limits限制
        if self.__episode_count < COMP_EPISODE_LEN:
            mask = np.zeros(ALL_ACTIONS_COUNT, dtype=bool)
            ai_low, ai_high = self.get_action_idx_limits()
            mask[ai_low:ai_high+1] = True
            return mask
        
        # 工艺阶段：根据当前状态返回掩码
        ep = self.__episode_count
        # 使用统一的最大动作空间
        max_action_space = max(ALL_ACTIONS_COUNT, MAX_PROC_ACTION_SPACE)
        mask = np.zeros(max_action_space, dtype=bool)
        
        if ep == 18:  # 初始状态选择
            mask[0:2] = True
        elif ep == 19:  # 变形决策
            if self.__is_wrought == 1:
                mask[0:2] = True
            else:
                mask[0] = True  # 只能选择不变形（实际上应该跳过）
        elif ep == 20:  # Def_Temp
            if self.__is_wrought == 1 and self.__deform_decision == 1:
                mask[0:len(DEF_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True  # 只能选择0
        elif ep == 21:  # Def_Strain
            if self.__is_wrought == 1 and self.__deform_decision == 1:
                mask[0:len(DEF_STRAIN_CANDIDATES)] = True
            else:
                mask[0] = True
        elif ep == 22:  # HT1决策
            mask[0:2] = True
        elif ep == 23:  # HT1_Temp
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif ep == 24:  # HT1_Time
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif ep == 25:  # HT1_冷却方式
            if self.__ht1_decision == 1:
                mask[0:3] = True
            else:
                mask[0] = True
        elif ep == 26:  # HT2决策
            if self.__ht1_decision == 1:
                mask[0:2] = True
            else:
                mask[0] = True  # 只能选择不进行
        elif ep == 27:  # HT2_Temp
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif ep == 28:  # HT2_Time
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif ep == 29:  # HT2_冷却方式
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:3] = True
            else:
                mask[0] = True
        else:
            # Episode结束
            pass
        
        return mask

    def generate_random_action(self, random_seed = None) -> ActionType:
        '''
            Generate one random action that can be applied to this state

            @output:    a random action in float
            
            Args:
                random_seed: 可以是RandomState对象、整数种子或None
        '''
        # 如果已经是RandomState对象，直接使用；否则创建新的
        if isinstance(random_seed, np.random.RandomState):
            random_state = random_seed
        else:
            random_state = ensure_rng(random_seed)
        
        comp_min_idx, comp_max_idx = self.get_action_idx_limits()
        if comp_max_idx < comp_min_idx:
            # 如果没有有效动作，返回0（最小动作）
            return ALL_ACTIONS[0]
        rand_comp_idx = random_state.randint(comp_min_idx, comp_max_idx + 1)
        return ALL_ACTIONS[rand_comp_idx]
    
    @staticmethod
    def encode_key(x):
        # 先对成分进行四舍五入，确保编码的一致性
        x_rounded = [round(val, COMPOSITION_ROUNDUP_DIGITS) for val in x]
        return STATE_DELIMETER_CHAR.join(map(str, x_rounded))
    
    @staticmethod
    def decode_key(key: str):
        return np.array(list(map(float, key.split(STATE_DELIMETER_CHAR)))).round(ROUND_DIGIT)   # check return data type

class Environment:
    def __init__(self, 
                 init_N = 50, 
                 random_seed = None):
        self.init_world_model()

        # self.all_actions = np.linspace(self.x_min, self.x_max, self.act_dim).round(ROUND_DIGIT).tolist()
        self.state_dim = len(self.reset().repr())  # 37维
        # 动作空间：使用最大动作空间（成分阶段和工艺阶段的最大值）
        from rl_proc_constants import MAX_PROC_ACTION_SPACE
        self.act_dim = max(ALL_ACTIONS_COUNT, MAX_PROC_ACTION_SPACE)

        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None

        '''
            "The answer to the ultimate question of life, 
            the universe, and everything is 42."
        '''
        self._random_state = 42
        self.surrogate = None
        self.surrogate_buffer = set()   # stores experimented xs
        self.surrogate_buffer_list = []
        self.cached_surrogate_pred_dict = dict()

        self.init_N = init_N
        self.init_surrogate(self.init_N, random_seed)
    
    def init_world_model(self,):
        # 获取返回的原始多目标函数及其尺度
        raw_func, mo_scale = get_mo_ground_truth_func()
        self.raw_mo_func_with_proc = raw_func
        self.mo_scale = np.array(mo_scale)
        # 初始权重：相等权重
        n_props = len(self.mo_scale)
        self.mo_weights = np.ones(n_props, dtype=float) / float(n_props)
        # self.func_with_proc 返回标量（按当前权重将向量合并）
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: \
            float(np.dot(self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar) / self.mo_scale, self.mo_weights))
        # 保持向后兼容
        self.raw_mo_func = lambda x: self.raw_mo_func_with_proc(
            x, 
            np.zeros(N_PROC_BOOL, dtype=np.float32),
            np.zeros(N_PROC_SCALAR, dtype=np.float32),
            calculate_phase_scalar(x)
        )
        self.func = lambda x: self.func_with_proc(
            x,
            np.zeros(N_PROC_BOOL, dtype=np.float32),
            np.zeros(N_PROC_SCALAR, dtype=np.float32),
            calculate_phase_scalar(x)
        )

    def set_mo_weights(self, weights):
        """Set multi-objective weights. `weights` should be array-like of length n_props.
        Weights will be normalized to sum to 1.
        """
        w = np.array(weights, dtype=float)
        assert w.ndim == 1 and len(w) == len(self.mo_scale), 'weights length mismatch'
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.mo_weights = w / w.sum()
        # update scalar func
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: \
            float(np.dot(self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar) / self.mo_scale, self.mo_weights))
        self.func = lambda x: self.func_with_proc(
            x,
            np.zeros(N_PROC_BOOL, dtype=np.float32),
            np.zeros(N_PROC_SCALAR, dtype=np.float32),
            calculate_phase_scalar(x)
        )

    def update_mo_weights(self, method='improvement', window=20, eps=1e-6):
        """Auto-update weights based on information in `surrogate_buffer_list`.

        method options:
        - 'improvement': weight objectives by their recent improvement magnitudes.
        - 'variance': weight by predictive variance (not implemented here).

        This implementation computes the difference between the best value in the
        last `window` experiments and the best value in the previous `window`.
        Larger improvements receive larger weights.
        """
        if len(self.surrogate_buffer_list) < 2:
            return
        n = len(self.surrogate_buffer_list)
        w = np.ones(len(self.mo_scale), dtype=float)
        if method == 'improvement':
            # evaluate raw properties for the buffer
            # 处理字典格式的数据（包含 'comp', 'proc_bool', 'proc_scalar', 'phase'）
            vals_list = []
            for x in self.surrogate_buffer_list:
                if isinstance(x, dict):
                    # 新格式：字典
                    val = self.raw_mo_func_with_proc(
                        x['comp'], x['proc_bool'], x['proc_scalar'], x['phase']
                    )
                else:
                    # 旧格式：成分数组（向后兼容）
                    val = self.raw_mo_func(x)
                vals_list.append(val)
            vals = np.array(vals_list)  # shape (n_samples, n_props)
            # divide into two windows
            k = max(1, min(window, n // 2))
            recent = vals[-k:]
            prev = vals[-2*k:-k] if n >= 2*k else vals[:k]
            recent_mean = np.mean(recent / self.mo_scale, axis=0)
            prev_mean = np.mean(prev / self.mo_scale, axis=0)
            imp = recent_mean - prev_mean
            rank = np.argsort(np.argsort(imp))
            Weight_List=[1, 1, 1, 1, 1] #TODO:dynamic weight[6, 5, 4, 3, 2]
            w = np.array([Weight_List[r] for r in rank])
            w=w / w.sum()
        # normalize and set
        self.set_mo_weights(w)

    def copy_initialization(self, env: Environment):
        ''' copy the randomly initialized exp point from a ref environment instance '''
        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None

        self.surrogate_buffer = deepcopy(env.surrogate_buffer)
        self.surrogate_buffer_list = deepcopy(env.surrogate_buffer_list)

        # train the model using copied initial exp points
        self.update_surrogate()

    def _generate_random_composition(self, random_state):
        '''直接随机生成成分（考虑约束，确保满足最小值要求）'''
        # 首先确保满足所有最小约束
        comp = [COMP_MIN_LIMITS[i] for i in range(ELEM_N)]
        min_sum = sum(comp)
        
        # 如果最小值的和已经>=1，无法生成有效成分
        if min_sum >= 1.0:
            # 使用默认值（主要是Ti）
            comp = [0.0] * ELEM_N
            comp[0] = 1.0  # Ti设为1
            return comp
        
        # 在剩余空间内随机分配
        remaining = 1.0 - min_sum
        
        # 为每个元素在剩余空间内随机分配额外的值
        # 但不超过最大值
        for i in range(ELEM_N):
            max_additional = min(COMP_MAX_LIMITS[i] - COMP_MIN_LIMITS[i], remaining)
            if max_additional > 0:
                additional = random_state.uniform(0, max_additional)
                comp[i] += additional
                remaining -= additional
                if remaining <= 0:
                    break
        
        # 如果还有剩余空间，随机分配给任意元素（不超过最大值）
        if remaining > 0:
            attempts = 0
            while remaining > 1e-6 and attempts < 100:
                i = random_state.randint(0, ELEM_N)
                max_additional = COMP_MAX_LIMITS[i] - comp[i]
                if max_additional > 0:
                    add = min(remaining, max_additional)
                    comp[i] += add
                    remaining -= add
                attempts += 1
        
        # 最终归一化（确保和为1）
        comp_sum = sum(comp)
        if comp_sum > 0:
            comp = [val / comp_sum for val in comp]
        else:
            comp = [0.0] * ELEM_N
            comp[0] = 1.0
        
        return comp
    
    def _normalize_and_constrain_composition(self, comp):
        '''归一化并约束成分值（迭代方法确保满足约束）'''
        # 先确保值在约束范围内
        comp = [max(COMP_MIN_LIMITS[i], min(COMP_MAX_LIMITS[i], val)) for i, val in enumerate(comp)]
        
        # 归一化
        comp_sum = sum(comp)
        if comp_sum > 0:
            comp = [val / comp_sum for val in comp]
        else:
            # 如果所有值都是0，使用默认值（主要是Ti）
            comp = [0.0] * ELEM_N
            comp[0] = 1.0  # Ti设为1
            return comp
        
        # 迭代调整，确保满足约束（最多迭代20次）
        for iteration in range(20):
            # 检查并修正超出约束的值
            adjusted = False
            for i in range(ELEM_N):
                if comp[i] < COMP_MIN_LIMITS[i]:
                    comp[i] = COMP_MIN_LIMITS[i]
                    adjusted = True
                elif comp[i] > COMP_MAX_LIMITS[i]:
                    comp[i] = COMP_MAX_LIMITS[i]
                    adjusted = True
            
            if not adjusted:
                break
            
            # 重新归一化
            comp_sum = sum(comp)
            if comp_sum > 0:
                comp = [val / comp_sum for val in comp]
            else:
                # 如果所有值都是0，使用默认值
                comp = [0.0] * ELEM_N
                comp[0] = 1.0  # Ti设为1
                break
        
        return comp
    
    def _check_composition_constraints(self, comp, tolerance=1e-5):
        '''检查成分是否满足约束（允许小的浮点误差）'''
        # 检查每个元素是否在约束范围内（允许小的浮点误差）
        for i, val in enumerate(comp):
            # 允许小的浮点误差
            if val < COMP_MIN_LIMITS[i] - tolerance or val > COMP_MAX_LIMITS[i] + tolerance:
                return False
        return True

    def init_surrogate(self, init_N, seed):
        ''' initialize surrogate with init_N randomly generated samples '''
        __counter, __max_acc_count = 0, int(1e5)  # 增加最大尝试次数
        random_state = ensure_rng(seed)
        unique_attempts = 0  # 记录成功生成成分的次数
        seen_compositions = set()  # 用于调试：记录所有见过的成分（不四舍五入）
        
        # 使用更简单的策略：直接随机生成成分，然后归一化
        # 这样可以确保生成多样化的成分
        while len(self.surrogate_buffer) < init_N:
            # 方法1：尝试使用State生成（如果失败，使用方法2）
            if __counter % 2 == 0:  # 交替使用两种方法
                _state = State(if_init = True)
                success = True
                
                # 成分阶段：生成 COMP_EPISODE_LEN 个元素（最后一个元素自动计算）
                for step in range(COMP_EPISODE_LEN):
                    try:
                        comp_min_idx, comp_max_idx = _state.get_action_idx_limits()
                        if comp_max_idx < comp_min_idx:
                            # 如果没有有效动作，使用默认值
                            action = 0.0
                        else:
                            action = _state.generate_random_action(random_state)
                        _state = State(previous_state = _state, action = action)
                    except (AssertionError, ValueError, IndexError, KeyError) as e:
                        # 如果生成动作失败，重新开始
                        success = False
                        break
                
                if success:
                    _x = _state.get_composition()
                else:
                    _x = None
            else:
                # 方法2：直接随机生成成分（更简单，确保多样性）
                _x = self._generate_random_composition(random_state)
            
            if _x is None:
                __counter += 1
                if __counter >= __max_acc_count:
                    raise RuntimeError(f'无法生成足够的唯一样本。已生成: {len(self.surrogate_buffer)}/{init_N}，尝试次数: {__counter}，成功生成成分次数: {unique_attempts}')
                continue
            
            unique_attempts += 1
            
            # 验证成分是否有效（和为1，且在合理范围内）
            comp_sum = sum(_x)
            if abs(comp_sum - 1.0) > 1e-4:  # 稍微放宽容差
                # 成分和不为1，尝试归一化
                if comp_sum > 0:
                    _x = [val / comp_sum for val in _x]
                    comp_sum = sum(_x)
                
                if abs(comp_sum - 1.0) > 1e-4:
                    # 归一化后仍然无效，跳过
                    __counter += 1
                    if __counter >= __max_acc_count:
                        raise RuntimeError(f'无法生成有效的成分样本。最后尝试的成分和: {comp_sum}, 成分: {_x}')
                    continue
            
            # 确保成分值在合理范围内并满足约束
            _x = self._normalize_and_constrain_composition(_x)
            
            # 检查成分是否满足约束（允许小的浮点误差）
            if not self._check_composition_constraints(_x, tolerance=1e-4):
                # 如果约束检查失败，尝试再次调整
                _x = self._normalize_and_constrain_composition(_x)
                if not self._check_composition_constraints(_x, tolerance=1e-3):
                    # 如果仍然失败，跳过这个样本
                    __counter += 1
                    if __counter >= __max_acc_count:
                        # 输出详细的约束违反信息
                        violations = []
                        for i, val in enumerate(_x):
                            if val < COMP_MIN_LIMITS[i] - 1e-3:
                                violations.append(f'元素{i}: {val:.6f} < {COMP_MIN_LIMITS[i]:.6f}')
                            elif val > COMP_MAX_LIMITS[i] + 1e-3:
                                violations.append(f'元素{i}: {val:.6f} > {COMP_MAX_LIMITS[i]:.6f}')
                        raise RuntimeError(f'无法生成满足约束的成分样本。违反约束: {violations[:5]}')
                    continue
            
            # 确保成分值在合理范围内
            _x = [round(val, COMPOSITION_ROUNDUP_DIGITS) for val in _x]
            
            # 用于调试：检查成分是否真的不同
            _x_raw_key = tuple(_x)  # 不四舍五入的key，用于调试
            if _x_raw_key not in seen_compositions:
                seen_compositions.add(_x_raw_key)
                if len(seen_compositions) <= 5:  # 只打印前5个不同的成分
                    print(f'调试：生成的不同成分 #{len(seen_compositions)}: sum={sum(_x):.6f}, 前5个元素={_x[:5]}')
            
            _x_key = State.encode_key(_x)
            if _x_key not in self.surrogate_buffer:
                self.surrogate_buffer.add(_x_key)
                self.surrogate_buffer_list.append({
                    'comp': _x,
                    'proc_bool': np.zeros(N_PROC_BOOL, dtype=np.float32),
                    'proc_scalar': np.zeros(N_PROC_SCALAR, dtype=np.float32),
                    'phase': calculate_phase_scalar(_x)
                })
                # 每生成10个样本输出一次进度
                if len(self.surrogate_buffer) % 10 == 0:
                    print(f'已生成 {len(self.surrogate_buffer)}/{init_N} 个唯一样本')

            # loop out with error
            __counter += 1
            if __counter >= __max_acc_count:
                raise RuntimeError(f'无法生成足够的唯一样本。已生成: {len(self.surrogate_buffer)}/{init_N}，尝试次数: {__counter}，成功生成成分次数: {unique_attempts}，见过的不同成分数: {len(seen_compositions)}')

        ''' train a GPR model with the latest experimented xs, ys '''
        self.update_surrogate()

    def update_surrogate(self):
        ''' update surrogate '''
        # 从buffer_list中提取成分用于GPR训练（GPR只使用成分）
        train_x = []
        for item in self.surrogate_buffer_list:
            if isinstance(item, dict):
                comp = item['comp']
            else:
                comp = item
            # 确保成分是19维的numpy数组
            comp = np.array(comp, dtype=np.float32).flatten()
            if len(comp) != ELEM_N:
                # 如果维度不对，尝试截取或填充
                if len(comp) > ELEM_N:
                    comp = comp[:ELEM_N]
                elif len(comp) < ELEM_N:
                    comp = np.pad(comp, (0, ELEM_N - len(comp)), mode='constant', constant_values=0.0)
            train_x.append(comp)
        
        # 计算reward
        train_y = [self.func(_x) for _x in train_x]
        train_x, train_y = np.array(train_x), np.array(train_y)

        # update self.best_score
        _best_idx = np.argmax(train_y.reshape(-1))
        self.best_score = train_y.reshape(-1)[_best_idx]
        if isinstance(self.surrogate_buffer_list[_best_idx], dict):
            self.best_x = self.surrogate_buffer_list[_best_idx]
            self.best_prop = self.raw_mo_func_with_proc(
                self.best_x['comp'], self.best_x['proc_bool'],
                self.best_x['proc_scalar'], self.best_x['phase']
            )
        else:
            self.best_x = {'comp': train_x[_best_idx], 
                          'proc_bool': np.zeros(N_PROC_BOOL, dtype=np.float32),
                          'proc_scalar': np.zeros(N_PROC_SCALAR, dtype=np.float32),
                          'phase': calculate_phase_scalar(train_x[_best_idx])}
            self.best_prop = self.raw_mo_func_with_proc(
                self.best_x['comp'], self.best_x['proc_bool'],
                self.best_x['proc_scalar'], self.best_x['phase']
            )

        ''' sklearn gpr '''
        self.surrogate = GaussianProcessRegressor(
            kernel = Matern(nu=2.5),
            alpha = 1e-6,
            normalize_y = True,
            n_restarts_optimizer = 5,
            random_state = self._random_state,
        )
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.surrogate.fit(train_x, train_y)

        ''' reset cached prediction '''
        self.cached_surrogate_pred_dict = dict()

    def surrogate_predict(self, x):
        _x_key = State.encode_key(x)
        if _x_key not in self.cached_surrogate_pred_dict:
            x = np.atleast_2d(x).reshape(1, -1)
            pred_val = self.surrogate.predict(x)[0]
            self.cached_surrogate_pred_dict[_x_key] = pred_val
            return pred_val
        else:
            return self.cached_surrogate_pred_dict[_x_key]

    def update_surrogate_buffer(self, sample_xs):
        ''' update experimented xs and ys '''
        sample_xs_arr = np.array(sample_xs)
        for _x in sample_xs_arr:
            _x_key = State.encode_key(_x)
            assert not self.check_collided(_x), \
                f'Logic leak, using duplicate xs that are already in the surrogate_buffer: {_x.tolist()}\n{sample_xs_arr}\n{self.surrogate_buffer_list}'
            self.surrogate_buffer.add(_x_key)
            self.surrogate_buffer_list.append(_x)

    def check_collided(self, sample_x):
        _x_key = State.encode_key(sample_x)
        return (_x_key in self.surrogate_buffer)

    def reset(self):
        ''' 
            For RL, using a fixed initial point to start exploration is reasonable. 
            Think of it as an empty chess board.
        '''
        return State(if_init = True)
    
    # def step(self, state: State, action_idx: int):
    #     '''
    #         Using internally maintained GPR model to calculate reward.

    #         NOTE: Current implementation uses immediately calculated im.R.
    #         TODO: Use lazily calculated im.R if time complexity scales up.
    #     '''
    #     next_state = State(previous_state = state, action = ALL_ACTIONS[action_idx])
    #     if next_state.done():
    #         next_score = self.func(next_state.get_composition())
    #         return next_state, next_score, True
    #     else:
    #         return next_state, 0., False

    # def step(self, state: State, action_idx: int):
    #     '''
    #         step (步进)
    #         Using internally maintained GPR model to calculate reward.

    #         NOTE: Current implementation uses immediately calculated im.R.
    #         TODO: Use lazily calculated im.R if time complexity scales up.
    #     '''
    #     next_state = State(previous_state = state, action = ALL_ACTIONS[action_idx])
    #     ''' direct reward '''
    #     curr_score, next_score = self.func(state.get_composition()), self.func(next_state.get_composition())  # NOTE direct reward
    #     self.update_interaction_stat(state, curr_score)
    #     self.update_interaction_stat(next_state,next_score)
    #     ''' surrogate reward '''
    #     # curr_score, next_score = self.surrogate_predict(state.get_composition()), self.surrogate_predict(next_state.get_composition())
    #     return next_state, (next_score - curr_score), next_state.done()
    
    def step(self, state: State, action_idx: int):
        '''
            step (步进)
            Using internally maintained GPR model to calculate reward.

            NOTE: Current implementation uses immediately calculated im.R.
            TODO: Use lazily calculated im.R if time complexity scales up. 
            TODO: 完善区分on-the-fly和surr reward的逻辑
        '''
        # 根据episode_count决定是成分阶段还是工艺阶段
        if state.get_episode_count() < COMP_EPISODE_LEN:
            # 成分阶段：使用原有的action（元素含量）
            action = ALL_ACTIONS[action_idx]
            next_state = State(previous_state=state, action=action)
            
            # 成分阶段结束时给中间奖励，帮助agent学习成分策略
            if next_state.get_episode_count() == COMP_EPISODE_LEN:
                # 成分阶段完成，使用默认工艺参数计算中间奖励
                comp = next_state.get_composition()
                phase = calculate_phase_scalar(comp)
                # 使用默认工艺参数（全0）计算成分性能
                intermediate_reward = self.func_with_proc(
                    comp,
                    np.zeros(N_PROC_BOOL, dtype=np.float32),
                    np.zeros(N_PROC_SCALAR, dtype=np.float32),
                    phase
                ) * 0.3  # 中间奖励权重0.3，避免过度影响最终奖励
                return next_state, intermediate_reward, False
        else:
            # 工艺阶段：使用action_idx直接更新工艺参数
            next_state = State(previous_state=state, action_idx=action_idx)
        
        ''' direct final reward '''
        if next_state.done():
            # Episode结束，计算reward
            next_state_comp = next_state.get_composition()
            next_state_proc_bool = next_state.get_proc_bool()
            next_state_proc_scalar = next_state.get_proc_scalar()
            next_state_phase = calculate_phase_scalar(next_state_comp)
            
            # 使用完整参数（成分+工艺）计算reward
            try:
                base_reward = self.func_with_proc(next_state_comp, next_state_proc_bool, 
                                            next_state_proc_scalar, next_state_phase)
            except Exception as e:
                # 如果计算失败，使用默认值
                print(f"Warning: func_with_proc failed: {e}, using default reward 0.0")
                base_reward = 0.0
            
            # 计算design_key用于检查是否为新设计
            design_key = State.encode_key(
                list(next_state_comp) + 
                list(next_state_proc_bool) + 
                list(next_state_proc_scalar) + 
                list(next_state_phase)
            )
            is_new_design = design_key not in self.surrogate_buffer
            
            # ============= 改进的奖励计算 =============
            
            # 1. 增加基础奖励的区分度
            # 使用非线性变换增加区分度：好的设计获得更高奖励
            reward_scale = 2.0  # 放大奖励差异
            scaled_base_reward = base_reward * reward_scale
            
            # 2. 工艺多样性奖励（显著增加，鼓励探索工艺参数）
            proc_diversity_bonus = 0.0
            is_wrought = next_state_proc_bool[1] > 0.5  # Is_Wrought
            has_deformation = next_state_proc_scalar[0] > 0 or next_state_proc_scalar[1] > 0
            has_ht1 = next_state_proc_scalar[2] > 0  # HT1_Temp > 0
            has_ht2 = next_state_proc_scalar[4] > 0  # HT2_Temp > 0
            
            # 锻造工艺奖励
            if is_wrought:
                proc_diversity_bonus += 0.05
                if has_deformation:
                    proc_diversity_bonus += 0.08  # 实际进行了变形
            
            # 热处理奖励（这是关键的工艺步骤）
            if has_ht1:
                proc_diversity_bonus += 0.12
                # HT1参数合理性奖励
                ht1_temp = next_state_proc_scalar[2]
                ht1_time = next_state_proc_scalar[3]
                if 600 <= ht1_temp <= 1200 and ht1_time > 0:
                    proc_diversity_bonus += 0.05  # 合理的热处理参数
            
            if has_ht2:
                proc_diversity_bonus += 0.15  # 进行了二次热处理，给予更高奖励
                ht2_temp = next_state_proc_scalar[4]
                ht2_time = next_state_proc_scalar[5]
                if 300 <= ht2_temp <= 900 and ht2_time > 0:
                    proc_diversity_bonus += 0.05
            
            # 3. 成分多样性奖励（鼓励多元素合金）
            comp_diversity_bonus = 0.0
            comp_array = np.array(next_state_comp)
            non_zero_elements = np.sum(comp_array > 0.01)  # 非零元素数量
            if non_zero_elements >= 3:
                comp_diversity_bonus = 0.03 * (non_zero_elements - 2)
                comp_diversity_bonus = min(comp_diversity_bonus, 0.15)
            
            # 4. 新设计发现奖励（大幅增加，强烈鼓励探索新设计）
            exploration_bonus = 0.0
            if is_new_design:
                exploration_bonus = 0.15  # 显著增加新设计奖励
                # 如果是非常不同的设计（成分变化大），额外奖励
                if hasattr(self, '_last_design_comp') and self._last_design_comp is not None:
                    comp_diff = np.sum(np.abs(comp_array - np.array(self._last_design_comp)))
                    if comp_diff > 0.2:  # 成分差异超过20%
                        exploration_bonus += 0.1
            self._last_design_comp = next_state_comp.copy()
            
            # 5. 计算总奖励
            reward = scaled_base_reward + proc_diversity_bonus + comp_diversity_bonus + exploration_bonus
            
            # 6. 相对改进奖励：如果超过当前最佳，给予额外奖励
            if scaled_base_reward > self.best_score:
                improvement_bonus = 0.2 * (scaled_base_reward - self.best_score)
                reward += improvement_bonus
            
            # ============= 调试信息 =============
            if not hasattr(self, '_reward_stats'):
                self._reward_stats = {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 
                                      'min': float('inf'), 'max': float('-inf'),
                                      'unique_designs': set()}
            
            self._reward_stats['count'] += 1
            self._reward_stats['sum'] += reward
            self._reward_stats['sum_sq'] += reward ** 2
            self._reward_stats['min'] = min(self._reward_stats['min'], reward)
            self._reward_stats['max'] = max(self._reward_stats['max'], reward)
            self._reward_stats['unique_designs'].add(design_key)
            
            # 每100个episode输出统计信息
            if self._reward_stats['count'] % 100 == 0:
                n = self._reward_stats['count']
                mean = self._reward_stats['sum'] / n
                var = self._reward_stats['sum_sq'] / n - mean ** 2
                std = np.sqrt(max(0, var))
                unique_count = len(self._reward_stats['unique_designs'])
                print(f"  [REWARD STATS] n={n}, mean={mean:.4f}, std={std:.4f}, "
                      f"range=[{self._reward_stats['min']:.4f}, {self._reward_stats['max']:.4f}], "
                      f"unique_designs={unique_count}")
            
            # 保存到buffer（使用完整设计）- design_key已在上面计算
            self.surrogate_buffer.add(design_key)
            self.surrogate_buffer_list.append({
                'comp': next_state_comp,
                'proc_bool': next_state_proc_bool,
                'proc_scalar': next_state_proc_scalar,
                'phase': next_state_phase
            })
            
            # 使用scaled_base_reward（基础性能分数）来更新best_score，而不是总reward
            # 这样best_score反映的是真实性能，而不是包含探索奖励的总分
            if scaled_base_reward > self.best_score:
                self.best_score = scaled_base_reward
                self.best_x = {
                    'comp': next_state_comp,
                    'proc_bool': next_state_proc_bool,
                    'proc_scalar': next_state_proc_scalar,
                    'phase': next_state_phase
                }
                self.best_prop = self.raw_mo_func_with_proc(
                    next_state_comp, next_state_proc_bool, 
                    next_state_proc_scalar, next_state_phase
                )
                # 发现新的最佳设计时输出
                print(f"  [NEW BEST] score={scaled_base_reward:.4f}, "
                      f"HT1={has_ht1}, HT2={has_ht2}, non_zero_elem={non_zero_elements}")
        else:
            reward = 0.
        return next_state, reward, next_state.done()
    
    def update_interaction_stat(self, state: State, state_score):
        state_comp = state.get_composition()
        self.surrogate_buffer.add(State.encode_key(state_comp))
        if sum([COMP_MIN_LIMITS[_i] <= state_comp[_i] <= COMP_MAX_LIMITS[_i] for _i in range(len(state_comp))]) == len(state_comp):
            self.best_score = max(self.best_score, state_score)
            self.best_x = state_comp
            self.best_prop = self.raw_mo_func(self.best_x)

    def sample_action(self, state: State):
        ai_low, ai_high = state.get_action_idx_limits()
        return np.random.randint(ai_low, ai_high + 1)
        #return ACTIONS_TO_INDEX_DICT[state.generate_random_action()]
    
    def get_best_score(self):
        return self.best_score
    
    def get_best_x(self):
        return self.best_x

    def get_best_prop(self):
        return self.best_prop
    
    def get_exp_number(self):
        assert len(self.surrogate_buffer) == len(self.surrogate_buffer_list), \
            'Something went wrong, len(surrogate_buffer) != len(surrogate_buffer_list)' + \
            f'{len(self.surrogate_buffer)}, {len(self.surrogate_buffer_list)}'
        return len(self.surrogate_buffer) - self.init_N
    
if __name__ == '__main__':
    env = Environment(
        init_N = 30,
        random_seed = 2,
    )
    print(env.get_best_score())
    print(env.get_best_x())
    print(env.get_best_prop())
    # print(env.get_exp_number())
    # print(env.surrogate_buffer)