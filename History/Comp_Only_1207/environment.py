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
from bayes_opt import UtilityFunction
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt.util import ensure_rng
import torch

from model_env_train import get_model, device, N_PROP

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
COMPOSITION_INTERVAL = 0.0001
COMPOSITION_ROUNDUP_DIGITS = 5
MANDATORY_MIN = 0.001

''' compositional bounds, NOTE USER PARAMETERS '''
C_MIN, C_MAX = 0., 0.0007
N_MIN, N_MAX = 0., 0.0020
O_MIN, O_MAX = 0., 0.0310
AL_MIN, AL_MAX = 0., 0.3000
SI_MIN, SI_MAX = 0., 0.0300
SC_MIN, SC_MAX = 0., 0.0020
TI_MIN, TI_MAX = 0.7, 1.0000
V_MIN, V_MAX = 0., 0.3000
CR_MIN, CR_MAX = 0., 0.2000
MN_MIN, MN_MAX = 0., 0.2000
FE_MIN, FE_MAX = 0., 0.2500
NI_MIN, NI_MAX = 0., 0.0040
CU_MIN, CU_MAX = 0., 0.1500
ZR_MIN, ZR_MAX = 0., 0.3500
NB_MIN, NB_MAX = 0., 0.3500
MO_MIN, MO_MAX = 0., 0.3000
SN_MIN, SN_MAX = 0., 0.1000
HF_MIN, HF_MAX = 0., 0.2700
TA_MIN, TA_MAX = 0., 0.4000
W_MIN, W_MAX = 0., 0.0020

COMP_LIMITS =(
    CompositionLimit(C_MIN, C_MAX),
    CompositionLimit(N_MIN, N_MAX),
    CompositionLimit(O_MIN, O_MAX),
    CompositionLimit(AL_MIN, AL_MAX),
    CompositionLimit(SI_MIN, SI_MAX),
    CompositionLimit(SC_MIN, SC_MAX),
    CompositionLimit(TI_MIN, TI_MAX),
    CompositionLimit(V_MIN, V_MAX),
    CompositionLimit(CR_MIN, CR_MAX),
    CompositionLimit(MN_MIN, MN_MAX),
    CompositionLimit(FE_MIN, FE_MAX),
    CompositionLimit(NI_MIN, NI_MAX),
    CompositionLimit(CU_MIN, CU_MAX),
    CompositionLimit(ZR_MIN, ZR_MAX),
    CompositionLimit(NB_MIN, NB_MAX),
    CompositionLimit(MO_MIN, MO_MAX),
    CompositionLimit(SN_MIN, SN_MAX),
    CompositionLimit(HF_MIN, HF_MAX),
    CompositionLimit(TA_MIN, TA_MAX),
    CompositionLimit(W_MIN, W_MAX),
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
EPISODE_LEN = ELEM_N-1

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
    '''
    model, d, scalers = get_model(model_path, data_path, resume = True)
    model.eval()    # NOTE important

    comp_scaler = scalers[0]
    proc_scaler = scalers[1]
    prop_scaler = scalers[2]

    # TODO modification needed, NOTE keep it fixed
    base_proc = np.array([0,], dtype=np.float32).reshape(1, -1)
    elem_ft = d[-1]

    # pre-calculated processing condition data
    _proc = proc_scaler.transform(base_proc)
    _proc = torch.tensor(_proc, dtype=torch.float32).reshape(1, 1, d[1].shape[-1], 1).to(device)

    def _func(x):
        ''' maps a composition (a list with the sumation of 1.) to its mechanical property prediction '''
        x = (np.array(x) * COMP_MULTIPLIER).round(ROUND_DIGIT)
        assert len(x) == ELEM_N, f'len(x) != ELEM_N: {len(x)}'
        _comp = x.reshape(1, -1)

        _comp = comp_scaler.transform(_comp)

        _comp = torch.tensor(_comp, dtype=torch.float32).reshape(1, 1, d[0].shape[-1], 1).to(device)
        elem_t = torch.tensor(elem_ft, dtype=torch.float32).reshape(1, 1, *(elem_ft.shape)).to(device)

        _prop = model(_comp, elem_t, _proc).detach().cpu().numpy()
        _prop = prop_scaler.inverse_transform(_prop)

        return _prop[0]  # 确保返回单个标量值而不是数组
    
    return _func

def get_mo_ground_truth_func():
    '''
        Build a linearly compounded multi-objective (MO) 'property' predicting function.
    '''
    _func = get_ground_truth_func('model_multi.pth', 'data_multi.pth')

    ''' local optimal maximums for YS, UTS, and ELongation '''
    _mo_scale = np.array([
        73,
        695,
        780,
        13.3,
        297,
    ])

    def _mo_raw_func(x):
        """
            返回原始多目标向量（未加权）的预测值，形如 (n_props,)
        """
        return np.array(_func(x))

    # 返回原始向量函数以及用于归一化的尺度向量
    return _mo_raw_func, _mo_scale

class State:
    def __init__(self, if_init: bool = False, 
                 previous_state: State = None, 
                 action: ActionType = None, 
                 episode_len = EPISODE_LEN):
        if if_init:
            # atomic fraction (%) of [C, Al, V, Cr, Mn, Fe, Co, Ni, Cu, Mo]
            # self.__composition = [0., 0., 0.1, 0.15, 0., 0.4, 0.1, 0.25, 0., 0.,]   # TODO further tests needed
            self.__composition = [0.] * ELEM_N
            self.__episode_len = episode_len
            self.__episode_count = -1
        else:
            self.__composition = deepcopy(previous_state.get_composition())
            previous_episode_no = previous_state.get_episode_count()
            self.__episode_len = episode_len
            self.__episode_count = previous_episode_no + 1

            ''' 
                substitution rule:
                    2024.07.08 - Only reach a rational composition when the episode ends.
            '''
            sub_idx = self.__episode_count
            _min = max(COMP_MIN_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]) - sum(COMP_MAX_LIMITS[sub_idx + 1:]))
            _max = min(COMP_MAX_LIMITS[sub_idx], 1 - sum(self.__composition[:sub_idx]))
            _min, _max = round(_min, COMPOSITION_ROUNDUP_DIGITS), round(_max, COMPOSITION_ROUNDUP_DIGITS)
            action = min(max(action, _min), _max)
            self.__composition[sub_idx] = action

            if sub_idx == self.__episode_len - 1:
                last = 1. - sum(self.__composition[:-1])
                if 0 < last < MANDATORY_MIN:
                    last = 0.0
                self.__composition[-1] = round(last, COMPOSITION_ROUNDUP_DIGITS)

            # round up compositions
            for idx in range(len(self.__composition)):
                self.__composition[idx] = round(self.__composition[idx], \
                    COMPOSITION_ROUNDUP_DIGITS)

    def get_episode_len(self) -> int:
        return self.__episode_len

    def get_episode_count(self) -> int:
        return self.__episode_count
    
    def get_composition(self):
        return self.__composition

    def repr(self):
        # len(feature) corresponds to flattened dimensions in DqlModel.
        feature = deepcopy(self.__composition)
        feature.append(self.__episode_count)
        return feature

    def done(self):
        return self.__episode_count == self.__episode_len - 1

    def get_action_idx_limits(self):
        '''
            Get action limits according to current state.

            @output:    (composition_min_idx, composition_max_idx)
                            composition_min_idx * COMPOSITION_INTERVAL == composition_lower_limit_in_float
                            composition_max_idx * COMPOSITION_INTERVAL == composition_upper_limit_in_float
        '''
        elem_index = self.__episode_count + 1
        assert elem_index < self.__episode_len, f'elem_index: {elem_index}, ELEM_N: {ELEM_N}'
        _min = max(COMP_MIN_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]) - sum(COMP_MAX_LIMITS[elem_index + 1:]))
        _max = min(COMP_MAX_LIMITS[elem_index], 1 - sum(self.__composition[:elem_index]) - sum(COMP_MIN_LIMITS[elem_index + 1:]))

        _min, _max = round(_min, COMPOSITION_ROUNDUP_DIGITS), round(_max, COMPOSITION_ROUNDUP_DIGITS)
        if _max < MANDATORY_MIN:
            _max = 0.0
        if 0 < _min < MANDATORY_MIN:
            _min = MANDATORY_MIN

        comp_min_idx = ACTIONS_TO_INDEX_DICT[round(_min, COMPOSITION_ROUNDUP_DIGITS)]
        comp_max_idx = ACTIONS_TO_INDEX_DICT[round(_max, COMPOSITION_ROUNDUP_DIGITS)]
        return comp_min_idx, comp_max_idx

    def generate_random_action(self, random_seed = None) -> ActionType:
        '''
            Generate one random action that can be applied to this state

            @output:    a random action in float
        '''
        random_state = ensure_rng(random_seed)
        comp_min_idx, comp_max_idx = self.get_action_idx_limits()
        assert comp_max_idx>=comp_min_idx,  f"{comp_min_idx},{comp_max_idx}\n{self.__composition}"
        rand_comp_idx = random_state.randint(comp_min_idx, comp_max_idx + 1)
        return ALL_ACTIONS[rand_comp_idx]
        return rand_comp_idx
    
    @staticmethod
    def encode_key(x):
        return STATE_DELIMETER_CHAR.join(map(str, x))
    
    @staticmethod
    def decode_key(key: str):
        return np.array(list(map(float, key.split(STATE_DELIMETER_CHAR)))).round(ROUND_DIGIT)   # check return data type

class Environment:
    def __init__(self, 
                 init_N = 50, 
                 enable_ei: bool = False,
                 random_seed = None):
        self.init_world_model()

        # self.all_actions = np.linspace(self.x_min, self.x_max, self.act_dim).round(ROUND_DIGIT).tolist()
        self.state_dim = len(self.reset().repr())
        self.act_dim = ALL_ACTIONS_COUNT

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
        self.enable_ei = enable_ei
        self.init_surrogate(self.init_N, random_seed)
    
    def init_world_model(self,):
        # 获取返回的原始多目标函数及其尺度
        raw_func, mo_scale = get_mo_ground_truth_func()
        self.raw_mo_func = raw_func
        self.mo_scale = np.array(mo_scale)
        # 初始权重：相等权重
        n_props = len(self.mo_scale)
        self.mo_weights = np.ones(n_props, dtype=float) / float(n_props)
        # self.func 返回标量（按当前权重将向量合并）
        self.func = lambda x: float(np.dot(self.raw_mo_func(x) / self.mo_scale, self.mo_weights))

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
        self.func = lambda x: float(np.dot(self.raw_mo_func(x) / self.mo_scale, self.mo_weights))

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
            vals = np.array([self.raw_mo_func(x) for x in self.surrogate_buffer_list])  # shape (n_samples, n_props)
            # divide into two windows
            k = max(1, min(window, n // 2))
            recent = vals[-k:]
            prev = vals[-2*k:-k] if n >= 2*k else vals[:k]
            recent_mean = np.mean(recent / self.mo_scale, axis=0)
            prev_mean = np.mean(prev / self.mo_scale, axis=0)
            imp = recent_mean - prev_mean
            rank = np.argsort(np.argsort(imp))
            Weight_List=[6, 5, 4, 3, 2]
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

    def init_surrogate(self, init_N, seed):
        ''' initialize surrogate with init_N randomly generated samples '''
        __counter, __max_acc_count = 0, int(1e4)
        random_state = ensure_rng(seed)
        while len(self.surrogate_buffer) < init_N:
            _state = State(if_init = True)
            for _ in range(EPISODE_LEN):
                _state = State(previous_state = _state, action = _state.generate_random_action(random_state))
            _x = _state.get_composition()
            _x_key = State.encode_key(_x)
            if _x_key not in self.surrogate_buffer:
                self.surrogate_buffer.add(_x_key)
                self.surrogate_buffer_list.append(_x)

            # loop out with error
            __counter += 1
            if __counter >= __max_acc_count:
                assert False, 'Potential permanent forloop!'

        ''' train a GPR model with the latest experimented xs, ys '''
        self.update_surrogate()

    def update_surrogate(self):
        ''' update surrogate '''
        train_x = [State.decode_key(_x_key) for _x_key in self.surrogate_buffer]
        train_y = [self.func(_x) for _x in train_x]
        train_x, train_y = np.array(train_x), np.array(train_y)

        # update self.best_score
        _best_idx = np.argmax(train_y.reshape(-1))
        self.best_score = train_y.reshape(-1)[_best_idx]
        self.best_x = train_x[_best_idx]
        self.best_prop = self.raw_mo_func(self.best_x)

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

        if self.enable_ei:
            self.ei_acqf = UtilityFunction(kind = "ei", xi = 0.0)

        ''' reset cached prediction '''
        self.cached_surrogate_pred_dict = dict()

    def surrogate_predict(self, x):
        _x_key = State.encode_key(x)
        if _x_key not in self.cached_surrogate_pred_dict:
            x = np.atleast_2d(x).reshape(1, -1)
            if not self.enable_ei:
                pred_val = self.surrogate.predict(x)[0]
            else:
                pred_val = _ei = self.ei_acqf.utility(x, gp = self.surrogate, y_max = self.best_score)
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
        next_state = State(previous_state = state, action = ALL_ACTIONS[action_idx])
        ''' direct final reward '''
        if next_state.done():
            next_state_comp = next_state.get_composition()
            reward = self.func(next_state_comp)      # NOTE direct final reward
            self.surrogate_buffer.add(State.encode_key(next_state_comp))
            self.surrogate_buffer_list.append(next_state_comp)
            if self.best_score < reward:
                self.best_score = max(self.best_score, reward)
                self.best_x = next_state_comp
                self.best_prop = self.raw_mo_func(self.best_x)
            # print(next_state_comp,'\n', self.best_x,'\n',round(reward, 4), round(self.best_score, 4))   # for debug
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