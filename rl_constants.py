import warnings
from collections import namedtuple
from dataclasses import dataclass
from math import inf

import numpy as np
import torch

from surrogate_train import get_model, device
from surrogate_model import Final, N_PROC_BOOL, N_PROC_SCALAR, N_PHASE_SCALAR, PROP

COMPOSITION_INTERVAL = 0.001
COMPOSITION_ROUNDUP_DIGITS = 3
MANDATORY_MIN = 0.001
MEMORY_CAPACITY = 3000

Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))
TrainingIndicator = namedtuple('TrainingIndicator', ('epoch', 'loss', 'total_q'))
CompositionLimit = namedtuple('CompositionLimit', ('min_bound', 'max_bound'))


@dataclass(frozen=True)
class ObjectiveSpec:
    trend: str
    lower: float = 0.0
    upper: float = inf
    scale: float = 1.0

TI_MIN, TI_MAX = 0.6, 1.0
AL_MIN, AL_MAX = 0.0, 0.1
V_MIN, V_MAX = 0.0, 0.1
CR_MIN, CR_MAX = 0.0, 0.1
FE_MIN, FE_MAX = 0.0, 0.08
ZR_MIN, ZR_MAX = 0.0, 0.3
NB_MIN, NB_MAX = 0.0, 0.4
MO_MIN, MO_MAX = 0.0, 0.15
SN_MIN, SN_MAX = 0.0, 0.1
TA_MIN, TA_MAX = 0.0, 0.3

COMP_LIMITS = (
    CompositionLimit(TI_MIN, TI_MAX),
    CompositionLimit(AL_MIN, AL_MAX),
    CompositionLimit(V_MIN, V_MAX),
    CompositionLimit(CR_MIN, CR_MAX),
    CompositionLimit(FE_MIN, FE_MAX),
    CompositionLimit(ZR_MIN, ZR_MAX),
    CompositionLimit(NB_MIN, NB_MAX),
    CompositionLimit(MO_MIN, MO_MAX),
    CompositionLimit(SN_MIN, SN_MAX),
    CompositionLimit(TA_MIN, TA_MAX),
)

COMP_MIN_LIMITS = list(CompositionLimit(*zip(*COMP_LIMITS)).min_bound)
COMP_MAX_LIMITS = list(CompositionLimit(*zip(*COMP_LIMITS)).max_bound)

NEW_COMP_MIN_LIMITS = []
NEW_COMP_MAX_LIMITS = []
for lo, hi in zip(COMP_MIN_LIMITS, COMP_MAX_LIMITS):
    if hi < MANDATORY_MIN:
        NEW_COMP_MIN_LIMITS.append(0.0)
        NEW_COMP_MAX_LIMITS.append(0.0)
    elif lo > MANDATORY_MIN:
        NEW_COMP_MIN_LIMITS.append(lo)
        NEW_COMP_MAX_LIMITS.append(hi)
    elif lo > 0.0:
        NEW_COMP_MIN_LIMITS.append(MANDATORY_MIN)
        NEW_COMP_MAX_LIMITS.append(hi)
    else:
        NEW_COMP_MIN_LIMITS.append(0.0)
        NEW_COMP_MAX_LIMITS.append(hi)

COMP_MIN_LIMITS = NEW_COMP_MIN_LIMITS
COMP_MAX_LIMITS = NEW_COMP_MAX_LIMITS

COMP_MULTIPLIER = 100.0
ELEM_N = len(COMP_LIMITS)
COMP_EPISODE_LEN = ELEM_N - 1
PROC_EP_WROUGHT = COMP_EPISODE_LEN
PROC_EP_DEF_TEMP = PROC_EP_WROUGHT + 1
PROC_EP_DEF_STRAIN = PROC_EP_WROUGHT + 2
PROC_EP_HT1 = PROC_EP_WROUGHT + 3
PROC_EP_HT1_TEMP = PROC_EP_WROUGHT + 4
PROC_EP_HT1_TIME = PROC_EP_WROUGHT + 5
PROC_EP_HT1_COOL = PROC_EP_WROUGHT + 6
PROC_EP_HT2 = PROC_EP_WROUGHT + 7
PROC_EP_HT2_TEMP = PROC_EP_WROUGHT + 8
PROC_EP_HT2_TIME = PROC_EP_WROUGHT + 9
PROC_EP_HT2_COOL = PROC_EP_WROUGHT + 10
MAX_EPISODE_LEN = PROC_EP_HT2_COOL + 1

EPSILON_START = 0.9
EPSILON_DECAY_COEF = 10000
EPSILON_END = 0.1
LEARNING_RATE = 1e-3
RL_TRAINING_EPOCHS = 1000
DEFAULT_LOG_INTERVAL = 1000
RL_SAMPLE_BATCH_SIZE = 128
GAMMA = 0.80
TARGET_UPDATE_PERIOD = 10
DQL_AGENT_PATH = 'dql_agent.pt'
DQL_TRAINING_INDICATOR_PATH = 'rl_agent_training_indicators.pk'

COMP_LOW_BOUND_INT = round(min(COMP_MIN_LIMITS) / COMPOSITION_INTERVAL)
COMP_HIGH_BOUND_INT = round(max(COMP_MAX_LIMITS) / COMPOSITION_INTERVAL)
COMP_LOW_BOUND = COMP_LOW_BOUND_INT * COMPOSITION_INTERVAL
COMP_HIGH_BOUND = COMP_HIGH_BOUND_INT * COMPOSITION_INTERVAL

COMP_ACTIONS = [0.0]
mandatory_idx = int(MANDATORY_MIN / COMPOSITION_INTERVAL)
for x in range(mandatory_idx, COMP_HIGH_BOUND_INT + 1):
    COMP_ACTIONS.append(round(x * COMPOSITION_INTERVAL, COMPOSITION_ROUNDUP_DIGITS))
COMP_ACTIONS = sorted(set(COMP_ACTIONS))
COMP_ACTIONS_COUNT = len(COMP_ACTIONS)
ACTIONS_TO_INDEX_DICT = dict(zip(COMP_ACTIONS, range(COMP_ACTIONS_COUNT)))

ROUND_DIGIT = 5
STATE_DELIMETER_CHAR = '*'


def get_ground_truth_func_with_proc(model_path='models/surrogate/model_Final.pth', data_path='models/surrogate/data.pth'):
    """Return predictor f(comp, proc_bool, proc_scalar, phase_scalar) -> properties."""
    with warnings.catch_warnings():
        from sklearn.base import InconsistentVersionWarning
        warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        model, train_d, _, scalers = get_model(model=Final(), model_path=model_path, data_path=data_path, resume=True, train=False)

    model.to(device)
    model.eval()

    comp_scaler = scalers[0]
    proc_bool_scaler = scalers[1]
    proc_scalar_scaler = scalers[2]
    phase_scalar_scaler = scalers[3]
    prop_scaler = scalers[4]
    elem_ft = train_d[6]

    def _func(comp, proc_bool, proc_scalar, phase_scalar):
        comp = np.array(comp, dtype=np.float32)
        if len(comp) != ELEM_N:
            raise ValueError(f'composition dim mismatch: {len(comp)} != {ELEM_N}')

        comp = (comp * COMP_MULTIPLIER).round(ROUND_DIGIT)
        _comp = comp.reshape(1, -1)
        _comp = comp_scaler.transform(_comp)

        _proc_bool = proc_bool_scaler.transform(proc_bool.reshape(1, -1))
        _proc_scalar = proc_scalar_scaler.transform(proc_scalar.reshape(1, -1))
        _phase_scalar = phase_scalar_scaler.transform(phase_scalar.reshape(1, -1))

        _proc_bool_mask = np.ones_like(_proc_bool, dtype=np.float32)
        _proc_scalar_mask = np.ones_like(_proc_scalar, dtype=np.float32)

        _comp = torch.tensor(_comp, dtype=torch.float32).reshape(1, 1, _comp.shape[1], 1).to(device)
        elem_t = torch.tensor(elem_ft, dtype=torch.float32).reshape(1, 1, *(elem_ft.shape)).to(device)
        _proc_bool = torch.tensor(_proc_bool, dtype=torch.float32).reshape(1, 1, N_PROC_BOOL, 1).to(device)
        _proc_scalar = torch.tensor(_proc_scalar, dtype=torch.float32).reshape(1, 1, N_PROC_SCALAR, 1).to(device)
        _phase_scalar = torch.tensor(_phase_scalar, dtype=torch.float32).reshape(1, 1, N_PHASE_SCALAR, 1).to(device)
        _proc_bool_mask = torch.tensor(_proc_bool_mask, dtype=torch.float32).reshape(1, 1, N_PROC_BOOL, 1).to(device)
        _proc_scalar_mask = torch.tensor(_proc_scalar_mask, dtype=torch.float32).reshape(1, 1, N_PROC_SCALAR, 1).to(device)

        _prop = model(_comp, elem_t, _proc_bool, _proc_scalar, _phase_scalar, _proc_bool_mask, _proc_scalar_mask).detach().cpu().numpy()
        _prop = prop_scaler.inverse_transform(_prop)
        return _prop[0]

    return _func


def calculate_phase_scalar(composition):
    """Calculate phase descriptors from composition."""
    comp = np.array(composition) * 100.0

    mo_eq = (
        1.0 * comp[7]
        + 0.67 * comp[2]
        + 0.28 * comp[6]
        + 0.22 * comp[9]
        + 1.6 * comp[3]
        - 1.0 * comp[1]
    ) 

    al_eq = (
        1.0 * comp[1]
        + comp[5] / 6.0
        + comp[8] / 3.0
    )

    beta_transform_T = (
        882.0
        + 2.1 * comp[1]
        - 9.5 * comp[7]
        + 4.2 * comp[8]
        - 6.9 * comp[5]
        - 11.8 * comp[2]
        - 12.1 * comp[3]
    )

    return np.array([mo_eq, al_eq, beta_transform_T], dtype=np.float32)


def get_mo_ground_truth_func():
    """Return raw multi-objective predictor and scaling vector."""
    _func = get_ground_truth_func_with_proc(
        model_path='models/surrogate/model_Final.pth',
        data_path='models/surrogate/data.pth',
    )

    _mo_scale = np.array([70, 813, 932, 13.5, 273])

    def _mo_raw_func(comp, proc_bool, proc_scalar, phase_scalar):
        return np.array(_func(comp, proc_bool, proc_scalar, phase_scalar))

    return _mo_raw_func, _mo_scale


OBJECTIVE_SPECS = {
    'YM': ObjectiveSpec(trend='maximize', lower=120.0, upper=inf, scale=70.0),
    'YS': ObjectiveSpec(trend='neutral', lower=1400.0, upper=inf, scale=813.0),
    'UTS': ObjectiveSpec(trend='neutral', lower=1600.0, upper=inf, scale=932.0),
    'El': ObjectiveSpec(trend='neutral', lower=0.0, upper=inf, scale=13.5),
    'HV': ObjectiveSpec(trend='neutral', lower=0.0, upper=inf, scale=273.0),
}


DEF_TEMP_CANDIDATES = np.array([
    0,
    600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875,
    900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150,
], dtype=np.float32)
DEF_TEMP_MAX = 1150.0
DEF_TEMP_COUNT = len(DEF_TEMP_CANDIDATES)

DEF_STRAIN_CANDIDATES = np.array([
    0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0
], dtype=np.float32)
DEF_STRAIN_MAX = 99.0
DEF_STRAIN_COUNT = len(DEF_STRAIN_CANDIDATES)

HT1_TEMP_CANDIDATES = np.array([
    0,
    400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700,
    725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
    1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275,
    1300, 1325, 1350, 1375, 1400,
], dtype=np.float32)
HT1_TEMP_MAX = 1400.0
HT1_TEMP_COUNT = len(HT1_TEMP_CANDIDATES)

HT1_TIME_CANDIDATES = np.array([
    0.0,
    0.017, 0.05, 0.083, 0.167, 0.25, 0.33, 0.5, 0.58,
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0,
    20.0, 24.0, 30.0,
], dtype=np.float32)
HT1_TIME_MAX = 30.0
HT1_TIME_COUNT = len(HT1_TIME_CANDIDATES)

HT2_TEMP_CANDIDATES = np.array([
    0,
    200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
    525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
], dtype=np.float32)
HT2_TEMP_MAX = 1000.0
HT2_TEMP_COUNT = len(HT2_TEMP_CANDIDATES)

HT2_TIME_CANDIDATES = np.array([
    0.0,
    0.167, 0.33, 0.5, 1.0, 1.5, 1.67, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 8.5, 9.0, 10.0, 12.0, 15.0, 18.0, 24.0, 30.0, 36.0,
    48.0, 72.0,
], dtype=np.float32)
HT2_TIME_MAX = 72.0
HT2_TIME_COUNT = len(HT2_TIME_CANDIDATES)

COOLING_METHODS_1 = ['Quench', 'Air', 'Furnace']
COOLING_METHOD_COUNT_1 = len(COOLING_METHODS_1)
COOLING_METHODS_2 = ['Quench', 'Air']
COOLING_METHOD_COUNT_2 = len(COOLING_METHODS_2)

PROC_ACTIONS = {
    'init_state': 2,
    'deform_decision': 2,
    'def_temp': DEF_TEMP_COUNT,
    'def_strain': DEF_STRAIN_COUNT,
    'ht1_decision': 2,
    'ht1_temp': HT1_TEMP_COUNT,
    'ht1_time': HT1_TIME_COUNT,
    'ht1_cooling': COOLING_METHOD_COUNT_1,
    'ht2_decision': 2,
    'ht2_temp': HT2_TEMP_COUNT,
    'ht2_time': HT2_TIME_COUNT,
    'ht2_cooling': COOLING_METHOD_COUNT_2,
}

MAX_PROC_ACTION_SPACE = max(PROC_ACTIONS.values())
EPISODE_COUNT_MAX = 30.0

PHASE_NORMALIZATION = {
    'mo_eq_min': -100.0,
    'mo_eq_max': 100.0,
    'al_eq_min': 0.0,
    'al_eq_max': 50.0,
    'beta_transform_T_min': 700.0,
    'beta_transform_T_max': 1200.0,
}


def get_proc_action_size(episode_count):
    """Return valid process action-space size for current stage."""
    if episode_count < COMP_EPISODE_LEN - 1:
        return COMP_ACTIONS_COUNT
    base = COMP_EPISODE_LEN - 1
    if episode_count == base:
        return PROC_ACTIONS['init_state']
    if episode_count == base + 1:
        return PROC_ACTIONS['def_temp']
    if episode_count == base + 2:
        return PROC_ACTIONS['def_strain']
    if episode_count == base + 3:
        return PROC_ACTIONS['ht1_decision']
    if episode_count == base + 4:
        return PROC_ACTIONS['ht1_temp']
    if episode_count == base + 5:
        return PROC_ACTIONS['ht1_time']
    if episode_count == base + 6:
        return PROC_ACTIONS['ht1_cooling']
    if episode_count == base + 7:
        return PROC_ACTIONS['ht2_decision']
    if episode_count == base + 8:
        return PROC_ACTIONS['ht2_temp']
    if episode_count == base + 9:
        return PROC_ACTIONS['ht2_time']
    if episode_count == base + 10:
        return PROC_ACTIONS['ht2_cooling']
    return 0
