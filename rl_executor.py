import random
import uuid
import warnings

import joblib
import numpy as np

from buffer import ReplayBuffer
from environment import COMPOSITION_ROUNDUP_DIGITS, Environment, ALL_ACTIONS_COUNT
from rl_dqn_agents import (
    DQNAgent, 
    collect_random, propose_candidates_to_exp, train_one_ep, retry_on_error,
    DEFAULT_INIT_EPSILON,
    device,
)
import torch
N_JOBS=1
torch.cuda.set_per_process_memory_fraction(0.95/N_JOBS)

def rl_dqn_serial(init_N = 20,
                  seed = 0,
                  train_ep_n = 150, 
                  ):
    '''
        Train one DQN agent using on-the-fly rewards.
    '''
    id_str = str(uuid.uuid4())[:8]

    env = Environment(init_N = init_N, enable_ei = False, random_seed = seed)

    ''' 
        Clear internal buffers.
        The internal buffers are intially built for GPR rewards.    TODO: 完善区分on-the-fly和surr reward的逻辑
    '''
    env.surrogate_buffer.clear()
    env.surrogate_buffer_list.clear()

    agent = DQNAgent(env)

    TRAIN_START_MIN_MEMORY = 3000
    
    ''' 
        Although TRAIN_START_MIN_MEMORY random samples are collected, the
        immediate rewards are lazily evaluated. This is reasonable as we can
        start training, using random samples online.
    '''
    agent.memory = ReplayBuffer(3000, 256, device, env)
    agent.epsilon = DEFAULT_INIT_EPSILON
    collect_random(env, agent.memory, TRAIN_START_MIN_MEMORY)

    traj = []

    ''' train ONE agent using on-the-fly rewards '''
    for ep in range(train_ep_n + 1):
        train_one_ep(agent, env, ep)
        if ep % 10 == 0 or ep<=10:
            bsf = round(env.best_score, COMPOSITION_ROUNDUP_DIGITS)
            _tmp_res = [ep, bsf]
            traj.append(_tmp_res)
            # also print current multi-objective weights for debugging
            try:
                weights = env.mo_weights if hasattr(env, 'mo_weights') else None
                w_str = np.round(weights, 4).tolist() if weights is not None else None
            except Exception:
                w_str = None
            print(*_tmp_res, 'mo_weights=', w_str)
            # print(*env.get_best_x())

    bsf_list = [env.func(_comp) for _comp in env.surrogate_buffer_list]

    bsf_list = np.maximum.accumulate(bsf_list).tolist()

    joblib.dump(bsf_list, f'rl_single_agent_direct_R-{id_str}.pkl')

    print(*env.get_best_x())

if __name__ == '__main__':

    joblib.Parallel(n_jobs = N_JOBS)(
        joblib.delayed(rl_dqn_serial)(
            train_ep_n = 2000, seed = sd
        ) for sd in [random.randint(0, 999) for _ in range(N_JOBS)])