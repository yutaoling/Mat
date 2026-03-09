import os
import random
import argparse
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env import Environment, COMP_MULTIPLIER
from surrogate_model import ID, COMP, PROC_BOOL, PROC_SCALAR, PHASE_SCALAR, PROP

N_JOBS = 1
torch.cuda.set_per_process_memory_fraction(0.95/N_JOBS)
PPO_N_STEPS = 128
PPO_BATCH_SIZE = 128
warnings.filterwarnings(
    "ignore",
    message=r".*bgemm_internal_cublaslt error: CUBLAS_STATUS_EXECUTION_FAILED.*Will attempt to recover by calling cublas instead.*",
    category=UserWarning,
)


class AlloyGymEnv(gym.Env):
    """Gymnasium wrapper for titanium alloy RL environment."""

    metadata = {"render_modes": []}

    def __init__(self, init_N=20, seed=0):
        super().__init__()
        self.base_env = Environment(init_N=init_N, random_seed=seed)
        self.base_env.reset_archive(init_N=0)
        self._state = None
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_env.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.base_env.act_dim)

    def action_masks(self):
        return self._state.get_action_mask().astype(bool)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self._state = self.base_env.reset()
        obs = self._state.repr().astype(np.float32)
        return obs, {}

    def step(self, action):
        next_state, reward, done = self.base_env.step(self._state, int(action))
        self._state = next_state
        obs = self._state.repr().astype(np.float32)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, {}


class RewardLogCallback(BaseCallback):
    """Log episodic reward only."""

    def __init__(self, log_every=10, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        for r, d in zip(rewards, dones):
            print(f"{self.num_timesteps}, {float(r):.6f}")
            self.episode_reward += float(r)
            if d:
                self.episode_count += 1
                self.episode_reward = 0.0
        return True


def save_results_to_csv(env: Environment, seed: int):
    """Save top-10 designs and predicted properties to CSV."""
    top_results = env.get_top_results(k=10)
    if not top_results:
        return

    rows = []
    for rank, item in enumerate(top_results, start=1):
        best_x = item["x"]
        best_prop = item["prop"]
        comp = np.array(best_x["comp"])
        proc_bool = np.array(best_x["proc_bool"])
        proc_scalar = np.array(best_x["proc_scalar"])
        phase = np.array(best_x["phase"])
        comp_percent = comp * COMP_MULTIPLIER

        data_dict = {
            "rank": rank,
            "reward": round(float(item["score"]), 6),
            "id": int(10000 + seed + rank - 1),
            "Activated": 1,
        }

        for i, elem_name in enumerate(COMP):
            data_dict[elem_name] = round(comp_percent[i], 3)

        for i, proc_name in enumerate(PROC_BOOL):
            data_dict[proc_name] = int(proc_bool[i]) if i < len(proc_bool) else 0

        for i, proc_name in enumerate(PROC_SCALAR):
            value = proc_scalar[i] if i < len(proc_scalar) else 0.0
            if proc_name in ["Def_Temp", "HT1_Temp", "HT2_Temp"]:
                data_dict[proc_name] = round(value, 1)
            elif proc_name in ["HT1_Time", "HT2_Time"]:
                data_dict[proc_name] = round(value, 3)
            else:
                data_dict[proc_name] = round(value, 1)

        for i, phase_name in enumerate(PHASE_SCALAR):
            data_dict[phase_name] = round(phase[i], 2) if i < len(phase) else 0.0

        prop_array = np.array(best_prop) if isinstance(best_prop, (list, np.ndarray)) else best_prop
        for i, prop_name in enumerate(PROP):
            data_dict[prop_name] = round(float(prop_array[i]), 1) if i < len(prop_array) else None

        rows.append(data_dict)

    df = pd.DataFrame(rows)
    column_order = ["rank", "reward"] + ID + COMP + PROC_BOOL + PROC_SCALAR + PHASE_SCALAR + PROP
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/rl_top10_results_seed{seed}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")


def rl_masked_ppo(init_N=20, seed=0, total_timesteps=10_000, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE):
    """Train MaskablePPO for titanium alloy optimization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            nj = max(1, int(os.environ.get("RL_N_JOBS", "1")))
            torch.cuda.set_per_process_memory_fraction(0.95 / nj)
        except Exception:
            pass
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def make_env():
        return AlloyGymEnv(init_N=init_N, seed=seed)

    vec_env = DummyVecEnv([make_env])

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=int(n_steps),
        batch_size=int(min(batch_size, n_steps)),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    reward_callback = RewardLogCallback(log_every=10, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=reward_callback)

    env = vec_env.envs[0].base_env
    print(f"final_reward={env.get_best_score():.6f}")
    save_results_to_csv(env, seed)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--init-n", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    args = parser.parse_args()
    os.environ["RL_N_JOBS"] = str(int(max(1, args.n_jobs)))

    base_seed = random.randint(0, 10000)
    joblib.Parallel(n_jobs=int(max(1, args.n_jobs)))(
        joblib.delayed(rl_masked_ppo)(
            init_N=args.init_n,
            seed=base_seed + i * 1000,
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        ) for i in range(int(max(1, args.n_jobs)))
    )
