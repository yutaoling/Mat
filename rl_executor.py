import os
import random
import warnings
from datetime import datetime
from pathlib import Path
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
from surrogate_model import COMP, PROC_BOOL, PROC_SCALAR, PHASE_SCALAR, PROP

N_JOBS = 1
PPO_N_STEPS = 128
PPO_BATCH_SIZE = 128
RL_CHECKPOINT_FREQ = 1000
INIT_N = 20
TOTAL_TIMESTEPS = 10_000
BASE_SEED = 42
RESUME_TRAINING = True
RL_MODEL_PATH = f"models/rl/maskableppo_seed{BASE_SEED}.zip"
RL_ENV_STATE_PATH = f"models/rl/maskableppo_seed{BASE_SEED}_env_state.pkl"
BEST_REWARD_LOG_PATH = f"logs/rl/best_reward_seed{BASE_SEED}.txt"
warnings.filterwarnings(
    "ignore",
    message=r".*bgemm_internal_cublaslt error: CUBLAS_STATUS_EXECUTION_FAILED.*Will attempt to recover by calling cublas instead.*",
    category=UserWarning,
)

torch.cuda.set_per_process_memory_fraction(0.95 / N_JOBS)

RESULT_CSV_COLUMNS = [
    "rank", "reward", "train_step",
    "Ti", "Al", "V", "Cr", "Mn", "Fe", "Cu", "Zr", "Nb", "Mo", "Sn", "Hf", "Ta", "W", "Si", "C", "N", "O", "Sc",
    "Is_Not_Wrought", "Is_Wrought",
    "Def_Temp", "Def_Strain",
    "HT1", "HT1_Temp", "HT1_Time", "HT1_Quench", "HT1_Air", "HT1_Furnace",
    "HT2", "HT2_Temp", "HT2_Time", "HT2_Quench", "HT2_Air",
    "Mo_eq", "Al_eq", "beta_transform_T",
    "YM", "YS", "UTS", "El", "HV",
]


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


class BestRewardLogCallback(BaseCallback):
    """Append current best reward after each training step."""

    def __init__(self, seed: int, log_path: str = None, verbose=0):
        super().__init__(verbose)
        self.seed = int(seed)
        self.log_path = Path(log_path or get_default_best_reward_log_path(seed))

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].base_env
        current_step = int(self.num_timesteps)
        best_reward = float(env.get_best_score())
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{current_step}\t{best_reward:.6f}\n")
        return True


class RLCheckpointCallback(BaseCallback):
    """Save model and archive snapshots during RL training."""

    def __init__(self, seed: int, checkpoint_freq: int = RL_CHECKPOINT_FREQ, verbose=0):
        super().__init__(verbose)
        self.seed = int(seed)
        self.checkpoint_freq = max(1, int(checkpoint_freq))
        self.last_checkpoint_step = 0
        self.model_dir = Path("models/rl")
        self.results_dir = Path("results")

    def _on_training_start(self) -> None:
        self.last_checkpoint_step = int(self.num_timesteps)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, step: int) -> None:
        model_path = self.model_dir / f"maskableppo_seed{self.seed}_step{step}.zip"
        latest_model_path = self.model_dir / f"maskableppo_seed{self.seed}.zip"
        env_state_path = self.model_dir / f"maskableppo_seed{self.seed}_step{step}_env_state.pkl"
        latest_env_state_path = self.model_dir / f"maskableppo_seed{self.seed}_env_state.pkl"

        self.model.save(str(model_path))
        self.model.save(str(latest_model_path))

        env = self.training_env.envs[0].base_env
        env_state = env.get_checkpoint_state()
        joblib.dump(env_state, env_state_path)
        joblib.dump(env_state, latest_env_state_path)

        incremental_results = [
            item for item in env.get_result_history()
            if self.last_checkpoint_step < int(item.get("training_step", 0)) <= step
            and float(item.get("score", float("-inf"))) >= 0.0
        ]
        incremental_results.sort(key=lambda item: float(item["score"]), reverse=True)

        save_results_to_csv(
            env,
            self.seed,
            csv_filename=self.results_dir / f"rl_top10_results_seed{self.seed}_step{step}.csv",
            results=incremental_results[:10],
        )
        save_results_to_csv(
            env,
            self.seed,
            csv_filename=self.results_dir / f"rl_top10_results_seed{self.seed}.csv",
        )

    def _on_step(self) -> bool:
        current_step = int(self.num_timesteps)
        if current_step - self.last_checkpoint_step >= self.checkpoint_freq:
            checkpoint_step = (current_step // self.checkpoint_freq) * self.checkpoint_freq
            self._save_checkpoint(checkpoint_step)
            self.last_checkpoint_step = checkpoint_step
        return True


def save_results_to_csv(env: Environment, seed: int, csv_filename=None, results=None):
    """Save top-10 designs and predicted properties to CSV."""
    top_results = results if results is not None else env.get_top_results(k=10)

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
            "train_step": int(item.get("training_step", 0)),
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
    df = df.reindex(columns=RESULT_CSV_COLUMNS)

    os.makedirs("results", exist_ok=True)
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = Path(f"results/rl_top10_results_seed{seed}_{timestamp}.csv")
    else:
        csv_filename = Path(csv_filename)
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    return str(csv_filename)


def get_default_rl_model_path(seed: int) -> str:
    return f"models/rl/maskableppo_seed{int(seed)}.zip"


def get_default_rl_env_state_path(seed: int) -> str:
    return f"models/rl/maskableppo_seed{int(seed)}_env_state.pkl"


def get_default_rl_results_path(seed: int) -> str:
    return f"results/rl_top10_results_seed{int(seed)}.csv"


def get_default_best_reward_log_path(seed: int) -> str:
    return f"logs/rl/best_reward_seed{int(seed)}.txt"


def rl_masked_ppo(
    init_N=20,
    seed=0,
    total_timesteps=10_000,
    n_steps=PPO_N_STEPS,
    batch_size=PPO_BATCH_SIZE,
    checkpoint_freq=RL_CHECKPOINT_FREQ,
    resume=False,
    model_path=None,
    env_state_path=None,
    best_reward_log_path=None,
):
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
    model_path = model_path or get_default_rl_model_path(seed)
    env_state_path = env_state_path or get_default_rl_env_state_path(seed)
    best_reward_log_path = best_reward_log_path or get_default_best_reward_log_path(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if resume and os.path.exists(model_path):
        model = MaskablePPO.load(model_path, env=vec_env, device=device)
        if os.path.exists(env_state_path):
            env_state = joblib.load(env_state_path)
            vec_env.envs[0].base_env.load_checkpoint_state(env_state)
    else:
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
            device=device,
        )

    reward_callback = RewardLogCallback(log_every=10, verbose=0)
    checkpoint_callback = RLCheckpointCallback(seed=seed, checkpoint_freq=checkpoint_freq, verbose=0)
    best_reward_log_callback = BestRewardLogCallback(seed=seed, log_path=best_reward_log_path, verbose=0)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_callback, checkpoint_callback, best_reward_log_callback],
        reset_num_timesteps=not resume,
    )

    env = vec_env.envs[0].base_env
    os.makedirs("models/rl", exist_ok=True)
    model.save(model_path)
    joblib.dump(env.get_checkpoint_state(), env_state_path)
    print(f"final_reward={env.get_best_score():.6f}")
    save_results_to_csv(env, seed, csv_filename=get_default_rl_results_path(seed))
    save_results_to_csv(env, seed)
    return env


if __name__ == "__main__":
    os.environ["RL_N_JOBS"] = str(int(max(1, N_JOBS)))

    base_seed = int(BASE_SEED) if BASE_SEED is not None else random.randint(0, 10000)
    joblib.Parallel(n_jobs=int(max(1, N_JOBS)))(
        joblib.delayed(rl_masked_ppo)(
            init_N=INIT_N,
            seed=base_seed + i * 1000,
            total_timesteps=TOTAL_TIMESTEPS,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            checkpoint_freq=RL_CHECKPOINT_FREQ,
            resume=RESUME_TRAINING,
            model_path=RL_MODEL_PATH if int(max(1, N_JOBS)) == 1 else None,
            env_state_path=RL_ENV_STATE_PATH if int(max(1, N_JOBS)) == 1 else None,
            best_reward_log_path=BEST_REWARD_LOG_PATH if int(max(1, N_JOBS)) == 1 else None,
        ) for i in range(int(max(1, N_JOBS)))
    )
