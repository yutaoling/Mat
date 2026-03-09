from __future__ import annotations

from copy import deepcopy

import numpy as np
from bayes_opt.util import ensure_rng

from surrogate_model import N_PROC_BOOL, N_PROC_SCALAR
from rl_constants import *


ActionType = float


class State:
    """Sequential design state for composition and process decisions."""

    def __init__(
        self,
        if_init: bool = False,
        previous_state: State = None,
        action: ActionType = None,
        action_idx: int = None,
    ):
        if if_init:
            self.__composition = [0.0] * ELEM_N
            self.__episode_count = -1
            self.__max_episode_len = MAX_EPISODE_LEN
            self.__proc_bool = np.zeros(N_PROC_BOOL, dtype=np.float32)
            self.__proc_scalar = np.zeros(N_PROC_SCALAR, dtype=np.float32)
            self.__is_wrought = 0
            self.__ht1_decision = 0
            self.__ht2_decision = 0
            return

        self.__composition = deepcopy(previous_state.get_composition())
        self.__episode_count = previous_state.get_episode_count() + 1
        self.__max_episode_len = MAX_EPISODE_LEN
        self.__proc_bool = deepcopy(previous_state.get_proc_bool())
        self.__proc_scalar = deepcopy(previous_state.get_proc_scalar())
        self.__is_wrought = previous_state.get_is_wrought()
        self.__ht1_decision = previous_state.get_ht1_decision()
        self.__ht2_decision = previous_state.get_ht2_decision()

        if self.__episode_count < COMP_EPISODE_LEN:
            sub_idx = self.__episode_count
            _min = max(
                COMP_MIN_LIMITS[sub_idx],
                1 - sum(self.__composition[:sub_idx]) - sum(COMP_MAX_LIMITS[sub_idx + 1:]),
            )
            _max = min(
                COMP_MAX_LIMITS[sub_idx],
                1 - sum(self.__composition[:sub_idx]) - sum(COMP_MIN_LIMITS[sub_idx + 1:]),
            )
            _min = round(_min, COMPOSITION_ROUNDUP_DIGITS)
            _max = round(_max, COMPOSITION_ROUNDUP_DIGITS)
            action = min(max(float(action), _min), _max)
            self.__composition[sub_idx] = action

            if sub_idx == COMP_EPISODE_LEN - 1:
                last = 1.0 - sum(self.__composition[:-1])
                if 0 < last < MANDATORY_MIN:
                    last = 0.0
                self.__composition[-1] = round(last, COMPOSITION_ROUNDUP_DIGITS)

            for idx in range(len(self.__composition)):
                self.__composition[idx] = round(self.__composition[idx], COMPOSITION_ROUNDUP_DIGITS)
        else:
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

    def get_ht1_decision(self):
        return self.__ht1_decision

    def get_ht2_decision(self):
        return self.__ht2_decision

    def _update_proc_params(self, action_idx: int):
        ep = self.__episode_count

        if ep == 18:
            if action_idx not in (0, 1):
                raise ValueError(f"is_wrought action must be 0 or 1, got {action_idx}")
            self.__is_wrought = int(action_idx)
            if action_idx == 1:
                self.__proc_bool[0] = 0
                self.__proc_bool[1] = 1
            else:
                self.__proc_bool[0] = 1
                self.__proc_bool[1] = 0
                self.__proc_scalar[0] = 0.0
                self.__proc_scalar[1] = 0.0

        elif ep == 19:
            if self.__is_wrought == 1:
                self.__proc_scalar[0] = DEF_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[0] = 0.0

        elif ep == 20:
            if self.__is_wrought == 1:
                self.__proc_scalar[1] = DEF_STRAIN_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[1] = 0.0

        elif ep == 21:
            if action_idx not in (0, 1):
                raise ValueError(f"HT1 decision action must be 0 or 1, got {action_idx}")
            self.__ht1_decision = int(action_idx)
            self.__proc_bool[2] = int(action_idx)
            if action_idx == 0:
                self.__proc_scalar[2] = 0.0
                self.__proc_scalar[3] = 0.0
                self.__proc_bool[3:6] = 0.0
                self.__ht2_decision = 0
                self.__proc_bool[6] = 0
                self.__proc_scalar[4] = 0.0
                self.__proc_scalar[5] = 0.0
                self.__proc_bool[7:9] = 0.0

        elif ep == 22:
            if self.__ht1_decision == 1:
                self.__proc_scalar[2] = HT1_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[2] = 0.0

        elif ep == 23:
            if self.__ht1_decision == 1:
                self.__proc_scalar[3] = HT1_TIME_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[3] = 0.0

        elif ep == 24:
            if self.__ht1_decision == 1:
                self.__proc_bool[3:6] = 0.0
                self.__proc_bool[3 + action_idx] = 1.0
            else:
                self.__proc_bool[3:6] = 0.0

        elif ep == 25:
            if self.__ht1_decision == 1:
                if action_idx not in (0, 1):
                    raise ValueError(f"HT2 decision action must be 0 or 1, got {action_idx}")
                self.__ht2_decision = int(action_idx)
                self.__proc_bool[6] = int(action_idx)
            else:
                self.__ht2_decision = 0
                self.__proc_bool[6] = 0
            if self.__ht2_decision == 0:
                self.__proc_scalar[4] = 0.0
                self.__proc_scalar[5] = 0.0
                self.__proc_bool[7:9] = 0.0

        elif ep == 26:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_scalar[4] = HT2_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[4] = 0.0

        elif ep == 27:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_scalar[5] = HT2_TIME_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[5] = 0.0

        elif ep == 28:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_bool[7:9] = 0.0
                self.__proc_bool[7 + action_idx] = 1.0
            else:
                self.__proc_bool[7:9] = 0.0

    def repr(self):
        """Return model observation vector."""
        state = deepcopy(self.__composition)
        state.append(self.__episode_count / EPISODE_COUNT_MAX)
        state.extend(self.__proc_bool.tolist())

        state.append(np.log(self.__proc_scalar[0] + 1) / np.log(DEF_TEMP_MAX + 1) if self.__proc_scalar[0] > 0 else 0.0)
        state.append(self.__proc_scalar[1] / DEF_STRAIN_MAX)
        state.append(np.log(self.__proc_scalar[2] + 1) / np.log(HT1_TEMP_MAX + 1) if self.__proc_scalar[2] > 0 else 0.0)
        state.append(np.log(self.__proc_scalar[3] + 1) / np.log(HT1_TIME_MAX + 1) if self.__proc_scalar[3] > 0 else 0.0)
        state.append(np.log(self.__proc_scalar[4] + 1) / np.log(HT2_TEMP_MAX + 1) if self.__proc_scalar[4] > 0 else 0.0)
        state.append(np.log(self.__proc_scalar[5] + 1) / np.log(HT2_TIME_MAX + 1) if self.__proc_scalar[5] > 0 else 0.0)

        phase_scalar = calculate_phase_scalar(self.__composition)
        mo_eq_norm = (phase_scalar[0] - PHASE_NORMALIZATION['mo_eq_min']) / (PHASE_NORMALIZATION['mo_eq_max'] - PHASE_NORMALIZATION['mo_eq_min'])
        al_eq_norm = (phase_scalar[1] - PHASE_NORMALIZATION['al_eq_min']) / (PHASE_NORMALIZATION['al_eq_max'] - PHASE_NORMALIZATION['al_eq_min'])
        beta_T_norm = (phase_scalar[2] - PHASE_NORMALIZATION['beta_transform_T_min']) / (PHASE_NORMALIZATION['beta_transform_T_max'] - PHASE_NORMALIZATION['beta_transform_T_min'])

        state.extend([
            np.clip(mo_eq_norm, 0.0, 1.0),
            np.clip(al_eq_norm, 0.0, 1.0),
            np.clip(beta_T_norm, 0.0, 1.0),
        ])
        return np.array(state, dtype=np.float32)

    def done(self):
        ep = self.__episode_count
        if ep >= self.__max_episode_len - 1:
            return True
        if ep < COMP_EPISODE_LEN:
            return False
        if ep >= 21 and self.__ht1_decision == 0:
            return True
        if ep >= 25 and self.__ht2_decision == 0:
            return True
        return ep >= 28

    def get_action_idx_limits(self):
        if self.__episode_count < COMP_EPISODE_LEN - 1:
            elem_index = self.__episode_count + 1
            _min = max(
                COMP_MIN_LIMITS[elem_index],
                1 - sum(self.__composition[:elem_index]) - sum(COMP_MAX_LIMITS[elem_index + 1:]),
            )
            _max = min(
                COMP_MAX_LIMITS[elem_index],
                1 - sum(self.__composition[:elem_index]) - sum(COMP_MIN_LIMITS[elem_index + 1:]),
            )
            _min = round(_min, COMPOSITION_ROUNDUP_DIGITS)
            _max = round(_max, COMPOSITION_ROUNDUP_DIGITS)
            if _max < MANDATORY_MIN:
                _max = 0.0
                _min = 0.0
            elif 0 < _min < MANDATORY_MIN:
                _min = MANDATORY_MIN
            comp_actions = np.asarray(COMP_ACTIONS, dtype=np.float64)
            ai_low = int(np.searchsorted(comp_actions, _min, side='left'))
            ai_high = int(np.searchsorted(comp_actions, _max, side='right') - 1)
            ai_low = max(0, min(ai_low, len(COMP_ACTIONS) - 1))
            ai_high = max(0, min(ai_high, len(COMP_ACTIONS) - 1))
            if ai_high < ai_low:
                ai_high = ai_low
            return ai_low, ai_high

        mask = self.get_action_mask()
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            raise ValueError(f"No valid actions at episode {self.__episode_count}")
        return int(valid_actions[0]), int(valid_actions[-1])

    def get_action_mask(self):
        """Return boolean action mask for current step."""
        if self.__episode_count < COMP_EPISODE_LEN - 1:
            mask = np.zeros(COMP_ACTIONS_COUNT, dtype=bool)
            ai_low, ai_high = self.get_action_idx_limits()
            mask[ai_low:ai_high + 1] = True
            return mask

        next_ep = self.__episode_count + 1
        mask = np.zeros(max(COMP_ACTIONS_COUNT, MAX_PROC_ACTION_SPACE), dtype=bool)
        if next_ep == 18:
            mask[0:2] = True
        elif next_ep == 19:
            if self.__is_wrought == 1:
                mask[0:len(DEF_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 20:
            if self.__is_wrought == 1:
                mask[0:len(DEF_STRAIN_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 21:
            mask[0:2] = True
        elif next_ep == 22:
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 23:
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 24:
            if self.__ht1_decision == 1:
                mask[0:3] = True
            else:
                mask[0] = True
        elif next_ep == 25:
            if self.__ht1_decision == 1:
                mask[0:2] = True
            else:
                mask[0] = True
        elif next_ep == 26:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 27:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == 28:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:2] = True
            else:
                mask[0] = True
        return mask

    def generate_random_action(self, random_seed=None) -> ActionType:
        """Sample a random valid composition action value."""
        random_state = random_seed if isinstance(random_seed, np.random.RandomState) else ensure_rng(random_seed)
        comp_min_idx, comp_max_idx = self.get_action_idx_limits()
        if comp_max_idx < comp_min_idx:
            raise ValueError(f"Invalid composition action range: [{comp_min_idx}, {comp_max_idx}]")
        rand_comp_idx = random_state.randint(comp_min_idx, comp_max_idx + 1)
        return COMP_ACTIONS[rand_comp_idx]

    @staticmethod
    def encode_key(x):
        x_rounded = [round(float(val), COMPOSITION_ROUNDUP_DIGITS) for val in x]
        return STATE_DELIMETER_CHAR.join(map(str, x_rounded))

    @staticmethod
    def decode_key(key: str):
        return np.array(list(map(float, key.split(STATE_DELIMETER_CHAR)))).round(ROUND_DIGIT)


class Environment:
    """Environment used by PPO and legacy callers."""

    def __init__(self, init_N=50, random_seed=None):
        self.init_world_model()
        self.state_dim = len(self.reset().repr())
        self.act_dim = max(COMP_ACTIONS_COUNT, MAX_PROC_ACTION_SPACE)
        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None
        self.surrogate_buffer = set()
        self.surrogate_buffer_list = []
        self.init_N = init_N
        self.init_surrogate(self.init_N, random_seed)

    def reset_archive(self, init_N=0):
        """Reset archive and best-tracking states to a clean baseline."""
        self.surrogate_buffer = set()
        self.surrogate_buffer_list = []
        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None
        self.init_N = int(init_N)

    def init_world_model(self):
        raw_func, mo_scale = get_mo_ground_truth_func()
        self.raw_mo_func_with_proc = raw_func
        self.mo_scale = np.array(mo_scale, dtype=float)
        self.mo_weights = np.ones(len(self.mo_scale), dtype=float) / float(len(self.mo_scale))
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: float(
            np.dot(self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar) / self.mo_scale, self.mo_weights)
        )

    @staticmethod
    def _is_valid_prop(prop) -> bool:
        """Feasibility check: predicted properties must be finite numbers."""
        arr = np.asarray(prop, dtype=float)
        return bool(np.all(np.isfinite(arr)))

    def _evaluate_design(self, comp, proc_bool, proc_scalar, phase):
        """Predict properties and return (reward, prop, feasible)."""
        prop = self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase)
        feasible = self._is_valid_prop(prop)
        if not feasible:
            return 0.0, np.asarray(prop), False
        reward = float(np.dot(np.asarray(prop, dtype=float) / self.mo_scale, self.mo_weights))
        return reward, np.asarray(prop), True

    def set_mo_weights(self, weights):
        """Set normalized multi-objective weights."""
        w = np.array(weights, dtype=float)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.mo_weights = w / w.sum()
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: float(
            np.dot(self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar) / self.mo_scale, self.mo_weights)
        )

    def update_mo_weights(self, method='improvement', window=20, eps=1e-6):
        """Update objective weights from recent improvement ranking."""
        if len(self.surrogate_buffer_list) < 2:
            return
        if method != 'improvement':
            return

        vals = []
        for item in self.surrogate_buffer_list:
            vals.append(self.raw_mo_func_with_proc(item['comp'], item['proc_bool'], item['proc_scalar'], item['phase']))
        vals = np.array(vals)

        n = len(vals)
        k = max(1, min(window, n // 2))
        recent = vals[-k:]
        prev = vals[-2 * k:-k] if n >= 2 * k else vals[:k]
        recent_mean = np.mean(recent / self.mo_scale, axis=0)
        prev_mean = np.mean(prev / self.mo_scale, axis=0)
        imp = recent_mean - prev_mean
        rank = np.argsort(np.argsort(imp))
        weight_list = [1, 1, 1, 1, 1]
        w = np.array([weight_list[r] for r in rank], dtype=float)
        self.set_mo_weights(w)

    def _normalize_and_constrain_composition(self, comp):
        comp = [max(COMP_MIN_LIMITS[i], min(COMP_MAX_LIMITS[i], float(v))) for i, v in enumerate(comp)]
        s = sum(comp)
        if s <= 0:
            out = [0.0] * ELEM_N
            out[0] = 1.0
            return out
        comp = [v / s for v in comp]

        for _ in range(20):
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
            s = sum(comp)
            if s <= 0:
                comp = [0.0] * ELEM_N
                comp[0] = 1.0
                break
            comp = [v / s for v in comp]
        return [round(v, COMPOSITION_ROUNDUP_DIGITS) for v in comp]

    def _random_composition(self, rng):
        comp = [rng.uniform(COMP_MIN_LIMITS[i], COMP_MAX_LIMITS[i]) for i in range(ELEM_N)]
        return self._normalize_and_constrain_composition(comp)

    def init_surrogate(self, init_N, seed):
        """Initialize archive with random feasible designs."""
        rng = ensure_rng(seed)
        while len(self.surrogate_buffer_list) < init_N:
            comp = self._random_composition(rng)
            proc_bool = np.zeros(N_PROC_BOOL, dtype=np.float32)
            proc_scalar = np.zeros(N_PROC_SCALAR, dtype=np.float32)
            phase = calculate_phase_scalar(comp)
            design_key = State.encode_key(list(comp) + list(proc_bool) + list(proc_scalar) + list(phase))
            if design_key in self.surrogate_buffer:
                continue

            try:
                score, prop, feasible = self._evaluate_design(comp, proc_bool, proc_scalar, phase)
            except Exception:
                continue
            if not feasible:
                continue
            self.surrogate_buffer.add(design_key)
            self.surrogate_buffer_list.append({
                'comp': comp,
                'proc_bool': proc_bool,
                'proc_scalar': proc_scalar,
                'phase': phase,
            })
            if score > self.best_score:
                self.best_score = score
                self.best_x = {
                    'comp': comp,
                    'proc_bool': proc_bool,
                    'proc_scalar': proc_scalar,
                    'phase': phase,
                }
                self.best_prop = prop

    def reset(self):
        """Reset to an empty decision state."""
        return State(if_init=True)

    def step(self, state: State, action_idx: int):
        """Environment step with dense scalarized performance reward."""
        mask = state.get_action_mask()
        if action_idx < 0 or action_idx >= len(mask) or not mask[action_idx]:
            raise ValueError(
                f"Invalid action {action_idx} at episode {state.get_episode_count()}, "
                f"valid={np.where(mask)[0].tolist()}"
            )

        if state.get_episode_count() < COMP_EPISODE_LEN - 1:
            next_state = State(previous_state=state, action=COMP_ACTIONS[action_idx])
        else:
            next_state = State(previous_state=state, action_idx=action_idx)

        comp = next_state.get_composition()
        proc_bool = next_state.get_proc_bool()
        proc_scalar = next_state.get_proc_scalar()
        is_wrought = next_state.get_is_wrought()
        ht1_decision = next_state.get_ht1_decision()
        ht2_decision = next_state.get_ht2_decision()
        ht1_cooling = proc_bool[3:6].tolist()
        ht2_cooling = proc_bool[7:9].tolist()
        if ht1_decision == 0:
            ht1_cooling_display = "N/A"
        elif sum(ht1_cooling) == 0:
            ht1_cooling_display = "PENDING"
        else:
            ht1_cooling_display = ht1_cooling

        if ht2_decision == 0:
            ht2_cooling_display = "N/A"
        elif sum(ht2_cooling) == 0:
            ht2_cooling_display = "PENDING"
        else:
            ht2_cooling_display = ht2_cooling
        print(
            f"[step={next_state.get_episode_count()}] "
            f"comp={comp}; "
            f"DEF(is_wrought={is_wrought}, temp={float(proc_scalar[0]):.3f}, strain={float(proc_scalar[1]):.3f}); "
            f"HT1(decision={ht1_decision}, temp={float(proc_scalar[2]):.3f}, time={float(proc_scalar[3]):.3f}, cooling={ht1_cooling_display}); "
            f"HT2(decision={ht2_decision}, temp={float(proc_scalar[4]):.3f}, time={float(proc_scalar[5]):.3f}, cooling={ht2_cooling_display})"
        )
        phase = calculate_phase_scalar(comp)

        reward, prop, feasible = self._evaluate_design(comp, proc_bool, proc_scalar, phase)

        done = next_state.done()
        if done and feasible:
            design_key = State.encode_key(list(comp) + list(proc_bool) + list(proc_scalar) + list(phase))
            if design_key not in self.surrogate_buffer:
                self.surrogate_buffer.add(design_key)
                self.surrogate_buffer_list.append({
                    'comp': comp,
                    'proc_bool': proc_bool,
                    'proc_scalar': proc_scalar,
                    'phase': phase,
                })

            if reward > self.best_score:
                self.best_score = reward
                self.best_x = {
                    'comp': comp,
                    'proc_bool': proc_bool,
                    'proc_scalar': proc_scalar,
                    'phase': phase,
                }
                self.best_prop = prop

            try:
                self.update_mo_weights(method='improvement', window=50)
            except Exception:
                pass

        return next_state, float(reward), bool(done)

    def sample_action(self, state: State):
        """Sample one valid action index."""
        ai_low, ai_high = state.get_action_idx_limits()
        return np.random.randint(ai_low, ai_high + 1)

    def get_best_score(self):
        return self.best_score

    def get_best_x(self):
        return self.best_x

    def get_best_prop(self):
        return self.best_prop

    def get_exp_number(self):
        return len(self.surrogate_buffer) - self.init_N


if __name__ == '__main__':
    pass
