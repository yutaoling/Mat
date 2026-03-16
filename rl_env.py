from __future__ import annotations

from copy import deepcopy

import numpy as np
from bayes_opt.util import ensure_rng

from surrogate_model import N_PROC_BOOL, N_PROC_SCALAR, PROP
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

        if ep == PROC_EP_WROUGHT:
            if action_idx not in (0, 1):
                raise ValueError(f"is_wrought action must be 0 or 1, got {action_idx}")
            self.__is_wrought = int(action_idx)
            if action_idx == 1:
                self.__proc_bool[0] = 1
            else:
                self.__proc_bool[0] = 0
                self.__proc_scalar[0] = 0.0
                self.__proc_scalar[1] = 0.0

        elif ep == PROC_EP_DEF_TEMP:
            if self.__is_wrought == 1:
                self.__proc_scalar[0] = DEF_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[0] = 0.0

        elif ep == PROC_EP_DEF_STRAIN:
            if self.__is_wrought == 1:
                self.__proc_scalar[1] = DEF_STRAIN_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[1] = 0.0

        elif ep == PROC_EP_HT1:
            if action_idx not in (0, 1):
                raise ValueError(f"HT1 decision action must be 0 or 1, got {action_idx}")
            self.__ht1_decision = int(action_idx)
            self.__proc_bool[1] = int(action_idx)
            if action_idx == 0:
                self.__proc_scalar[2] = 0.0
                self.__proc_scalar[3] = 0.0
                self.__proc_bool[2:5] = 0.0
                self.__ht2_decision = 0
                self.__proc_bool[5] = 0
                self.__proc_scalar[4] = 0.0
                self.__proc_scalar[5] = 0.0
                self.__proc_bool[6:8] = 0.0

        elif ep == PROC_EP_HT1_TEMP:
            if self.__ht1_decision == 1:
                self.__proc_scalar[2] = HT1_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[2] = 0.0

        elif ep == PROC_EP_HT1_TIME:
            if self.__ht1_decision == 1:
                self.__proc_scalar[3] = HT1_TIME_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[3] = 0.0

        elif ep == PROC_EP_HT1_COOL:
            if self.__ht1_decision == 1:
                self.__proc_bool[2:5] = 0.0
                self.__proc_bool[2 + action_idx] = 1.0
            else:
                self.__proc_bool[2:5] = 0.0

        elif ep == PROC_EP_HT2:
            if self.__ht1_decision == 1:
                if action_idx not in (0, 1):
                    raise ValueError(f"HT2 decision action must be 0 or 1, got {action_idx}")
                self.__ht2_decision = int(action_idx)
                self.__proc_bool[5] = int(action_idx)
            else:
                self.__ht2_decision = 0
                self.__proc_bool[5] = 0
            if self.__ht2_decision == 0:
                self.__proc_scalar[4] = 0.0
                self.__proc_scalar[5] = 0.0
                self.__proc_bool[6:8] = 0.0

        elif ep == PROC_EP_HT2_TEMP:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_scalar[4] = HT2_TEMP_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[4] = 0.0

        elif ep == PROC_EP_HT2_TIME:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_scalar[5] = HT2_TIME_CANDIDATES[action_idx]
            else:
                self.__proc_scalar[5] = 0.0

        elif ep == PROC_EP_HT2_COOL:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                self.__proc_bool[6:8] = 0.0
                self.__proc_bool[6 + action_idx] = 1.0
            else:
                self.__proc_bool[6:8] = 0.0

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
        if ep >= PROC_EP_HT1 and self.__ht1_decision == 0:
            return True
        if ep >= PROC_EP_HT2 and self.__ht2_decision == 0:
            return True
        return ep >= PROC_EP_HT2_COOL

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
        if next_ep == PROC_EP_WROUGHT:
            mask[0:2] = True
        elif next_ep == PROC_EP_DEF_TEMP:
            if self.__is_wrought == 1:
                mask[0:len(DEF_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_DEF_STRAIN:
            if self.__is_wrought == 1:
                mask[0:len(DEF_STRAIN_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT1:
            mask[0:2] = True
        elif next_ep == PROC_EP_HT1_TEMP:
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT1_TIME:
            if self.__ht1_decision == 1:
                mask[0:len(HT1_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT1_COOL:
            if self.__ht1_decision == 1:
                mask[0:3] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT2:
            if self.__ht1_decision == 1:
                mask[0:2] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT2_TEMP:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TEMP_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT2_TIME:
            if self.__ht1_decision == 1 and self.__ht2_decision == 1:
                mask[0:len(HT2_TIME_CANDIDATES)] = True
            else:
                mask[0] = True
        elif next_ep == PROC_EP_HT2_COOL:
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
        self.top_k = 10
        self.training_step = 0
        self.top_results = []
        self.result_history = []
        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None
        self.surrogate_buffer = set()
        self.surrogate_buffer_list = []
        self.init_N = init_N
        self.init_surrogate(self.init_N, random_seed)

    def reset_archive(self, init_N=0):
        """Reset archive and best-tracking states to a clean baseline."""
        self.training_step = 0
        self.surrogate_buffer = set()
        self.surrogate_buffer_list = []
        self.top_results = []
        self.result_history = []
        self.best_score = float('-inf')
        self.best_x = None
        self.best_prop = None
        self.init_N = int(init_N)

    def init_world_model(self):
        raw_func, mo_scale = get_mo_ground_truth_func()
        self.raw_mo_func_with_proc = raw_func
        self.mo_scale = np.array(mo_scale, dtype=float)
        self.objective_specs = [OBJECTIVE_SPECS[prop_name] for prop_name in PROP]
        self.mo_weights = np.ones(len(self.objective_specs), dtype=float) / float(len(self.objective_specs))
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: self._score_prediction(
            self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar)
        )

    def get_learning_conditions(self):
        return {
            'objective_specs': {
                prop_name: {
                    'mode': spec.mode,
                    'lower': float(spec.lower),
                    'upper': float(spec.upper),
                    'scale': float(spec.scale),
                }
                for prop_name, spec in zip(PROP, self.objective_specs)
            },
            'mo_weights': np.asarray(self.mo_weights, dtype=float).copy(),
        }

    @staticmethod
    def _objective_distance_reward(distance: float, scale: float) -> float:
        safe_scale = max(float(scale), 1e-8)
        return np.log1p(max(float(distance), 0.0) / safe_scale)

    def _objective_reward(self, value: float, spec: ObjectiveSpec) -> float:
        x = float(value)
        lower = float(spec.lower)
        upper = float(spec.upper)
        scale = max(float(spec.scale), 1e-8)
        mode = spec.mode

        if mode == 'ignore':
            return 0.0
        if mode == 'maximize':
            return self._objective_distance_reward(max(x, 0.0), scale)
        if mode == 'minimize':
            return -self._objective_distance_reward(max(x, 0.0), scale)
        if mode == 'maximize_after':
            if x >= lower:
                return self._objective_distance_reward(x - lower, scale)
            return -(lower - x)
        if mode == 'minimize_before':
            if x <= upper:
                return self._objective_distance_reward(upper - x, scale)
            return -(x - upper)
        if mode == 'at_least':
            return 0.0 if x >= lower else -(lower - x)
        if mode == 'at_most':
            return 0.0 if x <= upper else -(x - upper)
        if mode == 'range':
            if lower <= x <= upper:
                return 0.0
            if x < lower:
                return -(lower - x)
            return -(x - upper)
        raise ValueError(f"Unsupported objective mode: {mode}")

    def _objective_rewards(self, prop) -> np.ndarray:
        arr = np.asarray(prop, dtype=float)
        return np.array(
            [self._objective_reward(arr[i], self.objective_specs[i]) for i in range(len(PROP))],
            dtype=float,
        )

    def _score_prediction(self, prop) -> float:
        objective_rewards = self._objective_rewards(prop)
        reward = float(np.dot(objective_rewards, self.mo_weights))
        reward -= self._constraint_penalty(prop)
        return reward

    def _score_from_components(self, prop, objective_rewards=None) -> float:
        prop_arr = np.asarray(prop, dtype=float)
        reward_vec = (
            np.asarray(objective_rewards, dtype=float)
            if objective_rewards is not None
            else self._objective_rewards(prop_arr)
        )
        reward = float(np.dot(reward_vec, self.mo_weights))
        reward -= self._constraint_penalty(prop_arr)
        return reward

    @staticmethod
    def _is_valid_prop(prop) -> bool:
        """Feasibility check: predicted properties must be finite numbers."""
        arr = np.asarray(prop, dtype=float)
        return bool(np.all(np.isfinite(arr)))

    @staticmethod
    def _constraint_penalty(prop) -> float:
        """Large penalty for physically invalid predictions."""
        arr = np.asarray(prop, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0

        negative_penalty = float(np.sum(np.clip(-arr, 0.0, None)))

        ys_idx = PROP.index('YS')
        uts_idx = PROP.index('UTS')
        ys_uts_penalty = float(max(arr[ys_idx] - arr[uts_idx], 0.0))

        return 10.0 * negative_penalty + 10.0 * ys_uts_penalty

    def _evaluate_design(self, comp, proc_bool, proc_scalar, phase):
        """Predict properties and return (reward, prop, objective_rewards, feasible)."""
        prop = self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase)
        feasible = self._is_valid_prop(prop)
        if not feasible:
            return 0.0, np.asarray(prop), np.zeros(len(PROP), dtype=float), False
        prop_arr = np.asarray(prop, dtype=float)
        objective_rewards = self._objective_rewards(prop_arr)
        reward = float(np.dot(objective_rewards, self.mo_weights))
        reward -= self._constraint_penalty(prop_arr)
        return reward, prop_arr, objective_rewards, True

    def set_mo_weights(self, weights):
        """Set normalized multi-objective weights."""
        w = np.array(weights, dtype=float)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        self.mo_weights = w / w.sum()
        self.func_with_proc = lambda comp, proc_bool, proc_scalar, phase_scalar: self._score_prediction(
            self.raw_mo_func_with_proc(comp, proc_bool, proc_scalar, phase_scalar)
        )
        self._rescore_archive()

    def _rescore_archive(self):
        for item in self.top_results:
            item['score'] = self._score_from_components(
                item['prop'],
                item.get('objective_rewards'),
            )
        self.top_results.sort(key=lambda item: item['score'], reverse=True)
        if len(self.top_results) > self.top_k:
            self.top_results = self.top_results[:self.top_k]
        self._refresh_best_from_top()

    def update_mo_weights(self, method='improvement', window=20, eps=1e-6):
        """Update objective weights from recent improvement ranking."""
        if len(self.surrogate_buffer_list) < 2:
            return
        if method != 'improvement':
            return

        vals = []
        for item in self.surrogate_buffer_list:
            if 'objective_rewards' in item:
                vals.append(np.asarray(item['objective_rewards'], dtype=float))
            else:
                prop = self.raw_mo_func_with_proc(item['comp'], item['proc_bool'], item['proc_scalar'], item['phase'])
                vals.append(self._objective_rewards(prop))
        vals = np.array(vals, dtype=float)

        n = len(vals)
        k = max(1, min(window, n // 2))
        recent = vals[-k:]
        prev = vals[-2 * k:-k] if n >= 2 * k else vals[:k]
        recent_mean = np.mean(recent, axis=0)
        prev_mean = np.mean(prev, axis=0)
        imp = recent_mean - prev_mean
        rank = np.argsort(np.argsort(imp))
        w = np.array([len(imp) - r for r in rank], dtype=float)
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

    def _refresh_best_from_top(self):
        """Keep legacy best_* fields consistent with top_results."""
        if not self.top_results:
            self.best_score = float('-inf')
            self.best_x = None
            self.best_prop = None
            return
        top = self.top_results[0]
        self.best_score = float(top['score'])
        self.best_x = deepcopy(top['x'])
        self.best_prop = np.asarray(top['prop'], dtype=float).copy()

    def set_training_step(self, step):
        self.training_step = max(0, int(step))

    def get_training_step(self):
        return int(self.training_step)

    def _update_top_results(self, score, comp, proc_bool, proc_scalar, phase, prop, objective_rewards):
        """Insert or update one design in top-k leaderboard."""
        design_key = State.encode_key(list(comp) + list(proc_bool) + list(proc_scalar) + list(phase))
        score = float(score)
        x_data = {
            'comp': list(comp),
            'proc_bool': np.asarray(proc_bool, dtype=np.float32).copy(),
            'proc_scalar': np.asarray(proc_scalar, dtype=np.float32).copy(),
            'phase': np.asarray(phase, dtype=np.float32).copy(),
        }
        prop_data = np.asarray(prop, dtype=float).copy()
        objective_reward_data = np.asarray(objective_rewards, dtype=float).copy()

        existing_idx = None
        for i, item in enumerate(self.top_results):
            if item['key'] == design_key:
                existing_idx = i
                break

        if existing_idx is not None:
            if score <= self.top_results[existing_idx]['score']:
                self._refresh_best_from_top()
                return
            self._record_result_history(score, comp, proc_bool, proc_scalar, phase, prop, objective_rewards)
            self.top_results[existing_idx] = {
                'key': design_key,
                'score': score,
                'training_step': int(self.training_step),
                'x': x_data,
                'prop': prop_data,
                'objective_rewards': objective_reward_data,
            }
        else:
            self._record_result_history(score, comp, proc_bool, proc_scalar, phase, prop, objective_rewards)
            self.top_results.append({
                'key': design_key,
                'score': score,
                'training_step': int(self.training_step),
                'x': x_data,
                'prop': prop_data,
                'objective_rewards': objective_reward_data,
            })

        self.top_results.sort(key=lambda item: item['score'], reverse=True)
        if len(self.top_results) > self.top_k:
            self.top_results = self.top_results[:self.top_k]
        self._refresh_best_from_top()

    def _record_result_history(self, score, comp, proc_bool, proc_scalar, phase, prop, objective_rewards):
        self.result_history.append({
            'key': State.encode_key(list(comp) + list(proc_bool) + list(proc_scalar) + list(phase)),
            'score': float(score),
            'training_step': int(self.training_step),
            'x': {
                'comp': list(comp),
                'proc_bool': np.asarray(proc_bool, dtype=np.float32).copy(),
                'proc_scalar': np.asarray(proc_scalar, dtype=np.float32).copy(),
                'phase': np.asarray(phase, dtype=np.float32).copy(),
            },
            'prop': np.asarray(prop, dtype=float).copy(),
            'objective_rewards': np.asarray(objective_rewards, dtype=float).copy(),
        })

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
                score, prop, objective_rewards, feasible = self._evaluate_design(comp, proc_bool, proc_scalar, phase)
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
                'objective_rewards': np.asarray(objective_rewards, dtype=float).copy(),
            })
            self._update_top_results(score, comp, proc_bool, proc_scalar, phase, prop, objective_rewards)

    def reset(self):
        """Reset to an empty decision state."""
        return State(if_init=True)

    def step(self, state: State, action_idx: int):
        """Environment step with dense scalarized performance reward."""
        self.training_step += 1
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
        ht1_cooling = proc_bool[2:5].tolist()
        ht2_cooling = proc_bool[6:8].tolist()
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
        phase = calculate_phase_scalar(comp)

        reward, prop, objective_rewards, feasible = self._evaluate_design(comp, proc_bool, proc_scalar, phase)
        '''print(
            f"[step={next_state.get_episode_count()}] "
            f"comp={comp}; "
            f"DEF(is_wrought={is_wrought}, temp={float(proc_scalar[0]):.3f}, strain={float(proc_scalar[1]):.3f}); "
            f"HT1(decision={ht1_decision}, temp={float(proc_scalar[2]):.3f}, time={float(proc_scalar[3]):.3f}, cooling={ht1_cooling_display}); "
            f"HT2(decision={ht2_decision}, temp={float(proc_scalar[4]):.3f}, time={float(proc_scalar[5]):.3f}, cooling={ht2_cooling_display}); "
            f"reward={float(reward):.6f}"
        )'''

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
                    'objective_rewards': np.asarray(objective_rewards, dtype=float).copy(),
                })

            self._update_top_results(reward, comp, proc_bool, proc_scalar, phase, prop, objective_rewards)

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

    def get_top_results(self, k=10):
        k = max(1, int(k))
        top = self.top_results[:k]
        out = []
        for item in top:
            out.append({
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': deepcopy(item['x']),
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })
        return out

    def get_result_history(self):
        out = []
        for item in self.result_history:
            out.append({
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': deepcopy(item['x']),
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })
        return out

    def get_checkpoint_state(self):
        """Return archive state needed to resume RL search consistently."""
        top_results = []
        for item in self.top_results:
            top_results.append({
                'key': item['key'],
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': {
                    'comp': list(item['x']['comp']),
                    'proc_bool': np.asarray(item['x']['proc_bool'], dtype=np.float32).copy(),
                    'proc_scalar': np.asarray(item['x']['proc_scalar'], dtype=np.float32).copy(),
                    'phase': np.asarray(item['x']['phase'], dtype=np.float32).copy(),
                },
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })

        surrogate_buffer_list = []
        for item in self.surrogate_buffer_list:
            surrogate_buffer_list.append({
                'comp': list(item['comp']),
                'proc_bool': np.asarray(item['proc_bool'], dtype=np.float32).copy(),
                'proc_scalar': np.asarray(item['proc_scalar'], dtype=np.float32).copy(),
                'phase': np.asarray(item['phase'], dtype=np.float32).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })

        result_history = []
        for item in self.result_history:
            result_history.append({
                'key': item['key'],
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': {
                    'comp': list(item['x']['comp']),
                    'proc_bool': np.asarray(item['x']['proc_bool'], dtype=np.float32).copy(),
                    'proc_scalar': np.asarray(item['x']['proc_scalar'], dtype=np.float32).copy(),
                    'phase': np.asarray(item['x']['phase'], dtype=np.float32).copy(),
                },
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })

        return {
            'top_results': top_results,
            'result_history': result_history,
            'surrogate_buffer': sorted(self.surrogate_buffer),
            'surrogate_buffer_list': surrogate_buffer_list,
            'mo_weights': np.asarray(self.mo_weights, dtype=float).copy(),
            'learning_conditions': self.get_learning_conditions(),
            'init_N': int(self.init_N),
            'training_step': int(self.training_step),
        }

    def load_checkpoint_state(self, checkpoint_state):
        """Restore archive state from a previous checkpoint."""
        self.top_results = []
        for item in checkpoint_state.get('top_results', []):
            self.top_results.append({
                'key': item['key'],
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': {
                    'comp': list(item['x']['comp']),
                    'proc_bool': np.asarray(item['x']['proc_bool'], dtype=np.float32).copy(),
                    'proc_scalar': np.asarray(item['x']['proc_scalar'], dtype=np.float32).copy(),
                    'phase': np.asarray(item['x']['phase'], dtype=np.float32).copy(),
                },
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', self._objective_rewards(item['prop'])), dtype=float).copy(),
            })

        self.surrogate_buffer = set(checkpoint_state.get('surrogate_buffer', []))
        self.surrogate_buffer_list = []
        for item in checkpoint_state.get('surrogate_buffer_list', []):
            self.surrogate_buffer_list.append({
                'comp': list(item['comp']),
                'proc_bool': np.asarray(item['proc_bool'], dtype=np.float32).copy(),
                'proc_scalar': np.asarray(item['proc_scalar'], dtype=np.float32).copy(),
                'phase': np.asarray(item['phase'], dtype=np.float32).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', np.zeros(len(PROP))), dtype=float).copy(),
            })

        self.result_history = []
        for item in checkpoint_state.get('result_history', []):
            self.result_history.append({
                'key': item['key'],
                'score': float(item['score']),
                'training_step': int(item.get('training_step', 0)),
                'x': {
                    'comp': list(item['x']['comp']),
                    'proc_bool': np.asarray(item['x']['proc_bool'], dtype=np.float32).copy(),
                    'proc_scalar': np.asarray(item['x']['proc_scalar'], dtype=np.float32).copy(),
                    'phase': np.asarray(item['x']['phase'], dtype=np.float32).copy(),
                },
                'prop': np.asarray(item['prop'], dtype=float).copy(),
                'objective_rewards': np.asarray(item.get('objective_rewards', self._objective_rewards(item['prop'])), dtype=float).copy(),
            })

        self.init_N = int(checkpoint_state.get('init_N', self.init_N))
        self.training_step = int(checkpoint_state.get('training_step', self.training_step))
        learning_conditions = checkpoint_state.get('learning_conditions')
        if learning_conditions and 'objective_specs' in learning_conditions:
            objective_specs = []
            for prop_name in PROP:
                spec_dict = learning_conditions['objective_specs'].get(prop_name)
                if spec_dict is None:
                    objective_specs.append(OBJECTIVE_SPECS[prop_name])
                else:
                    objective_specs.append(
                        ObjectiveSpec(
                            mode=spec_dict['mode'],
                            lower=float(spec_dict['lower']),
                            upper=float(spec_dict['upper']),
                            scale=float(spec_dict['scale']),
                        )
                    )
            self.objective_specs = objective_specs
        if 'mo_weights' in checkpoint_state:
            self.set_mo_weights(checkpoint_state['mo_weights'])
        self._refresh_best_from_top()

    def get_exp_number(self):
        return len(self.surrogate_buffer) - self.init_N


if __name__ == '__main__':
    pass
