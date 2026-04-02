from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class InferenceController:
    """Chunked inference controller with optional validation branch.

    Notes:
    - `run_full_chunk_fn` is the *execution* chunk callback (historical name kept for compatibility).
    - `run_light_chunk_fn` is an optional validator callback when `enable_validation=True`.
    """

    def __init__(
        self,
        run_full_chunk_fn: Callable[[Dict, str], List[np.ndarray]],
        run_light_chunk_fn: Optional[Callable[[Dict, str], List[np.ndarray]]] = None,
        *,
        use_temporal_fusion: bool,
        full_num_actions_chunk: int,
        light_num_actions_chunk: int,
        enable_validation: bool = False,
        prime_light_cache_fn: Optional[Callable[[Dict, str], Any]] = None,
        deviation_threshold: float = 0.1,
        reanchor_step_multiple: int = 0,
    ) -> None:
        self.run_full_chunk_fn = run_full_chunk_fn
        self.run_light_chunk_fn = run_light_chunk_fn
        self.use_temporal_fusion = bool(use_temporal_fusion)
        self.full_num_actions_chunk = max(1, int(full_num_actions_chunk))
        self.light_num_actions_chunk = max(1, int(light_num_actions_chunk))
        self.enable_validation = bool(enable_validation)
        self.prime_light_cache_fn = prime_light_cache_fn
        self.deviation_threshold = float(deviation_threshold)
        # <=0 means no additional control: every validation point can trigger re-anchor.
        # >0 means only validation at step == reanchor_step_multiple * light_num_actions_chunk can re-anchor.
        self.reanchor_step_multiple = int(reanchor_step_multiple)

        self._step_in_full_chunk = 0
        self._next_validation_step = self._initial_validation_step()
        self._last_deviation: Optional[float] = None
        self._cached_full_chunk_actions: Optional[np.ndarray] = None

    def _initial_validation_step(self) -> int:
        """Return the first validation step index inside an execution chunk.

        When `reanchor_step_multiple > 1`, early validations before `n * light_chunk`
        are skipped for efficiency.
        """
        start_multiple = 1
        if self.reanchor_step_multiple > 1:
            start_multiple = self.reanchor_step_multiple
        return max(1, int(start_multiple * self.light_num_actions_chunk))

    def reset(self) -> None:
        """Reset per-episode controller state."""
        self._step_in_full_chunk = 0
        self._next_validation_step = self._initial_validation_step()
        self._last_deviation = None
        self._cached_full_chunk_actions = None

    @staticmethod
    def _clone_observation(obs: Dict) -> Dict:
        return deepcopy(obs)

    @staticmethod
    def _calculate_deviation(action_a: np.ndarray, action_b: np.ndarray) -> float:
        """MSE deviation for optional validation."""
        a = np.asarray(action_a, dtype=np.float32)
        b = np.asarray(action_b, dtype=np.float32)
        mse = float(np.mean((a - b) ** 2))
        return mse

    def _run_full_model_chunk(self, observation: Dict, task_description: str) -> List[np.ndarray]:
        """Run one execution chunk query (historically named `full`)."""
        obs = self._clone_observation(observation)
        # Full execution anchors reset first-frame cache by design.
        obs["is_first_frame"] = True
        return self.run_full_chunk_fn(obs, task_description)

    def _run_lightweight_chunk(self, observation: Dict, task_description: str) -> List[np.ndarray]:
        if self.run_light_chunk_fn is None:
            raise RuntimeError("Light-model callback is missing while temporal fusion validation is enabled.")
        obs = self._clone_observation(observation)
        obs["is_first_frame"] = False
        return self.run_light_chunk_fn(obs, task_description)

    @staticmethod
    def _to_chunk_array(actions: List[np.ndarray]) -> np.ndarray:
        if len(actions) == 0:
            raise ValueError("Policy returned empty action chunk.")
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk shape (N, A), got {chunk.shape}.")
        return chunk

    @staticmethod
    def _first_action(actions: List[np.ndarray]) -> np.ndarray:
        if len(actions) == 0:
            raise ValueError("Policy returned empty action chunk.")
        return np.asarray(actions[0], dtype=np.float32)

    @staticmethod
    def _slice_chunk_with_tail_pad(chunk: np.ndarray, start: int, length: int) -> np.ndarray:
        start = int(max(0, start))
        length = int(max(1, length))
        if chunk.shape[0] == 0:
            raise ValueError("Cannot slice from empty chunk.")
        if start >= chunk.shape[0]:
            return np.repeat(chunk[-1:], length, axis=0)

        sliced = chunk[start : start + length]
        if sliced.shape[0] == length:
            return sliced

        pad = np.repeat(sliced[-1:], length - sliced.shape[0], axis=0)
        return np.concatenate([sliced, pad], axis=0)

    def _anchor_full_chunk(self, observation: Dict, task_description: str) -> np.ndarray:
        """Refresh execution chunk cache at current anchor frame."""
        full_actions = self._run_full_model_chunk(observation, task_description)
        full_chunk = self._to_chunk_array(full_actions)
        if full_chunk.shape[0] < self.full_num_actions_chunk:
            full_chunk = self._slice_chunk_with_tail_pad(full_chunk, 0, self.full_num_actions_chunk)
        else:
            full_chunk = full_chunk[: self.full_num_actions_chunk]

        self._cached_full_chunk_actions = full_chunk
        self._step_in_full_chunk = 0
        self._next_validation_step = self._initial_validation_step()
        self._last_deviation = None

        if self.use_temporal_fusion and self.prime_light_cache_fn is not None:
            # Optional validator/light cache prime using the same anchor frame.
            prime_obs = self._clone_observation(observation)
            prime_obs["is_first_frame"] = True
            self.prime_light_cache_fn(prime_obs, task_description)

        return full_chunk

    def step(self, observation: Dict, task_description: str) -> Dict[str, Any]:
        """Run one env-step policy decision.

        Returns keys:
        - `action`: selected action for current step
        - `used_mode`: `full` | `lightweight` | `full_reanchor`
        - `deviation`: optional float
        - `full_chunk_step`: current step index inside cached full chunk
        """
        obs = self._clone_observation(observation)

        if not self.use_temporal_fusion:
            full_actions = self._run_full_model_chunk(obs, task_description)
            return {
                "action": self._first_action(full_actions),
                "used_mode": "full",
                "deviation": None,
                "full_chunk_step": 0,
            }

        # Start a new execution chunk when needed.
        if self._cached_full_chunk_actions is None or self._step_in_full_chunk >= self.full_num_actions_chunk:
            full_chunk = self._anchor_full_chunk(obs, task_description)
            selected_action = np.asarray(full_chunk[0], dtype=np.float32)
            self._step_in_full_chunk = 1
            return {
                "action": selected_action,
                "used_mode": "full",
                "deviation": None,
                "full_chunk_step": 0,
            }

        # Execution always comes from the cached execution chunk at current relative step.
        current_step = self._step_in_full_chunk
        selected_action = np.asarray(self._cached_full_chunk_actions[current_step], dtype=np.float32)

        if self.enable_validation:
            should_validate_now = current_step == self._next_validation_step and current_step < self.full_num_actions_chunk
            if should_validate_now:
                light_actions = self._run_lightweight_chunk(obs, task_description)
                light_chunk = self._to_chunk_array(light_actions)
                light_chunk = self._slice_chunk_with_tail_pad(light_chunk, 0, self.light_num_actions_chunk)

                ref_chunk = self._slice_chunk_with_tail_pad(
                    self._cached_full_chunk_actions,
                    start=current_step,
                    length=self.light_num_actions_chunk,
                )
                deviation = self._calculate_deviation(ref_chunk, light_chunk)
                self._last_deviation = deviation

                allow_reanchor_now = True
                if self.reanchor_step_multiple > 0:
                    allowed_step = self.reanchor_step_multiple * self.light_num_actions_chunk
                    allow_reanchor_now = current_step == allowed_step

                if deviation > self.deviation_threshold and allow_reanchor_now:
                    # Validation failed: current frame becomes new full-model anchor.
                    print(f"[InferenceController] Validation failed at step {current_step} (deviation={deviation:.4f} > {self.deviation_threshold:.4f}), re-anchoring full chunk.")
                    full_chunk = self._anchor_full_chunk(obs, task_description)
                    selected_action = np.asarray(full_chunk[0], dtype=np.float32)
                    self._step_in_full_chunk = 1
                    return {
                        "action": selected_action,
                        "used_mode": "full_reanchor",
                        "deviation": deviation,
                        "full_chunk_step": current_step,
                    }

                # Validation passed: schedule next light validation after `light_num_actions_chunk` steps.
                self._next_validation_step += self.light_num_actions_chunk
        else:
            self._last_deviation = None

        self._step_in_full_chunk += 1

        return {
            "action": selected_action,
            "used_mode": "lightweight",
            "deviation": self._last_deviation,
            "full_chunk_step": current_step,
        }
