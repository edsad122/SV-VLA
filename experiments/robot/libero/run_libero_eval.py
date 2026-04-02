"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.eval_utils import (
    _parse_int_csv,
    _save_context_frames_for_episode,
    load_initial_states,
    load_resume_state_from_log,
    log_message,
    write_resume_episode_marker,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_light_model,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.extern.hf.inference_controller import InferenceController
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/path/to/openvla-oft"     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    # temporal fusion / visual-tiny branch options (optional; most models ignore)
    use_temporal_fusion: bool = True                # Enable temporal fusion branch (frozen ViT-T + FC blend)
    vit_t_timm_model_id: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"  # TIMM model id for ViT-T
    vit_t_image_size: int = 224                      # Input image size for ViT-T
    vit_t_pretrained: bool = True                    # Whether ViT-T should load pretrained weights
    light_pretrained_checkpoint: Optional[Union[str, Path]] = None  # Path for light-model stats / action-head source
    light_resume_checkpoint: Optional[str] = None    # Path for light-model last-layer + temporal-fc checkpoint
    light_action_head_checkpoint: Optional[Union[str, Path]] = None  # Optional explicit action-head path for light model
    light_num_actions_chunk: int = 8                # <=0 means auto-use full model chunk length
    controller_deviation_threshold: float = 0.5     # Re-anchor if deviation > threshold
    controller_reanchor_step_multiple: int = 0      # <=0: no control; >0: only allow re-anchor at n*light_chunk step
    resume_checkpoint: str = "/path/to/last_layer_temporal_fc.pt"

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 64                     # Number of actions to execute open-loop before requerying policy
    replan_every_frame: bool = False                 # If True, query policy every env step (open-loop steps forced to 1)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    dataset_statistics_path: Optional[str] = None     # Optional override path for dataset_statistics.json

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL  # Task suite
    start_task_id: int = 0                         # Inclusive start task id (for partial evaluation)
    end_task_id: int = -1                          # Inclusive end task id; -1 means last task
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    resume_from_log_path: Optional[str] = None        # If set, resume evaluation state from this existing log file

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Visualization / diagnostics
    save_context_frames: bool = False                # Save +/-window frames around selected timeline frames
    context_frame_window: int = 4                    # Save [f-window, f+window]
    context_frame_indices: str = ""                 # Comma-separated policy-frame indices, e.g. "10,32"
    context_frame_use_reanchor_events: bool = True   # Also save around re-anchor frames
    context_frame_dir: str = "./experiments/frame_dumps"
    save_reanchor_event_log: bool = True             # Write per-reanchor events to independent JSONL log
    reanchor_event_log_path: Optional[str] = None    # Optional override path for reanchor event JSONL

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    print(
        "[Config] use_temporal_fusion=",
        cfg.use_temporal_fusion,
        " light_num_actions_chunk=",
        cfg.light_num_actions_chunk,
        sep="",
    )

    if cfg.use_temporal_fusion:
        assert cfg.light_num_actions_chunk != 0, "`light_num_actions_chunk` cannot be 0. Use -1 for auto mode or a positive integer."
        assert (
            cfg.light_resume_checkpoint or cfg.resume_checkpoint
        ), "`light_resume_checkpoint` (or fallback `resume_checkpoint`) is required for standalone light model."
        print("[Config] temporal-fusion controller validation is enabled.")

    # No additional validation required for temporal-fusion or ViT-T parameters; they are ignored by most checkpoints.

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


@dataclass
class PolicyRuntime:
    cfg: GenerateConfig
    model: Any
    light_model: Any
    action_head: Any
    light_action_head: Any
    proprio_projector: Any
    noisy_action_projector: Any
    processor: Any


def _build_runtime(cfg: GenerateConfig) -> PolicyRuntime:
    """Build one policy runtime (model + auxiliary modules)."""
    # Load model
    model = get_model(cfg)
    light_model = get_light_model(cfg, base_model=model) if cfg.use_temporal_fusion else None

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    light_action_head = None
    if cfg.use_temporal_fusion and (cfg.use_l1_regression or cfg.use_diffusion):
        light_head_path = cfg.light_action_head_checkpoint or cfg.light_pretrained_checkpoint
        if light_head_path:
            cfg_light_head = replace(cfg, pretrained_checkpoint=light_head_path)
            light_action_head = get_action_head(cfg_light_head, model.llm_dim)
        else:
            light_action_head = action_head

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return PolicyRuntime(
        cfg,
        model,
        light_model,
        action_head,
        light_action_head,
        proprio_projector,
        noisy_action_projector,
        processor,
    )


def initialize_model(cfg: GenerateConfig):
    """Initialize one runtime."""
    runtime = _build_runtime(cfg)
    cfg.unnorm_key = runtime.cfg.unnorm_key
    return runtime


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    if cfg.resume_from_log_path:
        local_log_filepath = str(Path(cfg.resume_from_log_path))
        os.makedirs(os.path.dirname(local_log_filepath), exist_ok=True)
        log_file = open(local_log_filepath, "a")
        run_id = Path(local_log_filepath).stem
        logger.info(f"Resuming and appending to local log file: {local_log_filepath}")
    else:
        # Create run ID
        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"

        # Set up local logging
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
        log_file = open(local_log_filepath, "w")
        logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    reanchor_event_log_file = None
    reanchor_event_log_path = None
    if cfg.save_reanchor_event_log:
        if cfg.reanchor_event_log_path:
            reanchor_event_log_path = str(cfg.reanchor_event_log_path)
        else:
            reanchor_event_log_path = str(Path(local_log_filepath).with_suffix(".reanchor.jsonl"))
        os.makedirs(os.path.dirname(reanchor_event_log_path), exist_ok=True)
        reanchor_event_log_file = open(reanchor_event_log_path, "a" if cfg.resume_from_log_path else "w")
        logger.info(f"Re-anchor event log: {reanchor_event_log_path}")

    return log_file, local_log_filepath, run_id, reanchor_event_log_file, reanchor_event_log_path


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def _clone_observation_for_policy(observation: dict) -> dict:
    """Clone observation so multiple policy calls don't accidentally share mutable arrays."""
    return {
        "full_image": observation["full_image"].copy(),
        "wrist_image": observation["wrist_image"].copy(),
        "state": observation["state"].copy(),
        "is_first_frame": bool(observation.get("is_first_frame", False)),
    }


def _force_reset_model_temporal_cache(model_obj: Any) -> None:
    """Best-effort clear of temporal cache state on a model wrapper.

    This helps keep episode results stable regardless of episode ordering/sampling.
    """
    if model_obj is None:
        return

    # Standalone light-model API.
    if hasattr(model_obj, "reset_temporal_cache"):
        try:
            model_obj.reset_temporal_cache()
        except Exception:
            pass

    # OpenVLA cache fields.
    for k in (
        "_cached_first_frame_feat",
        "_cached_penultimate_hidden",
        "_cached_causal_mask",
        "_cached_position_ids",
        "_cached_cache_position",
        "_cached_num_patch_tokens",
    ):
        if hasattr(model_obj, k):
            try:
                setattr(model_obj, k, None)
            except Exception:
                pass


def run_episode(
    cfg: GenerateConfig,
    env,
    task_id: int,
    episode_idx: int,
    task_description: str,
    policy_runtime: PolicyRuntime,
    resize_size,
    reanchor_event_log_file=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Hard reset model-side temporal caches to avoid cross-episode contamination.
    _force_reset_model_temporal_cache(policy_runtime.model)
    _force_reset_model_temporal_cache(policy_runtime.light_model)

    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue for non-temporal-fusion path only.
    open_loop_steps = 1 if cfg.replan_every_frame else cfg.num_open_loop_steps
    if not cfg.use_temporal_fusion and open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: open-loop steps ({open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    temporal_inference_controller = None
    episode_infer_time_s = 0.0
    episode_infer_count = 0
    episode_reanchor_count = 0
    episode_full_infer_time_s = 0.0
    episode_full_infer_count = 0
    episode_light_infer_time_s = 0.0
    episode_light_infer_count = 0
    episode_reanchor_events = []

    def _timed_get_action(model_obj, obs_in, task_desc_in, *, action_head_obj, inference_mode: str):
        nonlocal episode_infer_time_s, episode_infer_count
        nonlocal episode_full_infer_time_s, episode_full_infer_count
        nonlocal episode_light_infer_time_s, episode_light_infer_count
        t0 = time.perf_counter()
        actions_out = get_action(
            policy_runtime.cfg,
            model_obj,
            obs_in,
            task_desc_in,
            processor=policy_runtime.processor,
            action_head=action_head_obj,
            proprio_projector=policy_runtime.proprio_projector,
            noisy_action_projector=policy_runtime.noisy_action_projector,
            use_film=policy_runtime.cfg.use_film,
        )
        elapsed = time.perf_counter() - t0
        # `light_prime` runs for cache initialization only and is intentionally excluded from reported stats.
        if inference_mode != "light_prime":
            episode_infer_time_s += elapsed
            episode_infer_count += 1
        if inference_mode == "full":
            episode_full_infer_time_s += elapsed
            episode_full_infer_count += 1
        elif inference_mode == "light":
            episode_light_infer_time_s += elapsed
            episode_light_infer_count += 1
        return actions_out

    if cfg.use_temporal_fusion:
        effective_light_chunk = cfg.light_num_actions_chunk if cfg.light_num_actions_chunk > 0 else NUM_ACTIONS_CHUNK
        # Temporal-fusion path: full model provides execution chunk, light model validates.
        controller_enable_validation = bool(cfg.use_temporal_fusion)
        effective_full_chunk = open_loop_steps
        print(
            "[Controller] source=full_model, validator=light_model, "
            f"validation={controller_enable_validation}, "
            f"full_chunk={effective_full_chunk}, light_chunk={effective_light_chunk}, "
            f"threshold={cfg.controller_deviation_threshold}, "
            f"reanchor_step_multiple={cfg.controller_reanchor_step_multiple}"
        )

        def _infer_chunk_temporal_full(obs_in, task_desc_in):
            return _timed_get_action(
                policy_runtime.model,
                obs_in,
                task_desc_in,
                action_head_obj=policy_runtime.action_head,
                inference_mode="full",
            )

        temporal_cache_keys = (
            "_cached_first_frame_feat",
            "_cached_penultimate_hidden",
            "_cached_causal_mask",
            "_cached_position_ids",
            "_cached_cache_position",
            "_cached_num_patch_tokens",
        )

        def _snapshot_model_temporal_cache(model_obj):
            snap = {}
            for k in temporal_cache_keys:
                if not hasattr(model_obj, k):
                    continue
                v = getattr(model_obj, k)
                if isinstance(v, np.ndarray):
                    snap[k] = v.copy()
                elif torch.is_tensor(v):
                    snap[k] = v.detach().clone()
                else:
                    snap[k] = v
            return snap

        def _restore_model_temporal_cache(model_obj, snap):
            for k, v in snap.items():
                setattr(model_obj, k, v)

        def _infer_chunk_temporal_light(obs_in, task_desc_in):
            if policy_runtime.light_model is None:
                raise RuntimeError("Light model is required when `use_temporal_fusion=True`.")
            # Guard against accidental mutation of full-model temporal cache by validation path.
            full_model_cache_snapshot = _snapshot_model_temporal_cache(policy_runtime.model)
            try:
                return _timed_get_action(
                    policy_runtime.light_model,
                    obs_in,
                    task_desc_in,
                    action_head_obj=policy_runtime.light_action_head,
                    inference_mode="light",
                )
            finally:
                _restore_model_temporal_cache(policy_runtime.model, full_model_cache_snapshot)

        def _prime_light_cache_from_full(obs_in, task_desc_in):
            if policy_runtime.light_model is None:
                return None
            # Prefer direct cache transfer to avoid an extra light forward pass and reduce side effects.
            if hasattr(policy_runtime.light_model, "import_external_first_frame_cache") and hasattr(
                policy_runtime.light_model, "extract_first_frame_cache_from_model"
            ):
                external_cache = policy_runtime.light_model.extract_first_frame_cache_from_model(policy_runtime.model)
                policy_runtime.light_model.import_external_first_frame_cache(external_cache)
                return None

            prime_obs = _clone_observation_for_policy(obs_in)
            prime_obs["is_first_frame"] = True
            # Fallback: run explicit prime, while protecting full-model cache/state.
            full_model_cache_snapshot = _snapshot_model_temporal_cache(policy_runtime.model)
            try:
                return _timed_get_action(
                    policy_runtime.light_model,
                    prime_obs,
                    task_desc_in,
                    action_head_obj=policy_runtime.light_action_head,
                    inference_mode="light_prime",
                )
            finally:
                _restore_model_temporal_cache(policy_runtime.model, full_model_cache_snapshot)

        temporal_inference_controller = InferenceController(
            run_full_chunk_fn=_infer_chunk_temporal_full,
            run_light_chunk_fn=_infer_chunk_temporal_light,
            use_temporal_fusion=True,
            full_num_actions_chunk=effective_full_chunk,
            light_num_actions_chunk=effective_light_chunk,
            enable_validation=controller_enable_validation,
            prime_light_cache_fn=_prime_light_cache_from_full,
            deviation_threshold=cfg.controller_deviation_threshold,
            reanchor_step_multiple=int(cfg.controller_reanchor_step_multiple),
        )
        temporal_inference_controller.reset()

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # Temporal-fusion path: controller decides per-frame full/lightweight inference.
            if temporal_inference_controller is not None:
                policy_obs = _clone_observation_for_policy(observation)
                # Keep first-frame cache reset policy aligned with baseline path.
                policy_obs["is_first_frame"] = (t == cfg.num_steps_wait)
                step_out = temporal_inference_controller.step(policy_obs, task_description)
                action = step_out["action"]
                if step_out.get("used_mode") == "full_reanchor":
                    episode_reanchor_count += 1
                    policy_frame_idx = len(replay_images) - 1
                    event = {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "task_description": task_description,
                        "env_step": int(t),
                        "policy_frame_idx": int(policy_frame_idx),
                        "controller_step": int(step_out.get("full_chunk_step", -1)),
                        "deviation": float(step_out.get("deviation", 0.0) or 0.0),
                        "threshold": float(cfg.controller_deviation_threshold),
                        "event": "full_reanchor",
                    }
                    episode_reanchor_events.append(event)
                    if reanchor_event_log_file is not None:
                        reanchor_event_log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                        reanchor_event_log_file.flush()
            else:
                # Baseline path: open-loop action chunks.
                if len(action_queue) == 0:
                    # Reset temporal first-frame cache at every chunk query (episode start, step 65, ...).
                    observation["is_first_frame"] = True

                    # Query base policy to get executable action chunk.
                    policy_obs = _clone_observation_for_policy(observation)
                    actions = _timed_get_action(
                        policy_runtime.model,
                        policy_obs,
                        task_description,
                        action_head_obj=policy_runtime.action_head,
                        inference_mode="baseline",
                    )
                    action_queue.extend(actions)

                action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if cfg.save_context_frames:
        focus_frames = set(_parse_int_csv(cfg.context_frame_indices))
        if cfg.context_frame_use_reanchor_events:
            focus_frames.update(int(ev["policy_frame_idx"]) for ev in episode_reanchor_events)
        _save_context_frames_for_episode(
            cfg,
            replay_images,
            task_id,
            episode_idx,
            task_description,
            focus_frames,
            log_file,
        )

    episode_metrics = {
        "inference_time_s": float(episode_infer_time_s),
        "inference_count": int(episode_infer_count),
        "reanchor_count": int(episode_reanchor_count),
        "full_inference_time_s": float(episode_full_infer_time_s),
        "full_inference_count": int(episode_full_infer_count),
        "light_inference_time_s": float(episode_light_infer_time_s),
        "light_inference_count": int(episode_light_infer_count),
    }
    return success, replay_images, episode_metrics


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    policy_runtime: PolicyRuntime,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    reanchor_event_log_file=None,
    resume_task_state: Optional[dict] = None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    processed_episode_indices = set()
    task_episodes = 0
    task_successes = 0
    task_infer_time_s = 0.0
    task_infer_count = 0
    task_reanchor_count = 0
    task_full_infer_time_s = 0.0
    task_full_infer_count = 0
    task_light_infer_time_s = 0.0
    task_light_infer_count = 0
    if resume_task_state is not None:
        processed_episode_indices = set(resume_task_state.get("processed_episode_indices", []))
        task_episodes = int(resume_task_state.get("task_episodes", 0))
        task_successes = int(resume_task_state.get("task_successes", 0))
        task_infer_time_s = float(resume_task_state.get("task_infer_time_s", 0.0))
        task_infer_count = int(resume_task_state.get("task_infer_count", 0))
        task_reanchor_count = int(resume_task_state.get("task_reanchor_count", 0))
        task_full_infer_time_s = float(resume_task_state.get("task_full_infer_time_s", 0.0))
        task_full_infer_count = int(resume_task_state.get("task_full_infer_count", 0))
        task_light_infer_time_s = float(resume_task_state.get("task_light_infer_time_s", 0.0))
        task_light_infer_count = int(resume_task_state.get("task_light_infer_count", 0))

    task_infer_time_s_delta = 0.0
    task_infer_count_delta = 0
    task_reanchor_count_delta = 0
    task_full_infer_time_s_delta = 0.0
    task_full_infer_count_delta = 0
    task_light_infer_time_s_delta = 0.0
    task_light_infer_count_delta = 0

    if len(processed_episode_indices) > 0:
        log_message(
            f"Resuming task_id={task_id}: skipping {len(processed_episode_indices)} already-processed episode indices.",
            log_file,
        )

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        if episode_idx in processed_episode_indices:
            continue

        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                write_resume_episode_marker(
                    task_id,
                    episode_idx,
                    ran_episode=False,
                    success=False,
                    infer_time_s=0.0,
                    infer_count=0,
                    reanchor_count=0,
                    full_infer_time_s=0.0,
                    full_infer_count=0,
                    light_infer_time_s=0.0,
                    light_infer_count=0,
                    log_file=log_file,
                )
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, episode_metrics = run_episode(
            cfg,
            env,
            task_id,
            episode_idx,
            task_description,
            policy_runtime,
            resize_size,
            reanchor_event_log_file=reanchor_event_log_file,
            initial_state=initial_state,
            log_file=log_file,
        )
        task_infer_time_s += float(episode_metrics["inference_time_s"])
        task_infer_count += int(episode_metrics["inference_count"])
        task_reanchor_count += int(episode_metrics["reanchor_count"])
        task_full_infer_time_s += float(episode_metrics["full_inference_time_s"])
        task_full_infer_count += int(episode_metrics["full_inference_count"])
        task_light_infer_time_s += float(episode_metrics["light_inference_time_s"])
        task_light_infer_count += int(episode_metrics["light_inference_count"])
        task_infer_time_s_delta += float(episode_metrics["inference_time_s"])
        task_infer_count_delta += int(episode_metrics["inference_count"])
        task_reanchor_count_delta += int(episode_metrics["reanchor_count"])
        task_full_infer_time_s_delta += float(episode_metrics["full_inference_time_s"])
        task_full_infer_count_delta += int(episode_metrics["full_inference_count"])
        task_light_infer_time_s_delta += float(episode_metrics["light_inference_time_s"])
        task_light_infer_count_delta += int(episode_metrics["light_inference_count"])

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        write_resume_episode_marker(
            task_id,
            episode_idx,
            ran_episode=True,
            success=success,
            infer_time_s=float(episode_metrics["inference_time_s"]),
            infer_count=int(episode_metrics["inference_count"]),
            reanchor_count=int(episode_metrics["reanchor_count"]),
            full_infer_time_s=float(episode_metrics["full_inference_time_s"]),
            full_infer_count=int(episode_metrics["full_inference_count"]),
            light_infer_time_s=float(episode_metrics["light_inference_time_s"]),
            light_infer_count=int(episode_metrics["light_inference_count"]),
            log_file=log_file,
        )

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        ep_infer_time_s = float(episode_metrics["inference_time_s"])
        ep_infer_count = int(episode_metrics["inference_count"])
        ep_reanchor_count = int(episode_metrics["reanchor_count"])
        ep_avg_infer_s = (ep_infer_time_s / ep_infer_count) if ep_infer_count > 0 else 0.0
        if cfg.use_temporal_fusion:
            ep_full_infer_time_s = float(episode_metrics["full_inference_time_s"])
            ep_full_infer_count = int(episode_metrics["full_inference_count"])
            ep_full_avg_infer_s = (ep_full_infer_time_s / ep_full_infer_count) if ep_full_infer_count > 0 else 0.0
            ep_light_infer_time_s = float(episode_metrics["light_inference_time_s"])
            ep_light_infer_count = int(episode_metrics["light_inference_count"])
            ep_light_avg_infer_s = (
                (ep_light_infer_time_s / ep_light_infer_count) if ep_light_infer_count > 0 else 0.0
            )
            log_message(
                f"Episode inference stats: time_s={ep_infer_time_s:.6f}, calls={ep_infer_count}, "
                f"avg_time_s={ep_avg_infer_s:.6f}, reanchor={ep_reanchor_count}",
                log_file,
            )
            log_message(
                f"Episode inference split stats: full_time_s={ep_full_infer_time_s:.6f}, full_calls={ep_full_infer_count}, "
                f"full_avg_time_s={ep_full_avg_infer_s:.6f}, light_time_s={ep_light_infer_time_s:.6f}, "
                f"light_calls={ep_light_infer_count}, light_avg_time_s={ep_light_avg_infer_s:.6f}",
                log_file,
            )
        else:
            log_message(
                f"Episode inference stats: time_s={ep_infer_time_s:.6f}, calls={ep_infer_count}, "
                f"avg_time_s={ep_avg_infer_s:.6f}",
                log_file,
            )
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    task_avg_infer_s = (task_infer_time_s / task_infer_count) if task_infer_count > 0 else 0.0
    task_avg_reanchor = (task_reanchor_count / task_episodes) if task_episodes > 0 else 0.0
    if cfg.use_temporal_fusion:
        task_full_avg_infer_s = (task_full_infer_time_s / task_full_infer_count) if task_full_infer_count > 0 else 0.0
        task_light_avg_infer_s = (
            (task_light_infer_time_s / task_light_infer_count) if task_light_infer_count > 0 else 0.0
        )
        log_message(
            f"Task inference stats: total_time_s={task_infer_time_s:.6f}, total_calls={task_infer_count}, "
            f"avg_time_s={task_avg_infer_s:.6f}, reanchor_total={task_reanchor_count}, "
            f"reanchor_avg_per_episode={task_avg_reanchor:.6f}",
            log_file,
        )
        log_message(
            f"Task inference split stats: full_total_time_s={task_full_infer_time_s:.6f}, "
            f"full_total_calls={task_full_infer_count}, full_avg_time_s={task_full_avg_infer_s:.6f}, "
            f"light_total_time_s={task_light_infer_time_s:.6f}, light_total_calls={task_light_infer_count}, "
            f"light_avg_time_s={task_light_avg_infer_s:.6f}",
            log_file,
        )
    else:
        log_message(
            f"Task inference stats: total_time_s={task_infer_time_s:.6f}, total_calls={task_infer_count}, "
            f"avg_time_s={task_avg_infer_s:.6f}",
            log_file,
        )

    # Log to wandb if enabled
    if cfg.use_wandb:
        task_wandb_payload = {
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
            f"inference/total_time_s/{task_description}": task_infer_time_s,
            f"inference/total_calls/{task_description}": task_infer_count,
            f"inference/avg_time_s/{task_description}": task_avg_infer_s,
        }
        if cfg.use_temporal_fusion:
            task_wandb_payload.update(
                {
                    f"reanchor/total/{task_description}": task_reanchor_count,
                    f"reanchor/avg_per_episode/{task_description}": task_avg_reanchor,
                    f"inference/full_total_time_s/{task_description}": task_full_infer_time_s,
                    f"inference/full_total_calls/{task_description}": task_full_infer_count,
                    f"inference/full_avg_time_s/{task_description}": task_full_avg_infer_s,
                    f"inference/light_total_time_s/{task_description}": task_light_infer_time_s,
                    f"inference/light_total_calls/{task_description}": task_light_infer_count,
                    f"inference/light_avg_time_s/{task_description}": task_light_avg_infer_s,
                }
            )
        wandb.log(task_wandb_payload)

    task_metrics = {
        "task_infer_time_s_delta": float(task_infer_time_s_delta),
        "task_infer_count_delta": int(task_infer_count_delta),
        "task_reanchor_count_delta": int(task_reanchor_count_delta),
        "task_full_infer_time_s_delta": float(task_full_infer_time_s_delta),
        "task_full_infer_count_delta": int(task_full_infer_count_delta),
        "task_light_infer_time_s_delta": float(task_light_infer_time_s_delta),
        "task_light_infer_count_delta": int(task_light_infer_count_delta),
        "task_episodes_total": int(task_episodes),
    }

    return total_episodes, total_successes, task_metrics


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    policy_runtime = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id, reanchor_event_log_file, _ = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    start_task_id = int(max(0, cfg.start_task_id))
    end_task_id = (num_tasks - 1) if int(cfg.end_task_id) < 0 else int(cfg.end_task_id)
    end_task_id = int(min(end_task_id, num_tasks - 1))
    assert start_task_id <= end_task_id, (
        f"Invalid task-id range: start_task_id={start_task_id}, end_task_id={end_task_id}, num_tasks={num_tasks}"
    )

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Task range: [{start_task_id}, {end_task_id}] / total={num_tasks}", log_file)

    # Optionally resume state from existing log markers.
    resume_state = {
        "task_states": {},
        "total_episodes": 0,
        "total_successes": 0,
        "total_infer_time_s": 0.0,
        "total_infer_count": 0,
        "total_reanchor_count": 0,
        "total_full_infer_time_s": 0.0,
        "total_full_infer_count": 0,
        "total_light_infer_time_s": 0.0,
        "total_light_infer_count": 0,
    }
    if cfg.resume_from_log_path:
        resume_state = load_resume_state_from_log(cfg.resume_from_log_path, num_tasks, task_suite, log_file)

    # Start evaluation
    total_episodes = int(resume_state["total_episodes"])
    total_successes = int(resume_state["total_successes"])
    total_infer_time_s = float(resume_state["total_infer_time_s"])
    total_infer_count = int(resume_state["total_infer_count"])
    total_reanchor_count = int(resume_state["total_reanchor_count"])
    total_full_infer_time_s = float(resume_state.get("total_full_infer_time_s", 0.0))
    total_full_infer_count = int(resume_state.get("total_full_infer_count", 0))
    total_light_infer_time_s = float(resume_state.get("total_light_infer_time_s", 0.0))
    total_light_infer_count = int(resume_state.get("total_light_infer_count", 0))
    total_tasks_with_episodes = sum(
        1 for task_state in resume_state["task_states"].values() if int(task_state.get("task_episodes", 0)) > 0
    )
    for task_id in tqdm.tqdm(range(start_task_id, end_task_id + 1)):
        task_resume_state = resume_state["task_states"].get(task_id)
        had_episodes_before = bool(task_resume_state and int(task_resume_state.get("task_episodes", 0)) > 0)

        total_episodes, total_successes, task_metrics = run_task(
            cfg,
            task_suite,
            task_id,
            policy_runtime,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
            reanchor_event_log_file=reanchor_event_log_file,
            resume_task_state=task_resume_state,
        )
        total_infer_time_s += float(task_metrics["task_infer_time_s_delta"])
        total_infer_count += int(task_metrics["task_infer_count_delta"])
        total_reanchor_count += int(task_metrics["task_reanchor_count_delta"])
        total_full_infer_time_s += float(task_metrics["task_full_infer_time_s_delta"])
        total_full_infer_count += int(task_metrics["task_full_infer_count_delta"])
        total_light_infer_time_s += float(task_metrics["task_light_infer_time_s_delta"])
        total_light_infer_count += int(task_metrics["task_light_infer_count_delta"])
        if (not had_episodes_before) and int(task_metrics["task_episodes_total"]) > 0:
            total_tasks_with_episodes += 1

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    overall_avg_infer_s = (total_infer_time_s / total_infer_count) if total_infer_count > 0 else 0.0
    overall_avg_reanchor_per_task = (
        total_reanchor_count / total_tasks_with_episodes if total_tasks_with_episodes > 0 else 0.0
    )
    if cfg.use_temporal_fusion:
        overall_full_avg_infer_s = (
            (total_full_infer_time_s / total_full_infer_count) if total_full_infer_count > 0 else 0.0
        )
        overall_light_avg_infer_s = (
            (total_light_infer_time_s / total_light_infer_count) if total_light_infer_count > 0 else 0.0
        )
        log_message(
            f"Overall inference stats: total_time_s={total_infer_time_s:.6f}, total_calls={total_infer_count}, "
            f"avg_time_s={overall_avg_infer_s:.6f}, reanchor_total={total_reanchor_count}, "
            f"reanchor_avg_per_task={overall_avg_reanchor_per_task:.6f}",
            log_file,
        )
        log_message(
            f"Overall inference split stats: full_total_time_s={total_full_infer_time_s:.6f}, "
            f"full_total_calls={total_full_infer_count}, full_avg_time_s={overall_full_avg_infer_s:.6f}, "
            f"light_total_time_s={total_light_infer_time_s:.6f}, light_total_calls={total_light_infer_count}, "
            f"light_avg_time_s={overall_light_avg_infer_s:.6f}",
            log_file,
        )
    else:
        log_message(
            f"Overall inference stats: total_time_s={total_infer_time_s:.6f}, total_calls={total_infer_count}, "
            f"avg_time_s={overall_avg_infer_s:.6f}",
            log_file,
        )

    # Log to wandb if enabled
    if cfg.use_wandb:
        total_wandb_payload = {
            "success_rate/total": final_success_rate,
            "num_episodes/total": total_episodes,
            "inference/total_time_s/total": total_infer_time_s,
            "inference/total_calls/total": total_infer_count,
            "inference/avg_time_s/total": overall_avg_infer_s,
        }
        if cfg.use_temporal_fusion:
            total_wandb_payload.update(
                {
                    "reanchor/total/total": total_reanchor_count,
                    "reanchor/avg_per_task/total": overall_avg_reanchor_per_task,
                    "inference/full_total_time_s/total": total_full_infer_time_s,
                    "inference/full_total_calls/total": total_full_infer_count,
                    "inference/full_avg_time_s/total": overall_full_avg_infer_s,
                    "inference/light_total_time_s/total": total_light_infer_time_s,
                    "inference/light_total_calls/total": total_light_infer_count,
                    "inference/light_avg_time_s/total": overall_light_avg_infer_s,
                }
            )
        wandb.log(total_wandb_payload)
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()
    if reanchor_event_log_file:
        reanchor_event_log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
