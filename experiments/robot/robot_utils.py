"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)
from prismatic.extern.hf.lightweight_temporal_model import build_standalone_light_model
# The vla-scripts directory provides more flexible loading routines (e.g. temporal fusion).
# We import lazily inside `get_model` when needed to avoid adding extra path overhead during normal usage.


# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(cfg: Any, wrap_diffusion_policy_for_droid: bool = False) -> torch.nn.Module:
    """
    Load and initialize model for evaluation based on configuration.

    Args:
        cfg: Configuration object with model parameters
        wrap_diffusion_policy_for_droid: Whether to wrap diffusion policy for DROID

    Returns:
        torch.nn.Module: The loaded model

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family == "openvla":
        # Load full model from configured native/big-model checkpoint path.
        prev_temporal_flag = getattr(cfg, "use_temporal_fusion", False)
        try:
            setattr(cfg, "use_temporal_fusion", False)
            model = get_vla(cfg)
        finally:
            setattr(cfg, "use_temporal_fusion", prev_temporal_flag)
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    print(f"Loaded model: {type(model)}")
    return model


def get_light_model(cfg: Any, base_model: Optional[torch.nn.Module] = None) -> Optional[torch.nn.Module]:
    """Load standalone lightweight temporal model.

    Returns `None` when temporal fusion is disabled.
    """
    if cfg.model_family != "openvla":
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    if not bool(getattr(cfg, "use_temporal_fusion", False)):
        return None

    light_resume_checkpoint = getattr(cfg, "light_resume_checkpoint", None) or getattr(cfg, "resume_checkpoint", None)

    light_pretrained_checkpoint = getattr(cfg, "light_pretrained_checkpoint", None) or cfg.pretrained_checkpoint

    if base_model is None:
        raise ValueError("`get_light_model()` requires an already-loaded full model via `base_model`.")
    base_model_for_components = base_model
    light_model = build_standalone_light_model(
        base_model=base_model_for_components,
        checkpoint_path=light_resume_checkpoint,
        vit_t_timm_model_id=getattr(cfg, "vit_t_timm_model_id", "vit_tiny_patch16_224.augreg_in21k_ft_in1k"),
        vit_t_image_size=getattr(cfg, "vit_t_image_size", 224),
        light_num_actions_chunk=int(getattr(cfg, "light_num_actions_chunk", -1)),
        vit_t_pretrained=getattr(cfg, "vit_t_pretrained", True),
        action_head_checkpoint=(
            getattr(cfg, "light_action_head_checkpoint", None)
            or getattr(cfg, "light_pretrained_checkpoint", None)
            or cfg.pretrained_checkpoint
        ),
        norm_stats_path_or_repo=light_pretrained_checkpoint,
        use_l1_regression=bool(getattr(cfg, "use_l1_regression", True)),
        use_diffusion=bool(getattr(cfg, "use_diffusion", False)),
        num_diffusion_steps_train=int(getattr(cfg, "num_diffusion_steps_train", 50)),
        num_diffusion_steps_inference=int(getattr(cfg, "num_diffusion_steps_inference", 50)),
        map_location="cpu",
    )

    source_desc = light_resume_checkpoint if light_resume_checkpoint else "base-model temporal modules"
    print(f"Loaded standalone light model from {source_desc}")
    return light_model


def get_image_resize_size(cfg: Any) -> Union[int, tuple]:
    """
    Get image resize dimensions for a specific model.

    If returned value is an int, the resized image will be a square.
    If returned value is a tuple, the resized image will be a rectangle.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        Union[int, tuple]: Image resize dimensions

    Raises:
        ValueError: If model family is not supported
    """
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return MODEL_IMAGE_SIZES[cfg.model_family]


def get_action(
    cfg: Any,
    model: torch.nn.Module,
    obs: Dict[str, Any],
    task_label: str,
    processor: Optional[Any] = None,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Query the model to get action predictions.

    Args:
        cfg: Configuration object with model parameters
        model: The loaded model
        obs: Observation dictionary
        task_label: Text description of the task
        processor: Model processor for inputs
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        Union[List[np.ndarray], np.ndarray]: Predicted actions

    Raises:
        ValueError: If model family is not supported
    """
    with torch.no_grad():
        if cfg.model_family == "openvla":
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                processor=processor,
                obs=obs,
                task_label=task_label,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=use_film,
            )
        else:
            raise ValueError(f"Unsupported model family: {cfg.model_family}")

    return action


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action
