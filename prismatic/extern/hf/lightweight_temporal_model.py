from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, PrismaticVisionBackbone
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


class StandaloneLightTemporalModel:
    """Standalone lightweight temporal model wrapper.

    Loads only:
    - LLaMA last decoder layer (from checkpoint)
    - temporal fusion FC (from checkpoint)
    - ViT-T backbone (from pretrained config)

    Uses base model only for shared token/projector utilities and LLM pre-last-layer
    APIs, while keeping all temporal-specific logic and heads inside this module.
    """

    _CACHE_KEYS = (
        "_cached_first_frame_feat",
        "_cached_penultimate_hidden",
        "_cached_causal_mask",
        "_cached_position_ids",
        "_cached_cache_position",
        "_cached_num_patch_tokens",
    )

    def __init__(
        self,
        *,
        base_model: OpenVLAForActionPrediction,
        checkpoint_path: Optional[str],
        light_num_actions_chunk: int,
        vit_t_timm_model_id: str,
        vit_t_image_size: int,
        vit_t_pretrained: bool,
        action_head: nn.Module,
        norm_stats: Dict[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.base_model = base_model
        self.llm_dim = int(base_model.llm_dim)
        self.norm_stats = norm_stats
        self.action_head = action_head
        self.model_num_actions_chunk = int(NUM_ACTIONS_CHUNK)
        self.light_num_actions_chunk = int(light_num_actions_chunk)
        if self.light_num_actions_chunk <= 0:
            self.light_num_actions_chunk = self.model_num_actions_chunk
        if self.light_num_actions_chunk > self.model_num_actions_chunk:
            raise ValueError(
                f"`light_num_actions_chunk` ({self.light_num_actions_chunk}) cannot exceed model token chunk "
                f"capacity ({self.model_num_actions_chunk})."
            )

        ref_param = next(base_model.parameters())
        self.device = ref_param.device
        self.dtype = ref_param.dtype

        if not checkpoint_path:
            raise ValueError("`checkpoint_path` is required for standalone light model.")
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Lightweight checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        if "temporal_fusion_fc" not in ckpt or "llama_last_layer" not in ckpt:
            raise KeyError("Checkpoint missing `temporal_fusion_fc` or `llama_last_layer`.")

        # Load lightweight last LLaMA layer.
        self.light_last_layer: nn.Module = deepcopy(base_model.language_model.model.layers[-1])
        self.light_last_layer.load_state_dict(ckpt["llama_last_layer"], strict=True)
        if self.light_last_layer is base_model.language_model.model.layers[-1]:
            raise RuntimeError("Standalone light last layer must not alias base model last layer.")
        self.light_last_layer = self.light_last_layer.to(device=self.device, dtype=self.dtype)
        self.light_last_layer.eval()
        self.light_last_layer.requires_grad_(False)

        self.vit_t_backbone = PrismaticVisionBackbone(
            False,
            [int(vit_t_image_size)],
            [str(vit_t_timm_model_id)],
            [None],
            pretrained=bool(vit_t_pretrained),
        )
        self.vit_t_backbone = self.vit_t_backbone.to(device=self.device, dtype=self.dtype)
        self.vit_t_backbone.eval()
        self.vit_t_backbone.requires_grad_(False)

        self.temporal_fusion_fc = nn.Linear(self.llm_dim + self.vit_t_backbone.embed_dim, self.llm_dim)
        self.temporal_fusion_fc.load_state_dict(ckpt["temporal_fusion_fc"], strict=True)
        self.temporal_fusion_fc = self.temporal_fusion_fc.to(device=self.device, dtype=self.dtype)
        self.temporal_fusion_fc.eval()
        self.temporal_fusion_fc.requires_grad_(False)

        self._cache: Dict[str, Optional[torch.Tensor]] = {k: None for k in self._CACHE_KEYS}

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

    def reset_temporal_cache(self) -> None:
        for k in self._CACHE_KEYS:
            self._cache[k] = None

    def import_external_first_frame_cache(self, external_cache: Dict[str, Optional[torch.Tensor]]) -> None:
        """Import first-frame cache produced by external full model inference."""
        for k in self._CACHE_KEYS:
            v = external_cache.get(k, None)
            if isinstance(v, torch.Tensor):
                self._cache[k] = v.detach().to(device=self.device, dtype=v.dtype)
            else:
                self._cache[k] = v

    @classmethod
    def extract_first_frame_cache_from_model(cls, model: OpenVLAForActionPrediction) -> Dict[str, Optional[torch.Tensor]]:
        """Extract lightweight-required first-frame cache tensors from a full model."""
        out: Dict[str, Optional[torch.Tensor]] = {}
        for k in cls._CACHE_KEYS:
            v = getattr(model, k, None)
            out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        return out

    def _push_cache_to_base(self) -> None:
        for k in self._CACHE_KEYS:
            setattr(self.base_model, k, self._cache[k])

    def _pull_cache_from_base(self) -> None:
        for k in self._CACHE_KEYS:
            self._cache[k] = getattr(self.base_model, k, None)

    @staticmethod
    def _resize_token_sequence(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] == target_len:
            return tokens
        x = tokens.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def _extract_vit_t_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_input = pixel_values[:, :3, :, :] if pixel_values.shape[1] >= 3 else pixel_values

        target_size = self.vit_t_backbone.featurizer.patch_embed.img_size
        if isinstance(target_size, tuple):
            h, w = target_size
        else:
            h, w = target_size, target_size

        if vit_input.shape[-2:] != (h, w):
            vit_input = F.interpolate(vit_input, size=(h, w), mode="bilinear", align_corners=False)

        vit_param = next(self.vit_t_backbone.parameters())
        vit_input = vit_input.to(device=vit_param.device, dtype=vit_param.dtype)
        with torch.no_grad():
            vit_tokens = self.vit_t_backbone(vit_input)
        return vit_tokens

    def _unnormalize_actions(self, normalized_actions: np.ndarray, unnorm_key: Optional[str]) -> np.ndarray:
        if unnorm_key is None:
            if len(self.norm_stats) != 1:
                raise ValueError("Please pass `unnorm_key` for multi-dataset norm_stats.")
            unnorm_key = next(iter(self.norm_stats.keys()))
        if unnorm_key not in self.norm_stats:
            raise KeyError(f"Unknown `unnorm_key`: {unnorm_key}")

        action_norm_stats = self.norm_stats[unnorm_key]["action"]
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

    def predict_action(self, *args: Any, reset_first_frame_cache: bool = False, **kwargs: Any):
        """Predict action chunk using standalone light temporal modules.

        Uses internal first-frame cache; if empty, it bootstraps cache from current frame.
        """
        if reset_first_frame_cache:
            self.reset_temporal_cache()

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        unnorm_key = kwargs.get("unnorm_key", None)
        proprio = kwargs.get("proprio", None)
        proprio_projector = kwargs.get("proprio_projector", None)

        if input_ids is None or attention_mask is None or pixel_values is None:
            raise ValueError("`input_ids`, `attention_mask`, and `pixel_values` are required.")

        base = self.base_model

        # Keep prompt formatting behavior aligned with full model.
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871], device=input_ids.device).long(), dim=0)),
                dim=1,
            )

        labels = input_ids.clone()
        labels[:] = -100
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        cache_miss = self._cache["_cached_penultimate_hidden"] is None or self._cache["_cached_first_frame_feat"] is None
        use_proprio = proprio_projector is not None and proprio is not None

        if cache_miss:
            # Bootstrap cache once using full multimodal embedding path.
            input_ids, attention_mask = base._prepare_input_for_action_prediction(input_ids, attention_mask)
            labels = base._prepare_labels_for_action_prediction(labels, input_ids)

            input_embeddings = base.get_input_embeddings()(input_ids)
            all_actions_mask = base._process_action_masks(labels)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )

            projected_patch_embeddings = base._process_vision_features(
                pixel_values,
                language_embeddings,
                use_film=bool(kwargs.get("use_film", False)),
            )

            if use_proprio:
                proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
                projected_patch_embeddings = base._process_proprio_features(
                    projected_patch_embeddings,
                    proprio,
                    proprio_projector,
                )

            all_actions_mask = all_actions_mask.unsqueeze(-1)
            input_embeddings = input_embeddings * ~all_actions_mask
            multimodal_embeddings, multimodal_attention_mask = base._build_multimodal_attention(
                input_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )

            num_patch_tokens = projected_patch_embeddings.shape[1]

            penultimate_hidden_prime, causal_mask_prime, position_ids_prime, _, cache_position_prime = (
                base.language_model.model.forward_to_penultimate(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=multimodal_embeddings,
                    use_cache=False,
                    output_attentions=False,
                    cache_position=None,
                )
            )
            first_frame_tokens_prime = penultimate_hidden_prime[:, 1 : 1 + num_patch_tokens, :]
            self._cache["_cached_penultimate_hidden"] = penultimate_hidden_prime.detach()
            self._cache["_cached_causal_mask"] = (
                causal_mask_prime.detach() if isinstance(causal_mask_prime, torch.Tensor) else causal_mask_prime
            )
            self._cache["_cached_position_ids"] = (
                position_ids_prime.detach() if isinstance(position_ids_prime, torch.Tensor) else position_ids_prime
            )
            self._cache["_cached_cache_position"] = (
                cache_position_prime.detach() if isinstance(cache_position_prime, torch.Tensor) else cache_position_prime
            )
            self._cache["_cached_first_frame_feat"] = first_frame_tokens_prime.detach()
            self._cache["_cached_num_patch_tokens"] = num_patch_tokens

        cached_num_patch_tokens = self._cache.get("_cached_num_patch_tokens", None)
        if cached_num_patch_tokens is None:
            raise RuntimeError("Light model cache is missing `_cached_num_patch_tokens`.")
        num_patch_tokens = int(cached_num_patch_tokens)

        ref_hidden = self._cache["_cached_penultimate_hidden"]
        if not isinstance(ref_hidden, torch.Tensor):
            raise RuntimeError("Light model cache is missing `_cached_penultimate_hidden` tensor.")
        run_device = ref_hidden.device
        run_dtype = ref_hidden.dtype

        penultimate_hidden = self._cache["_cached_penultimate_hidden"].to(
            run_device,
            dtype=run_dtype,
        )
        first_frame_tokens = self._cache["_cached_first_frame_feat"].to(
            run_device,
            dtype=run_dtype,
        )

        causal_mask = self._cache["_cached_causal_mask"]
        if isinstance(causal_mask, torch.Tensor):
            causal_mask = causal_mask.to(run_device, dtype=run_dtype)
        position_ids = self._cache["_cached_position_ids"]
        if isinstance(position_ids, torch.Tensor):
            position_ids = position_ids.to(run_device)
        cache_position = self._cache["_cached_cache_position"]
        if isinstance(cache_position, torch.Tensor):
            cache_position = cache_position.to(run_device)

        if first_frame_tokens.shape[1] != num_patch_tokens:
            first_frame_tokens = self._resize_token_sequence(first_frame_tokens, num_patch_tokens)

        vit_t_tokens = self._extract_vit_t_feature(pixel_values).to(first_frame_tokens.dtype)
        vit_t_tokens = self._resize_token_sequence(vit_t_tokens, num_patch_tokens)
        fused_tokens = self.temporal_fusion_fc(torch.cat([first_frame_tokens, vit_t_tokens], dim=-1))

        hidden_for_last = penultimate_hidden.clone()
        hidden_for_last[:, 1 : 1 + num_patch_tokens, :] = fused_tokens

        # IMPORTANT: use standalone light last layer (loaded from temporal-fusion checkpoint),
        # not the base model's last layer, to keep light/full paths decoupled.
        layer_outputs = self.light_last_layer(
            hidden_for_last,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
        )
        last_hidden_states = layer_outputs[0]
        last_hidden_states = base.language_model.model.norm(last_hidden_states)

        NUM_PATCHES = num_patch_tokens

        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES
            + NUM_PROMPT_TOKENS : NUM_PATCHES
            + NUM_PROMPT_TOKENS
            + ACTION_DIM * self.light_num_actions_chunk,
            :,
        ]

        ah_dtype = self.action_head.model.fc1.weight.dtype
        normalized_actions = self.action_head.predict_action(actions_hidden_states.to(ah_dtype))
        if normalized_actions.ndim == 3:
            normalized_actions = normalized_actions[0]
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
        if normalized_actions.shape[0] != self.light_num_actions_chunk:
            normalized_actions = self._slice_chunk_with_tail_pad(normalized_actions, 0, self.light_num_actions_chunk)
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        return actions, actions_hidden_states


def build_standalone_light_model(
    *,
    base_model: OpenVLAForActionPrediction,
    checkpoint_path: Optional[str],
    vit_t_timm_model_id: str,
    vit_t_image_size: int,
    light_num_actions_chunk: int = -1,
    vit_t_pretrained: bool = True,
    action_head_checkpoint: Optional[str] = None,
    norm_stats_path_or_repo: Optional[str] = None,
    use_l1_regression: bool = True,
    use_diffusion: bool = False,
    num_diffusion_steps_train: int = 50,
    num_diffusion_steps_inference: int = 50,
    map_location: str | torch.device = "cpu",
) -> StandaloneLightTemporalModel:
    from experiments.robot.openvla_utils import get_action_head, model_is_on_hf_hub

    ah_checkpoint = action_head_checkpoint or norm_stats_path_or_repo
    if not ah_checkpoint:
        raise ValueError("`action_head_checkpoint` (or `norm_stats_path_or_repo`) is required for standalone light model.")

    ah_cfg = SimpleNamespace(
        pretrained_checkpoint=ah_checkpoint,
        use_l1_regression=use_l1_regression,
        use_diffusion=use_diffusion,
        num_diffusion_steps_train=num_diffusion_steps_train,
        num_diffusion_steps_inference=num_diffusion_steps_inference,
    )
    action_head = get_action_head(ah_cfg, base_model.llm_dim)

    if not norm_stats_path_or_repo:
        norm_stats_path_or_repo = ah_checkpoint

    if model_is_on_hf_hub(str(norm_stats_path_or_repo)):
        stats_path = hf_hub_download(repo_id=str(norm_stats_path_or_repo), filename="dataset_statistics.json")
    else:
        stats_path = os.path.join(str(norm_stats_path_or_repo), "dataset_statistics.json")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"dataset_statistics.json not found: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)

    return StandaloneLightTemporalModel(
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        light_num_actions_chunk=light_num_actions_chunk,
        vit_t_timm_model_id=vit_t_timm_model_id,
        vit_t_image_size=vit_t_image_size,
        vit_t_pretrained=vit_t_pretrained,
        action_head=action_head,
        norm_stats=norm_stats,
        map_location=map_location,
    )
