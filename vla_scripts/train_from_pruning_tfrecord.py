"""
Train OpenVLA temporal-fusion parameters using the same TFRecord data processing style as pruning.py.

What is trained:
- `vla.language_model.model.layers[-1]` (LLaMA final decoder layer)
- `vla.temporal_fusion_fc`

What is frozen:
- all other OpenVLA params
- external `action_head` (used only as supervision/readout from hidden states)
"""

from __future__ import annotations

import argparse
import glob
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure local repository modules are resolved first (avoid cross-repo `experiments.*` collisions).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_proprio_projector,
    get_processor,
    normalize_proprio,
    prepare_images_for_vla,
)
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK as FULL_NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone
from temporal_fusion_utils import (
    find_latest_temporal_fusion_checkpoint,
    load_openvla_with_temporal_fusion,
    load_temporal_fusion_checkpoint,
    save_temporal_fusion_checkpoint,
)


@dataclass
class TrainCfg:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_temporal_fusion: bool = True
    vit_t_timm_model_id: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    vit_t_image_size: int = 224
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = "libero_object_no_noops"
    lora_rank: int = 32

    # Data
    tfrecord_glob: str = ""
    dataset_statistics_path: Optional[str] = None

    # Training
    full_num_actions_chunk: int = 64
    light_num_actions_chunk: int = 8
    batch_size: int = 8
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.0
    log_every: int = 50
    save_dir: str = "runs/temporal_fusion_lastlayer"
    resume_checkpoint: Optional[str] = None
    auto_resume_latest: bool = True
    start_from_any_t: bool = True
    empty_cuda_cache_every_step: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


episode_desc = {
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),
    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/image": tf.io.VarLenFeature(tf.string),
    "steps/observation/wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),
}


def _parse_episode(serialized):
    ex = tf.io.parse_single_example(serialized, episode_desc)
    for k, v in list(ex.items()):
        if isinstance(v, tf.SparseTensor):
            ex[k] = tf.sparse.to_dense(v)

    T = tf.shape(ex["steps/is_first"])[0]
    ex["steps/action"] = tf.reshape(ex["steps/action"], [T, -1])
    ex["steps/observation/joint_state"] = tf.reshape(ex["steps/observation/joint_state"], [T, -1])
    ex["steps/observation/state"] = tf.reshape(ex["steps/observation/state"], [T, -1])
    return ex


class LiberoChunkDataset(Dataset):
    def __init__(
        self,
        tfrecord_glob: str,
        *,
        full_num_actions_chunk: int,
        light_num_actions_chunk: int,
        start_from_any_t: bool = True,
    ):
        files = sorted(glob.glob(tfrecord_glob))
        if len(files) == 0:
            raise FileNotFoundError(f"No TFRecord files matched: {tfrecord_glob}")

        print(f"[Dataset] matched_tfrecord_files={len(files)}; loading episodes into memory...")

        raw_ds = tf.data.TFRecordDataset(files).map(_parse_episode, num_parallel_calls=tf.data.AUTOTUNE)
        self.episodes = list(raw_ds.as_numpy_iterator())
        self.full_num_actions_chunk = int(full_num_actions_chunk)
        self.light_num_actions_chunk = int(light_num_actions_chunk)
        self.start_from_any_t = start_from_any_t

        self.index = []
        if self.start_from_any_t:
            # Any frame can be the start frame t.
            for ei, ep in enumerate(self.episodes):
                T = len(ep["steps/is_first"])
                for t in range(T):
                    self.index.append((ei, t))
        else:
            # Fallback: only use true episode starts.
            for ei, ep in enumerate(self.episodes):
                starts = np.where(ep["steps/is_first"].astype(bool))[0]
                for t in starts.tolist():
                    self.index.append((ei, int(t)))

        print(f"[Dataset] episodes={len(self.episodes)}, total_steps={len(self.index)}")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _dec(x) -> str:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8", errors="ignore")
        return str(x)

    def _get_action_chunk(self, ep, t: int, chunk_len: int) -> np.ndarray:
        # Build [chunk_len, ACTION_DIM], repeat last step if overflow.
        actions = ep["steps/action"]  # (T, A)
        T = actions.shape[0]
        out = []
        for dt in range(chunk_len):
            idx = min(t + dt, T - 1)
            out.append(actions[idx][:ACTION_DIM].astype("float32"))
        return np.stack(out, axis=0)

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        ep = self.episodes[ei]
        T = len(ep["steps/is_first"])

        # Build [t, t+1, ..., t+full_num_actions_chunk-1], clamped at episode end.
        full_seq, wrist_seq, instruction_seq, state_seq = [], [], [], []
        for dt in range(self.full_num_actions_chunk):
            ti = min(t + dt, T - 1)
            full_seq.append(tf.image.decode_jpeg(ep["steps/observation/image"][ti], channels=3).numpy())
            wrist_seq.append(tf.image.decode_jpeg(ep["steps/observation/wrist_image"][ti], channels=3).numpy())
            instruction_seq.append(self._dec(ep["steps/language_instruction"][ti]))
            state_seq.append(ep["steps/observation/state"][ti].astype("float32"))

        # Keep first-step chunk for compatibility/debug.
        action_chunk = self._get_action_chunk(ep, t, self.light_num_actions_chunk)
        # For every dt in full window, build aligned future light chunk [t+dt ...].
        action_chunk_seq = [
            self._get_action_chunk(ep, t + dt, self.light_num_actions_chunk)
            for dt in range(self.full_num_actions_chunk)
        ]

        return {
            "full_image_seq": full_seq,
            "wrist_image_seq": wrist_seq,
            "instruction_seq": instruction_seq,
            "state_seq": state_seq,
            "action_chunk": action_chunk,
            "action_chunk_seq": action_chunk_seq,
            "is_first": True,
        }


def collate_fn(batch):
    """Time-major collate for better CPU throughput and lower Python overhead in training loop."""
    B = len(batch)
    T = len(batch[0]["full_image_seq"])

    full_image_seq_by_t = []
    wrist_image_seq_by_t = []
    instruction_seq_by_t = []
    state_seq_by_t = []

    for t in range(T):
        full_image_seq_by_t.append([batch[i]["full_image_seq"][t] for i in range(B)])
        wrist_image_seq_by_t.append([batch[i]["wrist_image_seq"][t] for i in range(B)])
        instruction_seq_by_t.append([batch[i]["instruction_seq"][t] for i in range(B)])
        state_seq_by_t.append(np.stack([batch[i]["state_seq"][t] for i in range(B)], axis=0))

    action_chunk_seq = torch.tensor(
        np.stack([batch[i]["action_chunk_seq"] for i in range(B)], axis=0),
        dtype=torch.float32,
    )  # (B, full_chunk, light_chunk, A)

    return {
        "full_image_seq_by_t": full_image_seq_by_t,
        "wrist_image_seq_by_t": wrist_image_seq_by_t,
        "instruction_seq_by_t": instruction_seq_by_t,
        "state_seq_by_t": state_seq_by_t,
        "action_chunk_seq": action_chunk_seq,
    }


def _prepare_policy_inputs_batch(cfg: TrainCfg, vla, processor, obs_batch: dict, model_dtype: torch.dtype):
    """Prepare batched model inputs (prompt + image stack + optional proprio), shared by full/light paths."""
    prompts = [
        f"In: What action should the robot take to {instr.lower()}?\nOut:"
        for instr in obs_batch["instruction"]
    ]

    primary_images = prepare_images_for_vla(obs_batch["full_image"], cfg)
    inputs = processor(prompts, primary_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    if cfg.num_images_in_input > 1:
        wrist_images = prepare_images_for_vla(obs_batch["wrist_image"], cfg)
        wrist_inputs = processor(prompts, wrist_images, return_tensors="pt", padding=True)
        wrist_pixel_values = wrist_inputs["pixel_values"].to(DEVICE)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_pixel_values], dim=1)

    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype=model_dtype)

    proprio = None
    if cfg.use_proprio:
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        proprio_np = np.stack([normalize_proprio(s, proprio_norm_stats) for s in obs_batch["state"]], axis=0)
        proprio = torch.tensor(proprio_np, device=DEVICE, dtype=model_dtype)

    return inputs, proprio


def _resize_token_sequence(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[1] == target_len:
        return tokens
    x = tokens.transpose(1, 2)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.transpose(1, 2)


def _extract_vit_t_feature(vit_t_backbone: PrismaticVisionBackbone, pixel_values: torch.Tensor) -> torch.Tensor:
    vit_input = pixel_values[:, :3, :, :] if pixel_values.shape[1] >= 3 else pixel_values
    target_size = vit_t_backbone.featurizer.patch_embed.img_size
    if isinstance(target_size, tuple):
        h, w = target_size
    else:
        h, w = target_size, target_size

    if vit_input.shape[-2:] != (h, w):
        vit_input = F.interpolate(vit_input, size=(h, w), mode="bilinear", align_corners=False)

    ref = next(vit_t_backbone.parameters())
    vit_input = vit_input.to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        return vit_t_backbone(vit_input)


def _bootstrap_first_frame_cache(cfg: TrainCfg, vla, inputs: dict, proprio, proprio_projector=None) -> dict:
    """Run full model to penultimate layer once (t=0) and cache features needed by the light branch."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    if not torch.all(input_ids[:, -1] == 29871):
        eos = torch.full(
            (input_ids.shape[0], 1),
            29871,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, eos], dim=1)

        attn_pad = torch.ones(
            (attention_mask.shape[0], 1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, attn_pad], dim=1)

    num_prompt_tokens = input_ids.shape[-1] - 1
    labels = input_ids.clone()
    labels[:] = -100

    input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

    input_embeddings = vla.get_input_embeddings()(input_ids)
    all_actions_mask = vla._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )

    projected_patch_embeddings = vla._process_vision_features(pixel_values, language_embeddings, cfg.use_film)
    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        projected_patch_embeddings = vla._process_proprio_features(projected_patch_embeddings, proprio, proprio_projector)

    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask
    multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
        input_embeddings,
        projected_patch_embeddings,
        attention_mask,
    )

    num_patch_tokens = projected_patch_embeddings.shape[1]

    with torch.no_grad():
        penultimate_hidden, causal_mask, position_ids, _, cache_position = vla.language_model.model.forward_to_penultimate(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            use_cache=False,
            output_attentions=False,
            cache_position=None,
        )

    first_frame_tokens = penultimate_hidden[:, 1 : 1 + num_patch_tokens, :]
    return {
        "penultimate_hidden": penultimate_hidden.detach(),
        "first_frame_tokens": first_frame_tokens.detach(),
        "causal_mask": causal_mask.detach() if isinstance(causal_mask, torch.Tensor) else causal_mask,
        "position_ids": position_ids.detach() if isinstance(position_ids, torch.Tensor) else position_ids,
        "cache_position": cache_position.detach() if isinstance(cache_position, torch.Tensor) else cache_position,
        "num_patch_tokens": int(num_patch_tokens),
        "num_prompt_tokens": int(num_prompt_tokens),
    }


def build_inputs_and_extract_hs(
    cfg: TrainCfg,
    vla,
    obs: dict,
    *,
    cache: dict,
    vit_t_backbone: PrismaticVisionBackbone,
    temporal_fusion_fc: nn.Module,
):
    """Light branch forward for t>0 using cached first-frame/full-penultimate features."""
    pixel_values = obs["pixel_values"]

    hidden_device = cache["penultimate_hidden"].device
    hidden_dtype = cache["penultimate_hidden"].dtype
    penultimate_hidden = cache["penultimate_hidden"].to(device=hidden_device, dtype=hidden_dtype)
    first_frame_tokens = cache["first_frame_tokens"].to(device=hidden_device, dtype=hidden_dtype)

    num_patch_tokens = int(cache["num_patch_tokens"])
    if first_frame_tokens.shape[1] != num_patch_tokens:
        first_frame_tokens = _resize_token_sequence(first_frame_tokens, num_patch_tokens)

    vit_t_tokens = _extract_vit_t_feature(vit_t_backbone, pixel_values).to(device=hidden_device, dtype=hidden_dtype)
    vit_t_tokens = _resize_token_sequence(vit_t_tokens, num_patch_tokens)

    fused_tokens = temporal_fusion_fc(torch.cat([first_frame_tokens, vit_t_tokens], dim=-1))

    hidden_for_last = penultimate_hidden.clone()
    hidden_for_last[:, 1 : 1 + num_patch_tokens, :] = fused_tokens

    causal_mask = cache["causal_mask"]
    if isinstance(causal_mask, torch.Tensor):
        causal_mask = causal_mask.to(device=hidden_device, dtype=hidden_dtype)
    position_ids = cache["position_ids"]
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.to(device=hidden_device)
    cache_position = cache["cache_position"]
    if isinstance(cache_position, torch.Tensor):
        cache_position = cache_position.to(device=hidden_device)

    last_hidden_states = vla.language_model.model.forward_last_layer(
        hidden_states=hidden_for_last,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=cache_position,
    )

    num_prompt_tokens = int(cache["num_prompt_tokens"])
    actions_hidden_states = last_hidden_states[
        :,
        num_patch_tokens
        + num_prompt_tokens : num_patch_tokens
        + num_prompt_tokens
        + ACTION_DIM * cfg.light_num_actions_chunk,
        :,
    ]

    return actions_hidden_states  # (B, action_chunk_len * ACTION_DIM, D)


def train(cfg: TrainCfg):
    assert cfg.pretrained_checkpoint, "Please set pretrained_checkpoint"
    assert cfg.tfrecord_glob, "Please set tfrecord_glob"
    if int(cfg.full_num_actions_chunk) <= 0:
        raise ValueError("`full_num_actions_chunk` must be > 0.")
    if int(cfg.light_num_actions_chunk) <= 0:
        raise ValueError("`light_num_actions_chunk` must be > 0.")
    if int(cfg.light_num_actions_chunk) > int(cfg.full_num_actions_chunk):
        raise ValueError("`light_num_actions_chunk` cannot be larger than `full_num_actions_chunk`.")
    if int(cfg.light_num_actions_chunk) > int(FULL_NUM_ACTIONS_CHUNK):
        raise ValueError(
            f"`light_num_actions_chunk` ({cfg.light_num_actions_chunk}) cannot exceed model/action-head chunk "
            f"({FULL_NUM_ACTIONS_CHUNK})."
        )

    if not cfg.use_temporal_fusion:
        print("[Warn] `use_temporal_fusion` is deprecated in this trainer and ignored; temporal modules are always loaded.")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vla = load_openvla_with_temporal_fusion(
        pretrained_checkpoint=str(cfg.pretrained_checkpoint),
        device=DEVICE,
        num_images_in_input=cfg.num_images_in_input,
        vit_t_timm_model_id=cfg.vit_t_timm_model_id,
        vit_t_image_size=cfg.vit_t_image_size,
        vit_t_pretrained=True,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        dataset_statistics_path=cfg.dataset_statistics_path,
    )

    # Align all training-side modules/inputs to VLA parameter dtype to avoid fp16/bf16 mismatches.
    model_dtype = next((p.dtype for p in vla.parameters() if torch.is_floating_point(p)), torch.float32)

    # Autocast policy:
    # - If model dtype is bf16, only enable autocast when current CUDA supports bf16.
    # - If model dtype is fp16, enable fp16 autocast.
    # - Otherwise disable autocast.
    if DEVICE.type == "cuda":
        if model_dtype == torch.bfloat16:
            amp_autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
        elif model_dtype == torch.float16:
            amp_autocast_dtype = torch.float16
        else:
            amp_autocast_dtype = None
    else:
        amp_autocast_dtype = None

    print(
        f"[Precision] device={DEVICE}, model_dtype={model_dtype}, "
        f"autocast_dtype={amp_autocast_dtype}"
    )

    if cfg.use_proprio:
        if not hasattr(vla, "norm_stats") or not isinstance(vla.norm_stats, dict):
            raise ValueError(
                "Model norm_stats is missing. Please provide a valid dataset_statistics.json via "
                "cfg.dataset_statistics_path or place it under cfg.pretrained_checkpoint."
            )
        if cfg.unnorm_key not in vla.norm_stats:
            raise KeyError(
                f"unnorm_key '{cfg.unnorm_key}' not found in model.norm_stats. "
                f"Available keys: {list(vla.norm_stats.keys())}"
            )
    processor = get_processor(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        proprio_projector.eval()
        for p in proprio_projector.parameters():
            p.requires_grad_(False)

    # Frozen action head for supervision/readout.
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    action_head.eval()
    for p in action_head.parameters():
        p.requires_grad_(False)

    # Freeze all VLA params first.
    for p in vla.parameters():
        p.requires_grad_(False)

    # Build temporal modules outside of modeling_prismatic.
    vit_t_backbone = PrismaticVisionBackbone(
        False,
        [cfg.vit_t_image_size],
        [cfg.vit_t_timm_model_id],
        [None],
        pretrained=True,
    )
    vit_t_backbone = vit_t_backbone.to(device=DEVICE, dtype=model_dtype)
    vit_t_backbone.eval()
    vit_t_backbone.requires_grad_(False)

    temporal_fusion_fc = nn.Linear(vla.llm_dim + vit_t_backbone.embed_dim, vla.llm_dim)
    temporal_fusion_fc = temporal_fusion_fc.to(device=DEVICE, dtype=model_dtype)

    # Unfreeze last LLaMA layer.
    for p in vla.language_model.model.layers[-1].parameters():
        p.requires_grad_(True)

    # Train temporal fusion FC (external module).
    for p in temporal_fusion_fc.parameters():
        p.requires_grad_(True)

    trainable = [p for p in vla.parameters() if p.requires_grad] + list(temporal_fusion_fc.parameters())
    print(f"[Trainable params] {sum(p.numel() for p in trainable)}")

    # Keep eval mode for stable teacher-forcing style hidden-state extraction during training.
    vla.eval()
    temporal_fusion_fc.train()

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.L1Loss()

    resume_path = None
    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
    elif cfg.auto_resume_latest:
        resume_path = find_latest_temporal_fusion_checkpoint(save_dir)

    start_epoch = 0
    global_step = 0
    if resume_path is not None and resume_path.exists():
        resume_info = load_temporal_fusion_checkpoint(
            resume_path,
            vla=vla,
            temporal_fusion_fc=temporal_fusion_fc,
            optimizer=optimizer,
            map_location="cpu",
            load_optimizer=True,
        )
        start_epoch = int(resume_info["epoch"])
        global_step = int(resume_info["global_step"])
        print(f"[Resume] loaded {resume_path}; start_epoch={start_epoch}, global_step={global_step}")

    ds = LiberoChunkDataset(
        cfg.tfrecord_glob,
        full_num_actions_chunk=cfg.full_num_actions_chunk,
        light_num_actions_chunk=cfg.light_num_actions_chunk,
        start_from_any_t=cfg.start_from_any_t,
    )
    effective_persistent_workers = cfg.persistent_workers if cfg.num_workers > 0 else False
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.start_from_any_t,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=effective_persistent_workers,
        collate_fn=collate_fn,
    )

    for epoch in range(start_epoch, cfg.epochs):
        print(f"[Epoch {epoch+1}/{cfg.epochs}] starting...")
        running = 0.0
        printed_batches = 0

        for batch in dl:
            optimizer.zero_grad(set_to_none=True)
            action_chunk_seq = batch["action_chunk_seq"].to(DEVICE, non_blocking=True)
            batch_size_curr = action_chunk_seq.shape[0]
            # dt schedule:
            # - sample one training step every `light_num_actions_chunk` frames (e.g. every 8 frames)
            # - aligns with controller validation cadence used at inference
            max_dt = cfg.full_num_actions_chunk - cfg.light_num_actions_chunk
            if max_dt <= 0:
                max_dt = cfg.full_num_actions_chunk - 1
            frame_stride = max(1, int(cfg.light_num_actions_chunk))
            dt_start = min(frame_stride, max_dt) if max_dt > 0 else 1
            dt_values = list(range(dt_start, max_dt + 1, frame_stride)) if max_dt > 0 else [1]
            if len(dt_values) == 0:
                dt_values = [1]

            train_steps_per_sample = len(dt_values)
            normalizer = float(batch_size_curr * train_steps_per_sample)
            loss_value_sum = 0.0

            # t=0: full-model inference for the whole batch to populate temporal caches.
            obs0_batch = {
                "full_image": batch["full_image_seq_by_t"][0],
                "wrist_image": batch["wrist_image_seq_by_t"][0],
                "instruction": batch["instruction_seq_by_t"][0],
                "state": batch["state_seq_by_t"][0],
            }
            inputs0, proprio0 = _prepare_policy_inputs_batch(cfg, vla, processor, obs0_batch, model_dtype)
            with torch.no_grad():
                cache = _bootstrap_first_frame_cache(
                    cfg,
                    vla,
                    inputs0,
                    proprio0,
                    proprio_projector=proprio_projector,
                )

            # Sparse t-steps: small model runs every `frame_stride` frames.
            for dt in dt_values:
                obs_dt_batch = {
                    "full_image": batch["full_image_seq_by_t"][dt],
                    "wrist_image": batch["wrist_image_seq_by_t"][dt],
                    "instruction": batch["instruction_seq_by_t"][dt],
                    "state": batch["state_seq_by_t"][dt],
                }
                inputs_dt, _ = _prepare_policy_inputs_batch(cfg, vla, processor, obs_dt_batch, model_dtype)
                obs_light = {"pixel_values": inputs_dt["pixel_values"]}

                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_autocast_dtype)
                    if (DEVICE.type == "cuda" and amp_autocast_dtype is not None)
                    else nullcontext()
                )
                with amp_ctx:
                    hs = build_inputs_and_extract_hs(
                        cfg,
                        vla,
                        obs_light,
                        cache=cache,
                        vit_t_backbone=vit_t_backbone,
                        temporal_fusion_fc=temporal_fusion_fc,
                    )
                    pred_chunk = action_head.predict_action(hs.to(action_head.model.fc1.weight.dtype))
                    pred_chunk = pred_chunk.to(torch.float32)  # (B, N, A)

                gt_aligned = action_chunk_seq[:, dt, :, :]  # (B, light_chunk, A)
                step_loss = loss_fn(pred_chunk, gt_aligned)
                loss_value_sum += float(step_loss.detach().item() * batch_size_curr)

                # Backprop immediately to avoid retaining a huge graph for all frames.
                (step_loss / train_steps_per_sample).backward()

            optimizer.step()

            if cfg.empty_cuda_cache_every_step and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            running += loss_value_sum / normalizer
            printed_batches += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg_loss = running / printed_batches if printed_batches > 0 else float("nan")
                print(f"[Epoch {epoch+1}] step={global_step} loss={avg_loss:.6f}")
                running = 0.0
                printed_batches = 0

        ckpt = save_dir / f"vla_last_layer_temporal_fc_epoch_{epoch+1}.pt"
        save_temporal_fusion_checkpoint(
            ckpt,
            vla=vla,
            temporal_fusion_fc=temporal_fusion_fc,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            extra={"cfg": cfg.__dict__},
        )
        save_temporal_fusion_checkpoint(
            save_dir / "latest.pt",
            vla=vla,
            temporal_fusion_fc=temporal_fusion_fc,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            extra={"cfg": cfg.__dict__},
        )
        print(f"[Save] {ckpt}")

    print("[Done] training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal-fusion with separated full/light action chunks.")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/path/to/openvla-oft",
        help="Path or HF id for pretrained OpenVLA checkpoint.",
    )
    parser.add_argument(
        "--tfrecord-glob",
        type=str,
        default=(
            "/path/to/modified_libero_rlds"
        ),
        help="Glob for TFRecord shards.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/path/to/temporal_fusion_lastlayer",
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    # Keep original defaults available as optional CLI controls.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--full-num-actions-chunk", type=int, default=64)
    parser.add_argument("--light-num-actions-chunk", type=int, default=8)

    args = parser.parse_args()

    cfg = TrainCfg(
        pretrained_checkpoint=args.pretrained_checkpoint,
        tfrecord_glob=args.tfrecord_glob,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        log_every=args.log_every,
        full_num_actions_chunk=args.full_num_actions_chunk,
        light_num_actions_chunk=args.light_num_actions_chunk,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    train(cfg)
