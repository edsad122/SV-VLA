from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


def _is_hf_repo_id(path_or_repo: str) -> bool:
    return not os.path.isdir(path_or_repo)


def load_dataset_stats_into_model(
    *,
    model: OpenVLAForActionPrediction,
    pretrained_checkpoint: str,
    dataset_statistics_path: Optional[str] = None,
) -> None:
    stats_path = dataset_statistics_path
    if stats_path is None:
        if _is_hf_repo_id(pretrained_checkpoint):
            stats_path = hf_hub_download(repo_id=pretrained_checkpoint, filename="dataset_statistics.json")
        else:
            stats_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")

    if not os.path.isfile(stats_path):
        print(f"[Warn] dataset_statistics.json not found: {stats_path}")
        return

    with open(stats_path, "r", encoding="utf-8") as f:
        model.norm_stats = json.load(f)

    if isinstance(model.norm_stats, dict):
        print(f"[Stats] loaded dataset statistics from {stats_path}; keys={list(model.norm_stats.keys())[:5]}")


def load_openvla_with_temporal_fusion(
    *,
    pretrained_checkpoint: str,
    device: torch.device,
    num_images_in_input: int,
    vit_t_timm_model_id: str,
    vit_t_image_size: int,
    vit_t_pretrained: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    dataset_statistics_path: Optional[str] = None,
) -> OpenVLAForActionPrediction:
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        model = OpenVLAForActionPrediction.from_pretrained(
            pretrained_checkpoint,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            vit_t_timm_model_id=vit_t_timm_model_id,
            vit_t_image_size=vit_t_image_size,
            vit_t_pretrained=vit_t_pretrained,
        )
    except NotImplementedError as e:
        if "meta tensor" not in str(e):
            raise
        print("[Warn] hit meta-tensor loading path; retry with safe loader...")
        model = OpenVLAForActionPrediction.from_pretrained(
            pretrained_checkpoint,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            vit_t_timm_model_id=vit_t_timm_model_id,
            vit_t_image_size=vit_t_image_size,
            vit_t_pretrained=vit_t_pretrained,
        )

    model.vision_backbone.set_num_images_in_input(num_images_in_input)
    load_dataset_stats_into_model(
        model=model,
        pretrained_checkpoint=pretrained_checkpoint,
        dataset_statistics_path=dataset_statistics_path,
    )
    model.eval()
    if not load_in_8bit and not load_in_4bit:
        model = model.to(device)
    return model


def save_temporal_fusion_checkpoint(
    checkpoint_path: str | Path,
    *,
    vla: OpenVLAForActionPrediction,
    temporal_fusion_fc: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tf_fc = temporal_fusion_fc if temporal_fusion_fc is not None else getattr(vla, "temporal_fusion_fc", None)
    if tf_fc is None:
        raise AttributeError("No temporal_fusion_fc provided and model has no `temporal_fusion_fc` attribute.")

    payload: Dict[str, Any] = {
        "temporal_fusion_fc": tf_fc.state_dict(),
        "llama_last_layer": vla.language_model.model.layers[-1].state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, checkpoint_path)


def load_temporal_fusion_checkpoint(
    checkpoint_path: str | Path,
    *,
    vla: OpenVLAForActionPrediction,
    temporal_fusion_fc: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    load_optimizer: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location)
    if "temporal_fusion_fc" not in ckpt or "llama_last_layer" not in ckpt:
        raise KeyError("Checkpoint missing `temporal_fusion_fc` or `llama_last_layer`.")

    tf_fc = temporal_fusion_fc if temporal_fusion_fc is not None else getattr(vla, "temporal_fusion_fc", None)
    if tf_fc is None:
        raise AttributeError("No temporal_fusion_fc provided and model has no `temporal_fusion_fc` attribute.")

    tf_fc.load_state_dict(ckpt["temporal_fusion_fc"], strict=True)
    vla.language_model.model.layers[-1].load_state_dict(ckpt["llama_last_layer"], strict=True)

    if load_optimizer and optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "extra": ckpt.get("extra", None),
    }


def find_latest_temporal_fusion_checkpoint(save_dir: str | Path) -> Optional[Path]:
    save_dir = Path(save_dir)
    latest = save_dir / "latest.pt"
    if latest.exists():
        return latest

    candidates = sorted(save_dir.glob("vla_last_layer_temporal_fc_epoch_*.pt"))
    if not candidates:
        return None
    return candidates[-1]
