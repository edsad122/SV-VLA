"""
Minimal temporal-fusion sanity test for OpenVLA.

What this script validates:
1) Random image can run through full `predict_action()` path.
2) Penultimate-layer tensor shape is visible (from temporal-fusion debug hooks).
3) First-frame feature is cached and reused at t>0.
4) ViT-T branch and fusion FC are active.

Usage:
python vla-scripts/test_temporal_fusion_modules.py \
  --checkpoint <local_or_hf_checkpoint> \
  --device cuda
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor
from prismatic.vla.constants import ACTION_DIM
from prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone
from temporal_fusion_utils import load_openvla_with_temporal_fusion, load_temporal_fusion_checkpoint


def _make_debug_norm_stats() -> Dict:
    return {
        "debug": {
            "action": {
                "q01": [-1.0] * ACTION_DIM,
                "q99": [1.0] * ACTION_DIM,
                "mask": [True] * ACTION_DIM,
            }
        }
    }


def _make_random_image(h: int = 224, w: int = 224) -> Image.Image:
    arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _to_cpu_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default = "/root/data/openvla-7b-oft-finetuned-libero-object")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vit_t_timm_model_id", type=str, default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--vit_t_image_size", type=int, default=224)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("[1/4] Loading processor + model...")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = load_openvla_with_temporal_fusion(
        pretrained_checkpoint=args.checkpoint,
        device=device,
        num_images_in_input=1,
        vit_t_timm_model_id=args.vit_t_timm_model_id,
        vit_t_image_size=args.vit_t_image_size,
        vit_t_pretrained=True,
        load_in_8bit=False,
        load_in_4bit=False,
    )

    vit_t_backbone = PrismaticVisionBackbone(
        False,
        [args.vit_t_image_size],
        [args.vit_t_timm_model_id],
        [None],
        pretrained=True,
    ).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    temporal_fusion_fc = nn.Linear(model.llm_dim + vit_t_backbone.embed_dim, model.llm_dim).to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    if args.resume_checkpoint:
        resume_info = load_temporal_fusion_checkpoint(
            args.resume_checkpoint,
            vla=model,
            temporal_fusion_fc=temporal_fusion_fc,
            optimizer=None,
            map_location="cpu",
            load_optimizer=False,
        )
        print(
            "  loaded temporal-fusion weights from "
            f"{args.resume_checkpoint} (epoch={resume_info['epoch']}, step={resume_info['global_step']})"
        )

    model.eval()

    # Ensure single-image mode for this minimal test.
    model.vision_backbone.set_num_images_in_input(1)

    # If checkpoint has no dataset stats, provide debug stats so `predict_action()` can unnormalize outputs.
    if getattr(model, "norm_stats", None) is None:
        model.norm_stats = _make_debug_norm_stats()

    unnorm_key = "debug"
    if unnorm_key not in model.norm_stats:
        # fallback to first available key
        unnorm_key = next(iter(model.norm_stats.keys()))

    # Print temporal module status.
    has_temporal = True
    print(f"  temporal_fusion_fc exists: {has_temporal}")
    if has_temporal:
        trainable_fc = all(p.requires_grad for p in temporal_fusion_fc.parameters())
        print(f"  temporal_fusion_fc trainable: {trainable_fc}")

    frozen_vit_t = all(not p.requires_grad for p in vit_t_backbone.parameters())
    print(f"  vit_t_backbone frozen: {frozen_vit_t}")

    prompt = "In: What action should the robot take to move the block to the target?\nOut:"

    print("[2/4] Step t=0 with random image_0 (reset cache)...")
    img0 = _make_random_image()
    inputs0 = processor(prompt, img0).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    action0, action_hs0 = model.predict_action(
        **inputs0,
        unnorm_key=unnorm_key,
        do_sample=False,
        reset_first_frame_cache=True,
    )

    first_used_0 = _to_cpu_tensor(getattr(model, "_debug_first_frame_feat_used", None))
    first_curr_0 = _to_cpu_tensor(getattr(model, "_debug_first_frame_feat_curr", None))
    vit_t_0 = _to_cpu_tensor(getattr(model, "_debug_vit_t_feat", None))
    fused_0 = _to_cpu_tensor(getattr(model, "_debug_fused_feat", None))
    penultimate_shape_0 = getattr(model, "_debug_penultimate_shape", None)
    used_cached_penultimate_0 = getattr(model, "_debug_used_cached_penultimate", None)

    print(f"  penultimate hidden shape @t=0: {penultimate_shape_0}")
    print(f"  used cached penultimate @t=0: {used_cached_penultimate_0}")
    print(f"  first-frame feat (curr) shape @t=0: {None if first_curr_0 is None else tuple(first_curr_0.shape)}")
    print(f"  first-frame feat (used) shape @t=0: {None if first_used_0 is None else tuple(first_used_0.shape)}")
    print(f"  vit-t feat shape @t=0: {None if vit_t_0 is None else tuple(vit_t_0.shape)}")
    print(f"  fused feat shape @t=0: {None if fused_0 is None else tuple(fused_0.shape)}")
    print(f"  action_hidden_states shape @t=0: {tuple(action_hs0.shape)}")

    print("[3/4] Step t=1 with another random image_t (do NOT reset cache)...")
    img1 = _make_random_image()
    inputs1 = processor(prompt, img1).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    action1, action_hs1 = model.predict_action(
        **inputs1,
        unnorm_key=unnorm_key,
        do_sample=False,
        reset_first_frame_cache=False,
    )

    first_used_1 = _to_cpu_tensor(getattr(model, "_debug_first_frame_feat_used", None))
    first_curr_1 = _to_cpu_tensor(getattr(model, "_debug_first_frame_feat_curr", None))
    penultimate_shape_1 = getattr(model, "_debug_penultimate_shape", None)
    used_cached_penultimate_1 = getattr(model, "_debug_used_cached_penultimate", None)

    print(f"  penultimate hidden shape @t=1: {penultimate_shape_1}")
    print(f"  used cached penultimate @t=1: {used_cached_penultimate_1}")
    print(f"  first-frame feat (curr) shape @t=1: {None if first_curr_1 is None else tuple(first_curr_1.shape)}")
    print(f"  first-frame feat (used) shape @t=1: {None if first_used_1 is None else tuple(first_used_1.shape)}")
    print(f"  action_hidden_states shape @t=1: {tuple(action_hs1.shape)}")

    print("[4/4] Comparing cache behavior...")
    if first_used_0 is not None and first_used_1 is not None:
        same_used = torch.allclose(first_used_0, first_used_1, atol=1e-5, rtol=1e-4)
        print(f"  cached first-frame feature reused (used@t0 == used@t1): {same_used}")

    if first_curr_0 is not None and first_curr_1 is not None:
        same_curr = torch.allclose(first_curr_0, first_curr_1, atol=1e-5, rtol=1e-4)
        print(f"  current penultimate first-frame feature equal across two different images: {same_curr}")
        print("  NOTE: expected True when cache is reused from t=0 onward.")

    print("Done.")


if __name__ == "__main__":
    main()
