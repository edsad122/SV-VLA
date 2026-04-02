"""Test script for external `InferenceController` behavior.

What this script tests:
1) Frame-1 in each chunk uses full-model anchor (`used_mode=full`).
2) Frames 2..N use lightweight branch (`used_mode=lightweight`).
3) Frame N+1 starts a new chunk (`used_mode=full`).
4) Optional validation can force re-anchor (`used_mode=full_reanchor`).
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from prismatic.vla.constants import ACTION_DIM
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.extern.hf.inference_controller import InferenceController
from temporal_fusion_utils import load_openvla_with_temporal_fusion
from transformers import AutoProcessor


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


def _dtype_for_device(device: torch.device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vit_t_timm_model_id", type=str, default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--vit_t_image_size", type=int, default=224)
    parser.add_argument("--validation_threshold", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype_for_device(device)

    print("[1/4] Load processor + model")
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
    model.eval()
    model.vision_backbone.set_num_images_in_input(1)

    if getattr(model, "norm_stats", None) is None:
        model.norm_stats = _make_debug_norm_stats()
    unnorm_key = "debug" if "debug" in model.norm_stats else next(iter(model.norm_stats.keys()))

    prompt = "In: What action should the robot take to move the block to the target?\nOut:"

    def infer_chunk(obs: Dict, _task: str) -> List[np.ndarray]:
        image = obs["image"]
        model_inputs = processor(prompt, image).to(device, dtype=dtype)
        actions, _ = model.predict_action(
            **model_inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            reset_first_frame_cache=bool(obs.get("is_first_frame", False)),
        )
        return [actions[i] for i in range(len(actions))]

    print("[2/4] Controller without validation")
    controller = InferenceController(
        run_full_chunk_fn=infer_chunk,
        run_light_chunk_fn=infer_chunk,
        use_temporal_fusion=True,
        full_num_actions_chunk=NUM_ACTIONS_CHUNK,
        light_num_actions_chunk=NUM_ACTIONS_CHUNK,
        enable_validation=False,
        prime_light_cache_fn=infer_chunk,
    )
    controller.reset()

    used_modes = []
    for _ in range(NUM_ACTIONS_CHUNK + 1):
        out = controller.step({"image": _make_random_image()}, "debug")
        used_modes.append(out["used_mode"])

    print(f"  first mode={used_modes[0]}, second mode={used_modes[1] if len(used_modes) > 1 else 'N/A'}")
    print(f"  mode at frame N+1={used_modes[-1]}")

    print("[3/4] Controller with forced re-anchor validation")
    controller_val = InferenceController(
        run_full_chunk_fn=infer_chunk,
        run_light_chunk_fn=infer_chunk,
        use_temporal_fusion=True,
        full_num_actions_chunk=NUM_ACTIONS_CHUNK,
        light_num_actions_chunk=NUM_ACTIONS_CHUNK,
        enable_validation=True,
        prime_light_cache_fn=infer_chunk,
        deviation_threshold=-1.0,
    )
    controller_val.reset()

    out0 = controller_val.step({"image": _make_random_image()}, "debug")
    out1 = controller_val.step({"image": _make_random_image()}, "debug")

    print(f"  step0 mode={out0['used_mode']}")
    print(f"  step1 mode={out1['used_mode']} (expect full_reanchor)")

    print("[4/4] Summary")
    ok = bool(
        used_modes[0] == "full"
        and used_modes[-1] == "full"
        and (len(used_modes) == 1 or used_modes[1] == "lightweight")
        and out1["used_mode"] == "full_reanchor"
    )
    print(f"  PASS={ok}")


if __name__ == "__main__":
    main()
