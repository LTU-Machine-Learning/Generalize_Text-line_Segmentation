import os
import torch
from typing import Optional, Dict

def export_hisam_for_inference(
    model,
    out_path: str,
    *,
    model_type: str,
    dataset: str,
    zero_shot: bool,
    fg_points_num: int,
    score_thresh: float,
    nms_thresh: float,
    extra_cfg: Optional[Dict] = None,
):
    model_to_save = model.module if hasattr(model, "module") else model

    payload = {
        "task": "line_segmentation",
        "arch": "hisam",
        "version": 1,
        "model_type": model_type,
        "state_dict": model_to_save.state_dict(),
        "infer_cfg": {
            "dataset": dataset,
            "zero_shot": bool(zero_shot),
            "fg_points_num": int(fg_points_num),
            "score_thresh": float(score_thresh),
            "nms_thresh": float(nms_thresh),
        },
        "extra_cfg": extra_cfg or {},
    }

    torch.save(payload, out_path)
    print(f"Exported Hi-SAM inference artifact: {out_path}")
