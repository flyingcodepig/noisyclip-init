import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from infer import pixels, sha
from model_base import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/stage1/model.pt",
    )
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = json.loads((ROOT / "data/train_index.json").read_text(encoding="utf-8"))
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    candidates = np.load(ROOT / "cache/train_candidate_labels.npy", mmap_mode="r")
    cache = np.load(ROOT / "image_cache/semi_train_512.npy", mmap_mode="r")
    count = len(rows) if args.limit is None else min(args.limit, len(rows))
    if count < 1 or len(cache) != len(rows) or candidates.shape[0] != len(rows):
        raise ValueError("TRAIN data size mismatch")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads((ROOT / "train_config_stage1.json").read_text(encoding="utf-8"))
    if state["config"] != config:
        raise ValueError("Checkpoint configuration mismatch")
    model = load_model(ROOT, config, ROOT, initialize_official=False)
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.cuda().eval()
    matched_given = 0
    matched_candidate = 0
    with torch.inference_mode():
        for start in range(0, count, args.batch):
            stop = min(count, start + args.batch)
            images = pixels(cache, np.arange(start, stop), config["size"])
            with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(images)
            predicted = logits.argmax(-1).cpu().numpy()
            matched_given += int(np.count_nonzero(predicted == labels[start:stop]))
            matched_candidate += int(np.count_nonzero((predicted[:, None] == candidates[start:stop]).any(axis=1)))
    report = {
        "evaluation_set": "official_train_in_sample",
        "checkpoint_sha256": sha(args.checkpoint),
        "rows": count,
        "given_label_top1": matched_given / count,
        "candidate_label_top1": matched_candidate / count,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
