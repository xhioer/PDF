"""Re-score fixed train semantic confusers before/after formal training."""
from __future__ import absolute_import

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.default_img import _C
from models.clip_model import build_CLIP_from_openai_pretrained
from tools.rchrl_common import (ImagePathDataset, REPO_ROOT, feature_transform,
                                 image_io_path, json_dump, load_train_records,
    sha256_file, stable_hash)
from tools.rchrl_common import projected_image_cls


CORE_PATHS = {
    (0, "R00"): "seed0/R00_control",
    (0, "R04"): "seed0/R04_hybrid",
    (0, "R07"): "seed0/R07_hybrid_joint",
    (0, "R11"): "seed0/R11_full",
    (1, "R00"): "seed1/R00",
    (1, "R04"): "seed1/R04",
    (1, "R07"): "seed1/R07",
    (1, "R11"): "seed1/R11",
    (2, "R00"): "seed2/R00",
    (2, "R04"): "seed2/R04",
    (2, "R07"): "seed2/R07",
    (2, "R11"): "seed2/R11",
}


def extract_features(model, records, device):
    config = _C.clone()
    config.defrost()
    config.DATA.HEIGHT = 384
    config.DATA.WIDTH = 128
    config.freeze()
    loader = DataLoader(ImagePathDataset(records, feature_transform(config)),
                        batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True)
    result = np.zeros((len(records), 512), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            values = torch.nn.functional.normalize(
                projected_image_cls(model, images.to(device, non_blocking=True)).float(), dim=1)
            result[indices.numpy()] = values.cpu().numpy()
    return result


def rank_of_negative(features, anchor, negative, pids, hashes):
    scores = np.dot(features, features[anchor])
    valid = np.flatnonzero(pids != pids[anchor])
    target = float(scores[negative])
    greater = int(np.sum(scores[valid] > target))
    equal_before = int(np.sum((scores[valid] == target) &
                              (hashes[valid] < hashes[negative])))
    return greater + equal_before + 1


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model, _ = build_CLIP_from_openai_pretrained("ViT-B/16", (384, 128), 16, 150)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().float().to(device)
    model.freeze_text_encoder()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records, _, _, _ = load_train_records()
    graph = np.load(os.path.join(args.graph, "relation_index.npz"))
    before = np.load(os.path.join(args.graph, "visual_features.npy"))
    semantic_features = np.load(os.path.join(args.graph, "semantic_features.npy"))
    pids = np.asarray([row["person_id"] for row in records], dtype=np.int32)
    hashes = np.asarray([int(stable_hash(row["path"])[:16], 16) for row in records], dtype=np.uint64)
    rows = []
    missing = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("semantic confuser analysis requires CUDA")
    for (seed, run_id), relative in CORE_PATHS.items():
        checkpoint_path = os.path.join(args.root, relative, "epoch50_final.pth")
        if not os.path.exists(checkpoint_path):
            missing.append({"seed": seed, "run_id": run_id, "checkpoint": checkpoint_path})
            continue
        print("analyzing seed={} {}".format(seed, run_id), flush=True)
        model = load_model(checkpoint_path, device)
        after = extract_features(model, records, device)
        for confuser_set, indices in (("semantic_top1", graph["semantic_neg"][:, 0]),
                                      ("hybrid_top1", graph["hybrid_neg"][:, 0])):
            for anchor in range(min(100, len(records))):
                negative = int(indices[anchor])
                before_similarity = float(np.dot(before[anchor], before[negative]))
                after_similarity = float(np.dot(after[anchor], after[negative]))
                rows.append({
                    "seed": seed, "run_id": run_id, "confuser_set": confuser_set,
                    "anchor_index": anchor, "negative_index": negative,
                    "anchor_path": records[anchor]["path"],
                    "negative_path": records[negative]["path"],
                    "anchor_person_id": int(pids[anchor]),
                    "negative_person_id": int(pids[negative]),
                    "visual_similarity_before": before_similarity,
                    "visual_similarity_after": after_similarity,
                    "visual_rank_before": rank_of_negative(before, anchor, negative, pids, hashes),
                    "visual_rank_after": rank_of_negative(after, anchor, negative, pids, hashes),
                    "semantic_similarity_before": float(np.dot(
                        semantic_features[anchor], semantic_features[negative])),
                })
        del model, after
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["seed", "run_id"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows": len(rows), "missing": missing, "fixed_anchor_count": min(100, len(records)),
               "feature_definition": "raw projected CLS encode_image, L2 normalized; no flip",
               "semantic_confusers": "frozen graph semantic_top1 and hybrid_top1 different-ID train negatives",
               "test_data_used": False}
    if rows:
        for key in ("semantic_top1", "hybrid_top1"):
            subset = [row for row in rows if row["confuser_set"] == key]
            summary[key] = {
                "count": len(subset),
                "mean_similarity_before": float(np.mean([x["visual_similarity_before"] for x in subset])),
                "mean_similarity_after": float(np.mean([x["visual_similarity_after"] for x in subset])),
                "mean_delta": float(np.mean([x["visual_similarity_after"] - x["visual_similarity_before"] for x in subset])),
                "mean_rank_before": float(np.mean([x["visual_rank_before"] for x in subset])),
                "mean_rank_after": float(np.mean([x["visual_rank_after"] for x in subset])),
                "rank_improved_fraction": float(np.mean([x["visual_rank_after"] > x["visual_rank_before"] for x in subset])),
            }
    json_dump(summary, os.path.splitext(args.output)[0] + ".json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
