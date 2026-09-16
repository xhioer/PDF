#!/usr/bin/env python3
"""PDF Representation Leakage & Over-Suppression Diagnosis.

This tool is deliberately diagnostic-only.  It reconstructs the frozen PDF
model from three epoch-50 R00 checkpoints, extracts representations without
changing training code, and writes auditable train-only analyses.

The command stages are:

    prepare  - provenance, code audit, frozen split and pair manifests
    extract  - layer-wise visual / cloth-component features for one seed
    analyze  - H1/H2 metrics, offline alpha sweep and bootstrap summaries
    figures  - static PNG figures
    report   - final Markdown report
    all      - prepare, all extractions, analysis, figures and report

Run with the repository's PDF environment, for example:

    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64 \
      /data/envs/PDF/bin/python tools/pdf_representation_diagnosis.py all \
      --device cuda:0

No PRCC validation or test path is read by this file.  The only semantic
split consumed is the previously frozen 148-ID / 888-image TRAIN manifest.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "pdf_representation_diagnosis")
REPORT_ROOT = os.path.join(REPO_ROOT, "reports")
DATA_ROOT = "/data/datasets/PRCC"
TRAIN_ROOT = os.path.join(DATA_ROOT, "prcc", "rgb", "train")
CAPTION_PATH = os.path.join(REPO_ROOT, "data", "captions", "prcc.json")
SPLIT_SOURCE = (
    "/data/projects/PDF-worktrees/pdf-finegrained-semantics-v3/outputs/"
    "finegrained_semantics_v3/semantic_split_manifest.json"
)
SELECTION_SOURCE = (
    "/data/projects/PDF-worktrees/pdf-finegrained-semantics-v3/outputs/"
    "finegrained_semantics_v3/semantic_selection_888.json"
)
CHECKPOINT_CANDIDATES = {
    0: [
        "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs/"
        "crossclothes_relation_v1/seed0/R00_control/epoch50_final.pth",
        "/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/"
        "crossclothes_relation_confirm/seed0/R00/epoch50_final.pth",
    ],
    1: [
        "/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/"
        "crossclothes_relation_confirm/seed1/R00/epoch50_final.pth",
    ],
    2: [
        "/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs/"
        "crossclothes_relation_confirm/seed2/R00/epoch50_final.pth",
    ],
}
SOURCE_COMMIT_FALLBACK = {
    0: "7c890687ce51b6290e577ea2ee1b3cdbf6421214",
    1: "fe2eba095b5b1488d05aef723abe322ca83b867b",
    2: "fe2eba095b5b1488d05aef723abe322ca83b867b",
}
SEEDS = (0, 1, 2)
ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
PAIR_SEED = 20260916
NEGATIVE_COUNT = 100000
MAX_POSITIVE_PER_KIND = 100000
BOOTSTRAP_ITERATIONS = 10000
RIDGE_ALPHA = 1.0
EARLY_LAYERS = (1, 2, 3, 4)
MIDDLE_LAYERS = (5, 6, 7, 8)
LATE_LAYERS = (9, 10, 11, 12)


def ensure_dirs():
    paths = [
        OUT_ROOT,
        REPORT_ROOT,
        os.path.join(OUT_ROOT, "shared", "pair_manifest"),
        os.path.join(OUT_ROOT, "shared", "negative_manifest"),
        os.path.join(OUT_ROOT, "shared", "split_manifest"),
        os.path.join(OUT_ROOT, "figures"),
        REPORT_ROOT,
    ]
    for seed in SEEDS:
        paths.extend([
            os.path.join(OUT_ROOT, "seed{}".format(seed), "layers"),
            os.path.join(OUT_ROOT, "seed{}".format(seed), "decomposition"),
            os.path.join(OUT_ROOT, "seed{}".format(seed), "alpha_sweep"),
        ])
    for path in paths:
        os.makedirs(path, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_jsonl(rows, path):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(rows, path, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonical(path):
    return os.path.normpath(str(path))


def alpha_tag(alpha):
    return "{:.2f}".format(float(alpha)).replace(".", "p")


def seed_root(seed):
    return os.path.join(OUT_ROOT, "seed{}".format(seed))


def npy_sha(path):
    return sha256_file(path) if os.path.isfile(path) else None


def l2_normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def discover_checkpoint(seed):
    candidates = [p for p in CHECKPOINT_CANDIDATES[seed] if os.path.isfile(p)]
    if not candidates:
        search_roots = [
            "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/outputs",
            "/data/projects/PDF-worktrees/pdf-crossclothes-relation-confirm/outputs",
        ]
        for root in search_roots:
            pattern = os.path.join(root, "**", "seed{}".format(seed), "**", "epoch50_final.pth")
            candidates.extend(glob.glob(pattern, recursive=True))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError("Cannot locate a complete R00 epoch50_final checkpoint for seed{}".format(seed))

    # Validate every candidate before selecting one.  A non-R00 epoch50 file
    # must never silently become a baseline.
    valid = []
    for path in candidates:
        try:
            payload = load_checkpoint_cpu(path)
            role = payload.get("checkpoint_role")
            run_id = payload.get("run_id")
            epoch = int(payload.get("epoch", -1))
            train_seed = int(payload.get("training_seed", seed))
            if epoch == 50 and run_id == "R00" and train_seed == seed and role in ("epoch50_final", "primary epoch50 final"):
                valid.append(path)
        except Exception:
            continue
    if not valid:
        raise RuntimeError(
            "Found epoch50 candidates for seed{} but none is a validated R00 "
            "epoch50_final checkpoint: {}".format(seed, candidates))
    # Prefer the original V1 seed0 control and the confirmatory R00 paths;
    # multiple validated copies are required to be byte-identical.
    hashes = {p: sha256_file(p) for p in valid}
    if len(set(hashes.values())) > 1:
        raise RuntimeError("Multiple non-identical validated checkpoints for seed{}: {}".format(seed, hashes))
    preferred = [p for p in CHECKPOINT_CANDIDATES[seed] if p in valid]
    return preferred[0] if preferred else valid[0]


def load_checkpoint_cpu(path):
    import torch
    return torch.load(path, map_location="cpu")


def find_provenance(checkpoint_path):
    current = os.path.dirname(checkpoint_path)
    for _ in range(5):
        for name in ("run_provenance.json", "run_summary.json"):
            candidate = os.path.join(current, name)
            if os.path.isfile(candidate):
                try:
                    return candidate, read_json(candidate)
                except Exception:
                    pass
        current = os.path.dirname(current)
    return None, {}


def checkpoint_manifest():
    manifest = {
        "experiment": "PDF-RLOS-Diagnosis",
        "checkpoint_role": "R00 epoch50 final; best-test auxiliary is not selected",
        "test_checkpoint_selection_used": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_worktree": REPO_ROOT,
        "checkpoints": [],
    }
    for seed in SEEDS:
        path = discover_checkpoint(seed)
        payload = load_checkpoint_cpu(path)
        provenance_path, provenance = find_provenance(path)
        state = payload.get("model_state_dict", {})
        config = provenance.get("config_snapshot")
        if config is None:
            config = provenance.get("config")
        if config is None:
            config = payload.get("train_protocol")
        entry = {
            "seed": seed,
            "checkpoint_path": path,
            "checkpoint_sha256": sha256_file(path),
            "checkpoint_bytes": os.path.getsize(path),
            "epoch": int(payload.get("epoch", -1)),
            "run_id": payload.get("run_id"),
            "checkpoint_role": payload.get("checkpoint_role"),
            "training_seed": int(payload.get("training_seed", seed)),
            "best_test_epoch_auxiliary": payload.get("best_test_epoch_auxiliary"),
            "best_test_diff_r1_auxiliary": payload.get("best_test_diff_r1_auxiliary"),
            "source_commit": provenance.get("source_commit", SOURCE_COMMIT_FALLBACK[seed]),
            "provenance_path": provenance_path,
            "config": config,
            "state_dict_key_count": len(state),
            "state_dict_key_sha256": sha256_bytes("\n".join(sorted(state.keys())).encode("utf-8")),
            "source_core_file_sha256": sha256_file(os.path.join(REPO_ROOT, "models", "clip_model.py")),
        }
        if entry["epoch"] != 50 or entry["run_id"] != "R00" or entry["checkpoint_role"] not in ("epoch50_final", "primary epoch50 final"):
            raise RuntimeError("Checkpoint validation failed for seed{}: {}".format(seed, entry))
        manifest["checkpoints"].append(entry)
    return manifest


def make_code_audit(manifest):
    """Write an audit from the current checked-out model code and checkpoint shapes."""
    import re
    clip_path = os.path.join(REPO_ROOT, "models", "clip_model.py")
    lines = open(clip_path).read().splitlines()
    state = load_checkpoint_cpu(manifest["checkpoints"][0]["checkpoint_path"])["model_state_dict"]
    block_ids = sorted({int(k.split(".")[3]) for k in state if k.startswith("visual.transformer.resblocks.")})
    visual_width = int(state["visual.conv1.weight"].shape[0])
    patch = int(state["visual.conv1.weight"].shape[-1])
    proj_shape = list(state["visual.proj"].shape)
    pos_len = int(state["visual.positional_embedding"].shape[0])
    n_patches = pos_len - 1
    n_y = (384 - patch) // 16 + 1
    n_x = (128 - patch) // 16 + 1
    output = []
    output.append("# PDF representation audit\n")
    output.append("Generated from the checked-out `models/clip_model.py` and the validated R00 checkpoint state dictionaries; no paper-level representation definition was substituted.\n")
    output.append("## Source and architecture\n")
    output.append("- Code file: `{}`\n- Code SHA256: `{}`\n- Diagnosis worktree branch: `pdf-representation-diagnosis`\n- Checkpoint-derived visual state keys: `{}`\n".format(
        clip_path, sha256_file(clip_path), len(state)))
    output.append("- Visual backbone: `VisionTransformer` (the checkpoint has `visual.conv1.weight` with shape `{}`).\n".format(list(state["visual.conv1.weight"].shape)))
    output.append("- Input resolution: `(384, 128)`; patch/stride: `({},{})`; grid: `{} x {}`; tokens: `{} patches + CLS = {}`.\n".format(patch, 16, n_y, n_x, n_patches, pos_len))
    text_width = int(state["token_embedding.weight"].shape[1])
    text_block_ids = sorted({int(k.split(".")[2]) for k in state if k.startswith("transformer.resblocks.")})
    output.append("- Visual transformer: `{}` blocks, width `{}`, heads `{}` (`width // 64`); visual projection `{}` maps width `{}` to dimension `{}`.\n".format(len(block_ids), visual_width, visual_width // 64, proj_shape, proj_shape[0], proj_shape[1]))
    output.append("- Text transformer used by `encode_text_irra`: `{}` blocks, width `{}`, heads `{}` (`width // 64`). This head count is taken from the current `CLIP` constructor, not inferred from the in-projection matrix shape.\n".format(len(text_block_ids), text_width, text_width // 64))
    output.append("- Final classification/evaluation feature dimension: `512`; `bottleneck_proj` is `BatchNorm1d(512)`; classifier is `512 -> 150`.\n")
    output.append("\n## Actual forward variables\n")
    output.append("The ViT forward returns `(x, z)`. `z` is the post-transformer, post-`ln_post` token sequence in width 768. `x` is `z @ visual.proj`, in dimension 512. The diagnosis defines:\n\n")
    output.append("- `H_0 ... H_12`: the CLS token before the transformer and after each actual visual transformer block, captured with hooks in the current code.\n- `Z_l = visual.proj(ln_post(H_l))`, dimension 512. `Z_12` is exactly the current final projected visual CLS before BN; `Z_0 ... Z_11` are the same code-defined output-space map applied offline to the corresponding hooked CLS state so layers are comparable.\n- `F_visual = bottleneck_proj(Z_12)`, eval-mode BN output, dimension 512. This is the image-only `model(image)` output in the actual eval branch.\n- `F_cloth = com_proj = bottleneck_proj(EOT(combine(x, encode_text_irra(caption))))`, dimension 512. This is the training-only cloth-related path, evaluated deterministically with the checkpoint BN running statistics.\n- No `F_ir` is guessed or sampled during extraction. It is constructed offline only as `normalize(F_visual - alpha * F_cloth)` in the exact BN-output subtraction space.\n")
    output.append("\n## Actual subtraction and alpha generation\n")
    output.append("The frozen PDF training code uses `ir_features = reference_features_proj - com_proj * alpha`, where both operands are 512-dimensional BN outputs. `alpha` is sampled per sample as `torch.randn(B, 1) * 0.5 + 0.5`; the positive branch draws an independent alpha. This diagnosis does not alter that behavior and does not emulate an unrecorded random draw. It uses the pre-registered offline sweep `alpha ∈ {}`.\n".format(list(ALPHAS)))
    output.append("\n## Projection and inference audit\n")
    output.append("- Visual CLS receives `visual.proj` before `bottleneck_proj`; cloth `combine` is already dimension 512 and then receives the same `bottleneck_proj`.\n- Official image-only inference in the eval branch returns the BN output `F_visual`; the text-conditioned eval branch returns `F_visual + BN(combine(...))`, but it is not used as the image representation here.\n- `encode_image()` is not used because this checkout's helper predates the local `(projected_tokens, hidden_tokens)` visual return signature. The extraction calls `model.visual` directly and reproduces the forward variables above.\n- All diagnostic features are saved raw and L2-normalized; no training mode, optimizer, loss, classifier, prompt, relation branch, or new model head is invoked.\n")
    output.append("\n## Data and scope\n")
    output.append("- Dataset scope: PRCC TRAIN only, using the existing 17,896-image TRAIN inventory. No PRCC TEST metric is read.\n- H1 probe/alignment split: the frozen 148-ID / 888-image manifest, 100 identity-disjoint semantic-dev IDs (600 images) and 48 semantic-val IDs (288 images).\n- The existing parsing cache contains only PRCC TEST masks in this environment; no TRAIN parsing cache was found, so region analysis is skipped.\n")
    audit_path = os.path.join(REPORT_ROOT, "pdf_representation_audit.md")
    with open(audit_path, "w") as handle:
        handle.write("".join(output))
    return audit_path


def load_train_inventory():
    # Importing this frozen helper is read-only and explicitly touches only
    # PRCC TRAIN.  It uses the already validated 17,896-image path cache.
    sys.path.insert(0, REPO_ROOT)
    from tools.rchrl_common import load_train_records
    records, pid_strings, clothes_keys, _ = load_train_records()
    if len(records) != 17896:
        raise RuntimeError("Expected 17,896 PRCC TRAIN records, found {}".format(len(records)))
    if any("/test/" in canonical(row["path"]).lower() for row in records):
        raise RuntimeError("A non-TRAIN path entered the diagnosis inventory")
    return records, pid_strings, clothes_keys


def caption_key_for_path(path):
    marker = "/prcc/"
    path = canonical(path)
    pos = path.find(marker)
    if pos < 0:
        raise RuntimeError("Cannot map caption path: {}".format(path))
    return "data/prcc/" + path[pos + len(marker):]


def build_pair_manifests(records):
    pair_dir = os.path.join(OUT_ROOT, "shared", "pair_manifest")
    neg_dir = os.path.join(OUT_ROOT, "shared", "negative_manifest")
    n = len(records)
    pid = np.asarray([int(row["person_id"]) for row in records], dtype=np.int32)
    camera = np.asarray([row["camera"] for row in records])
    by_pid_cam = defaultdict(lambda: defaultdict(list))
    for idx, row in enumerate(records):
        by_pid_cam[int(row["person_id"])][row["camera"]].append(idx)

    def all_pairs(camera_a, camera_b):
        pairs = []
        for person in sorted(by_pid_cam):
            aa = sorted(by_pid_cam[person].get(camera_a, []))
            bb = sorted(by_pid_cam[person].get(camera_b, []))
            pairs.extend((a, b) for a in aa for b in bb)
        return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)

    candidates = {
        "same_ab": all_pairs("A", "B"),
        "diff_ac": all_pairs("A", "C"),
        "diff_bc": all_pairs("B", "C"),
    }
    positive = {}
    for offset, (kind, values) in enumerate(sorted(candidates.items())):
        rng = random.Random(PAIR_SEED + offset)
        count = min(MAX_POSITIVE_PER_KIND, len(values))
        chosen = sorted(rng.sample(range(len(values)), count)) if count < len(values) else list(range(len(values)))
        positive[kind] = values[np.asarray(chosen, dtype=np.int64)]
        np.save(os.path.join(pair_dir, "positive_{}.npy".format(kind)), positive[kind])

    rng = random.Random(PAIR_SEED + 100)
    negatives = np.empty((NEGATIVE_COUNT, 2), dtype=np.int32)
    filled = 0
    while filled < NEGATIVE_COUNT:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if pid[a] == pid[b]:
            continue
        negatives[filled] = (a, b)
        filled += 1
    np.save(os.path.join(neg_dir, "negative_pairs.npy"), negatives)

    retrieval = {
        "gallery_A": np.asarray([i for i, c in enumerate(camera) if c == "A"], dtype=np.int32),
        "query_B": np.asarray([i for i, c in enumerate(camera) if c == "B"], dtype=np.int32),
        "query_C": np.asarray([i for i, c in enumerate(camera) if c == "C"], dtype=np.int32),
    }
    np.savez(os.path.join(pair_dir, "retrieval_indices.npz"), **retrieval)

    manifest = {
        "pair_seed": PAIR_SEED,
        "negative_count": NEGATIVE_COUNT,
        "positive_sampling": "deterministic uniform sample without replacement, capped at 100,000 per kind",
        "positive_counts": {k: int(len(v)) for k, v in positive.items()},
        "positive_candidate_counts": {k: int(len(v)) for k, v in candidates.items()},
        "negative_pairs_path": os.path.join(neg_dir, "negative_pairs.npy"),
        "negative_pairs_sha256": npy_sha(os.path.join(neg_dir, "negative_pairs.npy")),
        "positive_paths": {},
        "positive_sha256": {},
        "retrieval_indices_path": os.path.join(pair_dir, "retrieval_indices.npz"),
        "retrieval_indices_sha256": npy_sha(os.path.join(pair_dir, "retrieval_indices.npz")),
        "camera_protocol": {
            "same_clothes": "A-B",
            "different_clothes": ["A-C", "B-C"],
            "retrieval": "A gallery versus B and C queries",
        },
    }
    for kind in positive:
        path = os.path.join(pair_dir, "positive_{}.npy".format(kind))
        manifest["positive_paths"][kind] = path
        manifest["positive_sha256"][kind] = npy_sha(path)
    write_json(manifest, os.path.join(pair_dir, "pair_manifest.json"))
    write_json({
        "seed": PAIR_SEED + 100,
        "count": NEGATIVE_COUNT,
        "sampling": "uniform image pairs rejected when person_id matches",
        "representation_independent": True,
        "sha256": manifest["negative_pairs_sha256"],
    }, os.path.join(neg_dir, "negative_manifest.json"))
    return manifest


def freeze_split(records):
    if not os.path.isfile(SPLIT_SOURCE) or not os.path.isfile(SELECTION_SOURCE):
        raise RuntimeError("The previously frozen semantic 148-ID split/selection manifest is missing")
    split = read_json(SPLIT_SOURCE)
    selection = read_json(SELECTION_SOURCE)
    if split.get("split_source") != "TRAIN only" or split.get("test_used_for_selection"):
        raise RuntimeError("Frozen split manifest is not a TRAIN-only non-test split")
    if len(split.get("semantic_dev_ids", [])) != 100 or len(split.get("semantic_val_ids", [])) != 48:
        raise RuntimeError("Frozen split does not contain the required 100/48 ID partition")
    if len(selection.get("records", [])) != 888:
        raise RuntimeError("Frozen semantic selection does not contain 888 images")
    path_to_index = {canonical(row["path"]): int(row["index"]) for row in records}
    selected = []
    missing = []
    for row in selection["records"]:
        path = canonical(row["image_path"])
        if path not in path_to_index:
            missing.append(path)
        else:
            selected.append(path_to_index[path])
    if missing or len(set(selected)) != 888:
        raise RuntimeError("Frozen selection cannot be mapped one-to-one to PRCC TRAIN")
    selected = np.asarray(selected, dtype=np.int32)
    pid_raw = np.asarray([row["person_id_raw"] for row in records])
    dev_ids = set(str(x) for x in split["semantic_dev_ids"])
    val_ids = set(str(x) for x in split["semantic_val_ids"])
    dev = np.asarray(sorted(i for i in selected if str(pid_raw[i]) in dev_ids), dtype=np.int32)
    val = np.asarray(sorted(i for i in selected if str(pid_raw[i]) in val_ids), dtype=np.int32)
    if len(dev) != 600 or len(val) != 288 or set(dev).intersection(set(val)):
        raise RuntimeError("Frozen split image counts or disjointness failed: dev={} val={}".format(len(dev), len(val)))
    split_dir = os.path.join(OUT_ROOT, "shared", "split_manifest")
    shutil.copyfile(SPLIT_SOURCE, os.path.join(split_dir, "semantic_split_manifest.json"))
    shutil.copyfile(SELECTION_SOURCE, os.path.join(split_dir, "semantic_selection_888.json"))
    np.save(os.path.join(split_dir, "semantic_dev_indices.npy"), dev)
    np.save(os.path.join(split_dir, "semantic_val_indices.npy"), val)
    resolution = {
        "source_split_manifest": SPLIT_SOURCE,
        "source_split_sha256": sha256_file(SPLIT_SOURCE),
        "copied_split_sha256": sha256_file(os.path.join(split_dir, "semantic_split_manifest.json")),
        "source_selection_manifest": SELECTION_SOURCE,
        "source_selection_sha256": sha256_file(SELECTION_SOURCE),
        "selected_image_count": int(len(selected)),
        "semantic_dev_ids": sorted(dev_ids),
        "semantic_val_ids": sorted(val_ids),
        "semantic_dev_image_count": int(len(dev)),
        "semantic_val_image_count": int(len(val)),
        "dev_indices_sha256": npy_sha(os.path.join(split_dir, "semantic_dev_indices.npy")),
        "val_indices_sha256": npy_sha(os.path.join(split_dir, "semantic_val_indices.npy")),
        "test_used": False,
    }
    write_json(resolution, os.path.join(split_dir, "split_resolution.json"))
    return resolution


def prepare():
    ensure_dirs()
    manifest = checkpoint_manifest()
    write_json(manifest, os.path.join(REPORT_ROOT, "baseline_checkpoint_manifest.json"))
    audit_path = make_code_audit(manifest)
    records, pid_strings, clothes_keys = load_train_inventory()
    rows = []
    for row in records:
        copied = dict(row)
        copied["caption_key"] = caption_key_for_path(row["path"])
        rows.append(copied)
    inventory_path = os.path.join(OUT_ROOT, "shared", "pair_manifest", "train_image_manifest.jsonl")
    write_jsonl(rows, inventory_path)
    inventory_info = {
        "count": len(rows),
        "sha256": sha256_file(inventory_path),
        "path": inventory_path,
        "train_root": TRAIN_ROOT,
        "person_count": len(pid_strings),
        "clothes_key_count": len(clothes_keys),
        "caption_path": CAPTION_PATH,
        "caption_sha256": sha256_file(CAPTION_PATH),
        "test_used": False,
    }
    write_json(inventory_info, os.path.join(OUT_ROOT, "shared", "pair_manifest", "train_image_manifest.json"))
    pairs = build_pair_manifests(records)
    split = freeze_split(records)
    region_cache = glob.glob("/tmp/rchrl_prcc/prcc/parsing/lip/train/**/*.png", recursive=True)
    region_info = {
        "status": "skipped",
        "reason": "No reliable PRCC TRAIN parsing cache found; discovered TRAIN mask count is {}".format(len(region_cache)),
        "discovered_test_mask_root_not_used": "/tmp/rchrl_prcc/prcc/parsing/lip/test",
        "new_parsing_model_deployed": False,
    }
    write_json(region_info, os.path.join(REPORT_ROOT, "region_analysis_status.json"))
    write_json({
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_manifest": os.path.join(REPORT_ROOT, "baseline_checkpoint_manifest.json"),
        "audit": audit_path,
        "inventory": inventory_info,
        "pair_manifest": pairs,
        "split": split,
        "region_analysis": region_info,
    }, os.path.join(REPORT_ROOT, "prepare_manifest.json"))
    print(json.dumps({"status": "prepared", "train_images": len(rows), "audit": audit_path}, indent=2))


def load_prepared_records():
    path = os.path.join(OUT_ROOT, "shared", "pair_manifest", "train_image_manifest.jsonl")
    if not os.path.isfile(path):
        raise RuntimeError("Preparation outputs are missing; run prepare first")
    rows = read_jsonl(path)
    if len(rows) != 17896:
        raise RuntimeError("Prepared inventory has {} rows, expected 17896".format(len(rows)))
    return rows


def build_model_from_checkpoint(entry):
    import torch
    from models.clip_model import CLIP
    payload = load_checkpoint_cpu(entry["checkpoint_path"])
    state = payload["model_state_dict"]
    visual_block_ids = sorted({int(k.split(".")[3]) for k in state if k.startswith("visual.transformer.resblocks.")})
    text_block_ids = sorted({int(k.split(".")[2]) for k in state if k.startswith("transformer.resblocks.")})
    vision_width = int(state["visual.conv1.weight"].shape[0])
    patch_size = int(state["visual.conv1.weight"].shape[-1])
    embed_dim = int(state["visual.proj"].shape[1])
    context_length = int(state["positional_embedding"].shape[0])
    vocab_size = int(state["token_embedding.weight"].shape[0])
    transformer_width = int(state["token_embedding.weight"].shape[1])
    # The checkpoint projection shape identifies the embedding width, but not
    # the number of attention heads.  The current PDF builder defines this
    # value as transformer_width // 64 (8 for the 512-D text transformer).
    transformer_heads = transformer_width // 64
    num_class = int(state["classifier_proj.weight"].shape[0])
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=(384, 128),
        vision_layers=len(visual_block_ids),
        vision_width=vision_width,
        vision_patch_size=patch_size,
        stride_size=16,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=len(text_block_ids),
        num_class=num_class,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("Checkpoint/model mapping mismatch; missing={} unexpected={}".format(missing, unexpected))
    model.float()
    model.freeze_text_encoder()
    model.eval()
    return model


def tokenize_captions(captions):
    from models.clip_model import tokenize
    from models.utils.simple_tokenizer import SimpleTokenizer
    tokens = tokenize(list(captions), SimpleTokenizer(), context_length=77, truncate=True)
    return tokens.long()


class DiagnosisDataset(object):
    def __init__(self, records, caption_path):
        from PIL import Image
        from data.img_transforms import Compose, Resize, ToTensor, Normalize
        from tools.rchrl_common import image_io_path
        self.Image = Image
        self.image_io_path = image_io_path
        self.records = records
        self.transform = Compose([
            Resize((384, 128)),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])
        with open(caption_path) as handle:
            values = json.load(handle)
        self.captions = {canonical(k): v[0] for k, v in values.items()}
        for row in self.records:
            key = canonical(row["caption_key"])
            if key not in self.captions:
                raise RuntimeError("Caption key missing: {}".format(key))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with self.Image.open(self.image_io_path(row["path"])) as image:
            image = image.convert("RGB")
        return self.transform(image), int(index), self.captions[canonical(row["caption_key"])]


def extract_seed(seed, device_name="cuda:0", batch_size=64, workers=2, force=False):
    import torch
    from torch.utils.data import DataLoader
    ensure_dirs()
    manifest = read_json(os.path.join(REPORT_ROOT, "baseline_checkpoint_manifest.json"))
    entries = {int(x["seed"]): x for x in manifest["checkpoints"]}
    if seed not in entries:
        raise RuntimeError("Seed {} missing from baseline manifest".format(seed))
    records = load_prepared_records()
    n = len(records)
    out = seed_root(seed)
    layer_raw_path = os.path.join(out, "layers", "layers_raw.npy")
    layer_l2_path = os.path.join(out, "layers", "layers_l2.npy")
    fvisual_path = os.path.join(out, "decomposition", "F_visual_raw.npy")
    fcloth_path = os.path.join(out, "decomposition", "F_cloth_raw.npy")
    done_path = os.path.join(out, "extraction_manifest.json")
    if not force and os.path.isfile(done_path):
        done = read_json(done_path)
        if done.get("checkpoint_sha256") == entries[seed]["checkpoint_sha256"] and all(os.path.isfile(p) for p in [layer_raw_path, layer_l2_path, fvisual_path, fcloth_path]):
            print("seed{} extraction already complete; use --force to repeat".format(seed))
            return done

    if not torch.cuda.is_available() and str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA/PPU requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model = build_model_from_checkpoint(entries[seed]).to(device)
    model.eval()
    dataset = DiagnosisDataset(records, CAPTION_PATH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=False, drop_last=False)

    layers = model.visual.transformer.resblocks
    capture = {"pre": None, "blocks": [None] * len(layers)}

    def pre_hook(_module, inputs):
        capture["pre"] = inputs[0].detach()

    def block_hook(index):
        def _hook(_module, _inputs, output):
            capture["blocks"][index] = output.detach()
        return _hook

    handles = [model.visual.transformer.register_forward_pre_hook(pre_hook)]
    handles.extend(block.register_forward_hook(block_hook(i)) for i, block in enumerate(layers))
    raw_layers = np.lib.format.open_memmap(layer_raw_path, mode="w+", dtype=np.float32,
                                           shape=(n, len(layers) + 1, int(model.visual.proj.shape[1])))
    l2_layers = np.lib.format.open_memmap(layer_l2_path, mode="w+", dtype=np.float32,
                                          shape=raw_layers.shape)
    fvisual = np.lib.format.open_memmap(fvisual_path, mode="w+", dtype=np.float32,
                                        shape=(n, int(model.in_planes_proj)))
    fcloth = np.lib.format.open_memmap(fcloth_path, mode="w+", dtype=np.float32,
                                       shape=(n, int(model.in_planes_proj)))
    fvisual_pre_bn_path = os.path.join(out, "decomposition", "F_visual_pre_bn_raw.npy")
    fvisual_pre_bn = np.lib.format.open_memmap(fvisual_pre_bn_path, mode="w+", dtype=np.float32,
                                               shape=(n, int(model.visual.proj.shape[1])))
    first_batch = True
    max_z12_error = 0.0
    max_visual_error = 0.0
    start_time = time.time()
    try:
        with torch.no_grad():
            for images, indices, captions in loader:
                indices_np = indices.numpy().astype(np.int64)
                images = images.to(device, non_blocking=False)
                tokens = tokenize_captions(captions).to(device)
                visual_output = model.visual(images.type(model.dtype))
                x, _z = visual_output
                pre = capture["pre"]
                blocks = capture["blocks"]
                if pre is None or any(value is None for value in blocks):
                    raise RuntimeError("Visual transformer hooks did not capture every layer")
                hidden_cls = [pre[0]] + [value[0] for value in blocks]
                hidden = torch.stack(hidden_cls, dim=1)  # [B, 13, 768]
                hidden_flat = hidden.reshape(-1, hidden.shape[-1])
                projected = model.visual.proj
                z_all = model.visual.ln_post(hidden_flat).matmul(projected).reshape(hidden.shape[0], hidden.shape[1], -1)
                z12 = x[:, 0, :].float()
                max_z12_error = max(max_z12_error, float((z_all[:, -1, :] - z12).abs().max().item()))
                f_visual_batch = model.bottleneck_proj(z12)
                t = model.encode_text_irra(tokens)
                combined = model.combine(x, t)
                eot = tokens.argmax(dim=-1)
                c_raw = combined[torch.arange(combined.shape[0], device=device), eot]
                f_cloth_batch = model.bottleneck_proj(c_raw)
                if first_batch:
                    expected_eval = model(images)
                    max_visual_error = float((expected_eval - f_visual_batch).abs().max().item())
                    if max_z12_error > 5e-4 or max_visual_error > 5e-4:
                        raise RuntimeError("Feature mapping sanity failed: z12_error={} visual_error={}".format(max_z12_error, max_visual_error))
                    first_batch = False
                z_np = z_all.float().cpu().numpy()
                fv_np = f_visual_batch.float().cpu().numpy()
                fc_np = f_cloth_batch.float().cpu().numpy()
                raw_layers[indices_np] = z_np
                l2_layers[indices_np] = l2_normalize(z_np, axis=-1)
                fvisual[indices_np] = fv_np
                fcloth[indices_np] = fc_np
                fvisual_pre_bn[indices_np] = z12.float().cpu().numpy()
                if (int(indices_np[-1]) + 1) % (batch_size * 10) < batch_size:
                    print("seed{} extracted {}/{} images".format(seed, int(indices_np[-1]) + 1, n), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        raw_layers.flush()
        l2_layers.flush()
        fvisual.flush()
        fcloth.flush()
        fvisual_pre_bn.flush()

    # Save normalized decomposition features after the raw arrays are closed.
    for source_name in ("F_visual", "F_cloth", "F_visual_pre_bn"):
        source_path = os.path.join(out, "decomposition", source_name + "_raw.npy")
        normalized_path = os.path.join(out, "decomposition", source_name + "_l2.npy")
        source = np.load(source_path, mmap_mode="r")
        # These decomposition arrays are only 36 MB each.  A regular atomic
        # np.save is less fragile on the object-backed filesystem than a tiny
        # partially-created memmap header if a worker/device exits at teardown.
        normalized = l2_normalize(np.asarray(source))
        np.save(normalized_path, normalized.astype(np.float32))
        del normalized

    extraction = {
        "seed": seed,
        "checkpoint_path": entries[seed]["checkpoint_path"],
        "checkpoint_sha256": entries[seed]["checkpoint_sha256"],
        "count": n,
        "layer_names": ["Z_{}".format(i) for i in range(len(layers) + 1)],
        "layer_count": len(layers) + 1,
        "layer_dimension": int(model.visual.proj.shape[1]),
        "feature_dimension": int(model.in_planes_proj),
        "raw_paths": {
            "layers": layer_raw_path,
            "F_visual": fvisual_path,
            "F_cloth": fcloth_path,
            "F_visual_pre_bn": fvisual_pre_bn_path,
        },
        "l2_paths": {
            "layers": layer_l2_path,
            "F_visual": os.path.join(out, "decomposition", "F_visual_l2.npy"),
            "F_cloth": os.path.join(out, "decomposition", "F_cloth_l2.npy"),
            "F_visual_pre_bn": os.path.join(out, "decomposition", "F_visual_pre_bn_l2.npy"),
        },
        "transform": "Resize(384,128), ToTensor, CLIP Normalize; deterministic eval transform; no flip",
        "caption_path": CAPTION_PATH,
        "caption_sha256": sha256_file(CAPTION_PATH),
        "batch_size": batch_size,
        "num_workers": workers,
        "device": str(device),
        "model_mode": "eval; frozen text encoder; no gradients",
        "cloth_route": "bottleneck_proj(EOT(combine(x, encode_text_irra(caption))))",
        "visual_route": "bottleneck_proj(Z_12)",
        "residual_route": "offline normalize(F_visual - alpha * F_cloth) in BN-output 512-D space",
        "hook_definition": "Z_0 is pre-transformer CLS and Z_l (l>=1) is CLS after visual block l, each passed through current ln_post and visual.proj for common 512-D audit space",
        "sanity": {
            "max_z12_vs_visual_output_error": max_z12_error,
            "max_fvisual_vs_model_eval_image_only_error": max_visual_error,
        },
        "elapsed_seconds": time.time() - start_time,
        "file_sha256": {},
        "diagnosis_script_sha256": sha256_file(os.path.abspath(__file__)),
    }
    for path in list(extraction["raw_paths"].values()) + list(extraction["l2_paths"].values()):
        extraction["file_sha256"][path] = sha256_file(path)
    write_json(extraction, done_path)
    print(json.dumps({"status": "extracted", "seed": seed, "count": n, "elapsed_seconds": extraction["elapsed_seconds"], "sanity": extraction["sanity"]}, indent=2))
    return extraction


def load_seed_arrays(seed):
    out = seed_root(seed)
    manifest_path = os.path.join(out, "extraction_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Seed{} extraction manifest is missing".format(seed))
    manifest = read_json(manifest_path)
    raw_layers = np.load(manifest["raw_paths"]["layers"], mmap_mode="r")
    l2_layers = np.load(manifest["l2_paths"]["layers"], mmap_mode="r")
    fvisual = np.load(manifest["raw_paths"]["F_visual"], mmap_mode="r")
    fvisual_l2 = np.load(manifest["l2_paths"]["F_visual"], mmap_mode="r")
    fcloth = np.load(manifest["raw_paths"]["F_cloth"], mmap_mode="r")
    fcloth_l2 = np.load(manifest["l2_paths"]["F_cloth"], mmap_mode="r")
    return manifest, raw_layers, l2_layers, fvisual, fvisual_l2, fcloth, fcloth_l2


def verify_extractions():
    """Validate every saved feature file before any H1/H2 calculation."""
    records = load_prepared_records()
    expected_n = len(records)
    expected_script = sha256_file(os.path.abspath(__file__))
    rows = []
    for seed in SEEDS:
        manifest_path = os.path.join(seed_root(seed), "extraction_manifest.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError("seed{} extraction manifest is missing".format(seed))
        manifest = read_json(manifest_path)
        baseline = read_json(os.path.join(REPORT_ROOT, "baseline_checkpoint_manifest.json"))
        expected_checkpoint = next(x for x in baseline["checkpoints"] if int(x["seed"]) == seed)
        if manifest.get("checkpoint_sha256") != expected_checkpoint["checkpoint_sha256"]:
            raise RuntimeError("seed{} feature/checkpoint SHA mismatch".format(seed))
        raw_layers = np.load(manifest["raw_paths"]["layers"], mmap_mode="r")
        l2_layers = np.load(manifest["l2_paths"]["layers"], mmap_mode="r")
        shapes = {
            "layers_raw": list(raw_layers.shape),
            "layers_l2": list(l2_layers.shape),
            "F_visual_raw": list(np.load(manifest["raw_paths"]["F_visual"], mmap_mode="r").shape),
            "F_visual_l2": list(np.load(manifest["l2_paths"]["F_visual"], mmap_mode="r").shape),
            "F_cloth_raw": list(np.load(manifest["raw_paths"]["F_cloth"], mmap_mode="r").shape),
            "F_cloth_l2": list(np.load(manifest["l2_paths"]["F_cloth"], mmap_mode="r").shape),
            "F_visual_pre_bn_raw": list(np.load(manifest["raw_paths"]["F_visual_pre_bn"], mmap_mode="r").shape),
            "F_visual_pre_bn_l2": list(np.load(manifest["l2_paths"]["F_visual_pre_bn"], mmap_mode="r").shape),
        }
        expected_shapes = {
            "layers_raw": [expected_n, 13, 512], "layers_l2": [expected_n, 13, 512],
            "F_visual_raw": [expected_n, 512], "F_visual_l2": [expected_n, 512],
            "F_cloth_raw": [expected_n, 512], "F_cloth_l2": [expected_n, 512],
            "F_visual_pre_bn_raw": [expected_n, 512], "F_visual_pre_bn_l2": [expected_n, 512],
        }
        if shapes != expected_shapes:
            raise RuntimeError("seed{} feature shape mismatch: {}".format(seed, shapes))
        finite_checks = {
            "layers_raw": bool(np.isfinite(raw_layers[::max(1, expected_n // 32)]).all()),
            "layers_l2": bool(np.isfinite(l2_layers[::max(1, expected_n // 32)]).all()),
        }
        if not all(finite_checks.values()):
            raise RuntimeError("seed{} has non-finite feature sample".format(seed))
        manifest["verified_against_diagnosis_script_sha256"] = expected_script
        manifest["text_transformer_heads"] = 8
        manifest["feature_file_shape_verification"] = shapes
        manifest["verification_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(manifest, manifest_path)
        rows.append({"seed": seed, "checkpoint_sha256": manifest["checkpoint_sha256"], "count": expected_n,
                     "layers_shape": shapes["layers_raw"], "feature_dimension": 512,
                     "finite_sample_check": finite_checks, "sanity": manifest["sanity"],
                     "manifest_path": manifest_path})
    result = {"status": "verified", "seed_rows": rows, "script_sha256": expected_script,
              "no_test_used": True}
    write_json(result, os.path.join(REPORT_ROOT, "extraction_verification.json"))
    print(json.dumps(result, indent=2))
    return result


def metric_auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_pos = ranks[:len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def metric_ap(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    scores = np.concatenate([pos, neg])
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    return float((cumulative[hits == 1] / ranks[hits == 1]).sum() / max(1, len(pos)))


def metric_eer(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = len(pos) - tp
    tn = len(neg) - fp
    fpr = fp / max(1, len(neg))
    fnr = fn / max(1, len(pos))
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def verification_metrics(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    return {
        "positive_count": int(len(pos)),
        "negative_count": int(len(neg)),
        "positive_mean": float(np.mean(pos)),
        "negative_mean": float(np.mean(neg)),
        "D_gap": float(np.mean(pos) - np.mean(neg)),
        "ROC_AUC": metric_auc(pos, neg),
        "PR_AUC": metric_ap(pos, neg),
        "EER": metric_eer(pos, neg),
    }


def pair_scores(features_l2, pairs):
    pairs = np.asarray(pairs)
    return np.sum(features_l2[pairs[:, 0]] * features_l2[pairs[:, 1]], axis=1, dtype=np.float64)


def ridge_fit_predict(x_train, y_train, x_eval, alpha=RIDGE_ALPHA):
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xs = (x_train - mean) / scale
    xe = (x_eval - mean) / scale
    yc = y_train.mean(axis=0)
    y0 = y_train - yc
    gram = xs.T.dot(xs)
    gram.flat[::gram.shape[0] + 1] += alpha
    rhs = xs.T.dot(y0)
    try:
        weight = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return (xe.dot(weight) + yc).astype(np.float32)


def regression_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    target_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    mse = np.mean((pred - target) ** 2)
    denom = np.sum((target - target.mean(axis=0)) ** 2)
    return {
        "cosine": float(np.mean(np.sum(pred_norm * target_norm, axis=1))),
        "R2": float(1.0 - np.sum((pred - target) ** 2) / max(1e-12, denom)),
        "NMSE": float(mse / max(1e-12, np.mean(target ** 2))),
    }


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T.dot(y)
    numerator = np.sum(cross * cross)
    denominator = math.sqrt(np.sum((x.T.dot(x)) ** 2) * np.sum((y.T.dot(y)) ** 2))
    # Z_0 is the pre-transformer CLS state and is constant across images in
    # this architecture, so centered linear CKA has no defined denominator.
    # Preserve that as an explicit missing value rather than a numeric NaN.
    return float(numerator / denominator) if denominator > 1e-12 else None


def direct_alignment(x, c):
    xn = l2_normalize(x)
    cn = l2_normalize(c)
    cosine = np.sum(xn * cn, axis=1)
    projection_energy = cosine * cosine
    return {
        "mean_cosine": float(np.mean(cosine)),
        "mean_absolute_cosine": float(np.mean(np.abs(cosine))),
        "projection_energy": float(np.mean(projection_energy)),
        "linear_CKA": linear_cka(x, c),
    }


def bootstrap_ci(values_by_identity, seed, iterations=BOOTSTRAP_ITERATIONS):
    values = np.asarray(values_by_identity, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_identity": 0, "iterations": 0}
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(values), size=(iterations, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_identity": int(len(values)),
        "iterations": int(iterations),
    }


def per_identity_gap(pos_scores, neg_scores, pos_pairs, neg_pairs, person_ids, n_ids):
    pos_sum = np.zeros(n_ids, dtype=np.float64)
    pos_count = np.zeros(n_ids, dtype=np.int64)
    for score, pair in zip(pos_scores, pos_pairs):
        p = int(person_ids[pair[0]])
        pos_sum[p] += score
        pos_count[p] += 1
    neg_sum = np.zeros(n_ids, dtype=np.float64)
    neg_count = np.zeros(n_ids, dtype=np.int64)
    for score, pair in zip(neg_scores, neg_pairs):
        for p in (int(person_ids[pair[0]]), int(person_ids[pair[1]])):
            neg_sum[p] += score
            neg_count[p] += 1
    values = (pos_sum / np.maximum(pos_count, 1)) - (neg_sum / np.maximum(neg_count, 1))
    valid = (pos_count > 0) & (neg_count > 0)
    return values[valid], np.flatnonzero(valid)


def per_identity_auc(pos_scores, neg_scores, pos_pairs, neg_pairs, person_ids, n_ids):
    pos_by_id = [[] for _ in range(n_ids)]
    neg_by_id = [[] for _ in range(n_ids)]
    for score, pair in zip(pos_scores, pos_pairs):
        pos_by_id[int(person_ids[pair[0]])].append(float(score))
    for score, pair in zip(neg_scores, neg_pairs):
        a = int(person_ids[pair[0]])
        b = int(person_ids[pair[1]])
        neg_by_id[a].append(float(score))
        neg_by_id[b].append(float(score))
    values = []
    ids = []
    for identity in range(n_ids):
        if pos_by_id[identity] and neg_by_id[identity]:
            values.append(metric_auc(np.asarray(pos_by_id[identity]), np.asarray(neg_by_id[identity])))
            ids.append(identity)
    return np.asarray(values, dtype=np.float64), np.asarray(ids, dtype=np.int32)


def retrieval_metrics(features_l2, gallery_indices, query_indices, person_ids, device_name="cpu", batch_queries=256):
    """Compute A-gallery/B-or-C-query retrieval with deterministic ordering."""
    gallery_indices = np.asarray(gallery_indices, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    gallery_features = np.asarray(features_l2[gallery_indices], dtype=np.float32)
    query_features = np.asarray(features_l2[query_indices], dtype=np.float32)
    gallery_pid = np.asarray(person_ids[gallery_indices])
    query_pid = np.asarray(person_ids[query_indices])
    r1, r5, r10, aps = [], [], [], []
    use_torch = False
    torch = None
    device = None
    if str(device_name).startswith("cuda"):
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                torch = _torch
                device = torch.device(device_name)
                use_torch = True
        except Exception:
            use_torch = False
    if use_torch:
        g = torch.from_numpy(gallery_features).to(device)
    for start in range(0, len(query_indices), batch_queries):
        end = min(len(query_indices), start + batch_queries)
        if use_torch:
            sim = (torch.from_numpy(query_features[start:end]).to(device).matmul(g.T)).detach().cpu().numpy()
        else:
            sim = query_features[start:end].dot(gallery_features.T)
        order = np.argsort(-sim, axis=1, kind="mergesort")
        for row_no, ranking in enumerate(order):
            relevant = gallery_pid[ranking] == query_pid[start + row_no]
            r1.append(float(relevant[:1].any()))
            r5.append(float(relevant[:min(5, len(relevant))].any()))
            r10.append(float(relevant[:min(10, len(relevant))].any()))
            total = int(relevant.sum())
            if total:
                hit_positions = np.flatnonzero(relevant) + 1
                aps.append(float(np.mean(np.arange(1, total + 1, dtype=np.float64) / hit_positions)))
            else:
                aps.append(0.0)
    return {
        "query_count": int(len(query_indices)),
        "gallery_count": int(len(gallery_indices)),
        "R1": float(np.mean(r1)),
        "R5": float(np.mean(r5)),
        "R10": float(np.mean(r10)),
        "mAP": float(np.mean(aps)),
        "query_r1": np.asarray(r1, dtype=np.float32),
        "query_ap": np.asarray(aps, dtype=np.float32),
        "query_person_id": query_pid.astype(np.int32),
    }


def prototype_retention(features_l2, records, n_ids):
    by_id_camera = defaultdict(list)
    for idx, row in enumerate(records):
        by_id_camera[(int(row["person_id"]), row["camera"])].append(idx)
    prototypes = {}
    for identity in range(n_ids):
        for camera in ("A", "B", "C"):
            indices = by_id_camera.get((identity, camera), [])
            if indices:
                values = features_l2[indices]
                prototypes[(identity, camera)] = l2_normalize(np.mean(values, axis=0, keepdims=True))[0]
    positive = []
    negative = []
    retention_by_id = []
    valid_identities = [identity for identity in range(n_ids)
                        if (identity, "A") in prototypes and (identity, "C") in prototypes]
    for identity in valid_identities:
        # PRCC TRAIN has one identity without a B image.  Keep the available
        # A-C protocol for that identity and only add B-C when both the
        # positive and matched negative prototypes exist.
        protocols = [("A", "C")]
        if (identity, "B") in prototypes:
            protocols.append(("B", "C"))
        other = next((candidate for offset in range(1, n_ids + 1)
                      for candidate in [(identity + offset) % n_ids]
                      if (candidate, "C") in prototypes and
                      all((candidate, source) in prototypes for source, _ in protocols)), identity)
        pos_values = [float(np.dot(prototypes[(identity, source)], prototypes[(identity, target)]))
                      for source, target in protocols]
        neg_values = [float(np.dot(prototypes[(identity, source)], prototypes[(other, target)]))
                      for source, target in protocols]
        positive.extend(pos_values)
        negative.extend(neg_values)
        retention_by_id.append(np.mean(pos_values) - np.mean(neg_values))
    return {
        "positive": np.asarray(positive, dtype=np.float64),
        "negative": np.asarray(negative, dtype=np.float64),
        "retention_by_identity": np.asarray(retention_by_id, dtype=np.float64),
    }


def analysis_for_seed(seed, device_name="cuda:0"):
    records = load_prepared_records()
    person_ids = np.asarray([int(row["person_id"]) for row in records], dtype=np.int32)
    n_ids = int(person_ids.max() + 1)
    pair_manifest = read_json(os.path.join(OUT_ROOT, "shared", "pair_manifest", "pair_manifest.json"))
    pairs = {kind: np.load(path) for kind, path in pair_manifest["positive_paths"].items()}
    negatives = np.load(pair_manifest["negative_pairs_path"])
    split_dir = os.path.join(OUT_ROOT, "shared", "split_manifest")
    dev_indices = np.load(os.path.join(split_dir, "semantic_dev_indices.npy"))
    val_indices = np.load(os.path.join(split_dir, "semantic_val_indices.npy"))
    _manifest, raw_layers, l2_layers, fvisual_raw, fvisual_l2, fcloth_raw, fcloth_l2 = load_seed_arrays(seed)
    if len(raw_layers) != len(records):
        raise RuntimeError("Seed{} features do not align with prepared inventory".format(seed))
    results_dir = os.path.join(seed_root(seed), "alpha_sweep")
    os.makedirs(results_dir, exist_ok=True)
    h1a_rows, h1b_rows, identity_rows = [], [], []
    for layer in range(raw_layers.shape[1]):
        x_train = np.asarray(raw_layers[dev_indices, layer], dtype=np.float32)
        x_val = np.asarray(raw_layers[val_indices, layer], dtype=np.float32)
        y_train = np.asarray(fcloth_raw[dev_indices], dtype=np.float32)
        y_val = np.asarray(fcloth_raw[val_indices], dtype=np.float32)
        pred_train = ridge_fit_predict(x_train, y_train, x_train)
        pred_val = ridge_fit_predict(x_train, y_train, x_val)
        m_train = regression_metrics(pred_train, y_train)
        m_val = regression_metrics(pred_val, y_val)
        for scope, metric in (("dev", m_train), ("val", m_val)):
            h1a_rows.append({"seed": seed, "scope": scope, "layer": layer, **metric, "ridge_alpha": RIDGE_ALPHA})
        for scope, indices in (("dev", dev_indices), ("val", val_indices)):
            h1b_rows.append({"seed": seed, "scope": scope, "layer": layer,
                             **direct_alignment(np.asarray(raw_layers[indices, layer]), np.asarray(fcloth_raw[indices]))})

        pos = np.concatenate([pair_scores(l2_layers[:, layer], pairs["diff_ac"]), pair_scores(l2_layers[:, layer], pairs["diff_bc"])])
        neg = pair_scores(l2_layers[:, layer], negatives)
        identity_rows.append({"seed": seed, "layer": layer, **verification_metrics(pos, neg)})

    write_csv(h1a_rows, os.path.join(seed_root(seed), "h1a_layer_predictability.csv"),
              ["seed", "scope", "layer", "cosine", "R2", "NMSE", "ridge_alpha"])
    write_csv(h1b_rows, os.path.join(seed_root(seed), "h1b_direct_alignment.csv"),
              ["seed", "scope", "layer", "mean_cosine", "mean_absolute_cosine", "projection_energy", "linear_CKA"])
    write_csv(identity_rows, os.path.join(seed_root(seed), "layer_identity_discrimination.csv"),
              ["seed", "layer", "positive_count", "negative_count", "positive_mean", "negative_mean", "D_gap", "ROC_AUC", "PR_AUC", "EER"])

    rep_rows = []
    rep_features = {
        "F_visual": np.asarray(fvisual_l2),
        "F_cloth": np.asarray(fcloth_l2),
    }
    retrieval_rows = []
    retrieval_payload = {}
    retrieval_indices = np.load(os.path.join(OUT_ROOT, "shared", "pair_manifest", "retrieval_indices.npz"))
    gallery_a = retrieval_indices["gallery_A"]
    query_b = retrieval_indices["query_B"]
    query_c = retrieval_indices["query_C"]
    for name, feature in list(rep_features.items()):
        same = pair_scores(feature, pairs["same_ab"])
        diff_ac = pair_scores(feature, pairs["diff_ac"])
        diff_bc = pair_scores(feature, pairs["diff_bc"])
        diff = np.concatenate([diff_ac, diff_bc])
        neg = pair_scores(feature, negatives)
        for kind, positive in (("same_clothes_A_B", same), ("different_clothes_A_C_B_C", diff),
                               ("different_clothes_A_C", diff_ac), ("different_clothes_B_C", diff_bc)):
            rep_rows.append({"seed": seed, "representation": name, "metric_scope": kind, **verification_metrics(positive, neg)})
        for query_name, query_indices in (("A_gallery_B_query", query_b), ("A_gallery_C_query", query_c)):
            retrieved = retrieval_metrics(feature, gallery_a, query_indices, person_ids, device_name=device_name)
            key = name + "__" + query_name
            retrieval_payload[key] = retrieved
            retrieval_rows.append({"seed": seed, "representation": name, "query_protocol": query_name,
                                   "R1": retrieved["R1"], "R5": retrieved["R5"], "R10": retrieved["R10"],
                                   "mAP": retrieved["mAP"], "query_count": retrieved["query_count"], "gallery_count": retrieved["gallery_count"]})

    alpha_rows = []
    alpha_retrieval_rows = []
    alpha_query_payload = {}
    alpha_features = []
    fvisual = np.asarray(fvisual_raw, dtype=np.float32)
    fcloth = np.asarray(fcloth_raw, dtype=np.float32)
    for alpha in ALPHAS:
        residual_raw = fvisual - float(alpha) * fcloth
        residual_l2 = l2_normalize(residual_raw)
        tag = alpha_tag(alpha)
        np.save(os.path.join(results_dir, "residual_{}_raw.npy".format(tag)), residual_raw.astype(np.float32))
        np.save(os.path.join(results_dir, "residual_{}_l2.npy".format(tag)), residual_l2.astype(np.float32))
        same = pair_scores(residual_l2, pairs["same_ab"])
        diff_ac = pair_scores(residual_l2, pairs["diff_ac"])
        diff_bc = pair_scores(residual_l2, pairs["diff_bc"])
        diff = np.concatenate([diff_ac, diff_bc])
        neg = pair_scores(residual_l2, negatives)
        alpha_rows.append({"seed": seed, "alpha": float(alpha), **verification_metrics(diff, neg),
                           "same_clothes_positive_mean": float(same.mean()),
                           "diff_clothes_positive_mean": float(diff.mean()),
                           "same_clothes_D_gap": float(same.mean() - neg.mean())})
        retrieved = retrieval_metrics(residual_l2, gallery_a, query_c, person_ids, device_name=device_name)
        alpha_retrieval_rows.append({"seed": seed, "alpha": float(alpha), "R1": retrieved["R1"], "R5": retrieved["R5"],
                                     "R10": retrieved["R10"], "mAP": retrieved["mAP"],
                                     "query_count": retrieved["query_count"], "gallery_count": retrieved["gallery_count"]})
        alpha_query_payload[tag] = retrieved
        alpha_features.append((float(alpha), residual_l2))

    write_csv(rep_rows, os.path.join(seed_root(seed), "representation_verification.csv"),
              ["seed", "representation", "metric_scope", "positive_count", "negative_count", "positive_mean", "negative_mean", "D_gap", "ROC_AUC", "PR_AUC", "EER"])
    write_csv(retrieval_rows, os.path.join(seed_root(seed), "retrieval_metrics.csv"),
              ["seed", "representation", "query_protocol", "R1", "R5", "R10", "mAP", "query_count", "gallery_count"])
    write_csv(alpha_rows, os.path.join(seed_root(seed), "alpha_sweep_verification.csv"),
              ["seed", "alpha", "positive_count", "negative_count", "positive_mean", "negative_mean", "D_gap", "ROC_AUC", "PR_AUC", "EER", "same_clothes_positive_mean", "diff_clothes_positive_mean", "same_clothes_D_gap"])
    write_csv(alpha_retrieval_rows, os.path.join(seed_root(seed), "alpha_sweep_retrieval.csv"),
              ["seed", "alpha", "R1", "R5", "R10", "mAP", "query_count", "gallery_count"])

    # Identity-level bootstrap inputs for the core H2 metrics.
    cloth_pos = np.concatenate([pair_scores(np.asarray(fcloth_l2), pairs["diff_ac"]), pair_scores(np.asarray(fcloth_l2), pairs["diff_bc"])])
    cloth_pos_pairs = np.concatenate([pairs["diff_ac"], pairs["diff_bc"]])
    cloth_neg = pair_scores(np.asarray(fcloth_l2), negatives)
    gap_values, gap_ids = per_identity_gap(cloth_pos, cloth_neg, cloth_pos_pairs, negatives, person_ids, n_ids)
    auc_values, auc_ids = per_identity_auc(cloth_pos, cloth_neg, cloth_pos_pairs, negatives, person_ids, n_ids)
    prototype = prototype_retention(np.asarray(fcloth_l2), records, n_ids)
    # Pick the best alpha using A/C R1 first and mAP as tie-break; no val/test
    # selection is involved because this is a TRAIN diagnostic sweep.
    alpha_by_value = {round(float(row["alpha"]), 2): row for row in alpha_retrieval_rows}
    best = max(alpha_retrieval_rows, key=lambda row: (row["R1"], row["mAP"], -abs(float(row["alpha"]) - 0.5)))
    one = alpha_by_value[1.0]
    best_tag = alpha_tag(best["alpha"])
    one_tag = alpha_tag(1.0)
    best_query = alpha_query_payload[best_tag]
    one_query = alpha_query_payload[one_tag]
    query_delta = best_query["query_ap"] - one_query["query_ap"]
    r1_delta = best_query["query_r1"] - one_query["query_r1"]
    by_pid_delta = []
    by_pid_r1_delta = []
    for identity in range(n_ids):
        mask = best_query["query_person_id"] == identity
        if np.any(mask):
            by_pid_delta.append(float(np.mean(query_delta[mask])))
            by_pid_r1_delta.append(float(np.mean(r1_delta[mask])))
    bootstrap = [
        {"seed": seed, "metric": "F_cloth_diff_clothes_D_gap", "unit": "identity", **bootstrap_ci(gap_values, 91000 + seed)},
        {"seed": seed, "metric": "F_cloth_diff_clothes_ROC_AUC", "unit": "identity", **bootstrap_ci(auc_values, 92000 + seed)},
        {"seed": seed, "metric": "alpha_best_minus_1_mAP", "unit": "identity", **bootstrap_ci(by_pid_delta, 93000 + seed)},
        {"seed": seed, "metric": "alpha_best_minus_1_R1", "unit": "identity", **bootstrap_ci(by_pid_r1_delta, 94000 + seed)},
        {"seed": seed, "metric": "F_cloth_identity_retention_prototype", "unit": "identity", **bootstrap_ci(prototype["retention_by_identity"], 95000 + seed)},
    ]
    dist_rng = np.random.RandomState(96000 + seed)
    dist_payload = {
        "F_visual_diff": pair_scores(np.asarray(fvisual_l2), np.concatenate([pairs["diff_ac"], pairs["diff_bc"]])),
        "F_visual_neg": pair_scores(np.asarray(fvisual_l2), negatives),
        "F_cloth_diff": cloth_pos,
        "F_cloth_neg": cloth_neg,
        "residual_best_diff": pair_scores(alpha_features[[a for a, _ in alpha_features].index(float(best["alpha"]))][1], cloth_pos_pairs),
        "residual_best_neg": pair_scores(alpha_features[[a for a, _ in alpha_features].index(float(best["alpha"]))][1], negatives),
    }
    for key, values in list(dist_payload.items()):
        if len(values) > 20000:
            dist_payload[key] = values[dist_rng.choice(len(values), 20000, replace=False)]
    np.savez(os.path.join(seed_root(seed), "distribution_samples.npz"), **dist_payload)
    bootstrap_path = os.path.join(seed_root(seed), "bootstrap_summary.json")
    write_json({
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "seed": seed,
        "rows": bootstrap,
        "best_alpha_by_train_A_C_R1_then_mAP": best,
        "alpha_1_row": one,
        "best_minus_1_mAP": float(best["mAP"] - one["mAP"]),
        "best_minus_1_R1": float(best["R1"] - one["R1"]),
    }, bootstrap_path)
    summary = {
        "seed": seed,
        "h1a_path": os.path.join(seed_root(seed), "h1a_layer_predictability.csv"),
        "h1b_path": os.path.join(seed_root(seed), "h1b_direct_alignment.csv"),
        "identity_path": os.path.join(seed_root(seed), "layer_identity_discrimination.csv"),
        "representation_path": os.path.join(seed_root(seed), "representation_verification.csv"),
        "retrieval_path": os.path.join(seed_root(seed), "retrieval_metrics.csv"),
        "alpha_verification_path": os.path.join(seed_root(seed), "alpha_sweep_verification.csv"),
        "alpha_retrieval_path": os.path.join(seed_root(seed), "alpha_sweep_retrieval.csv"),
        "bootstrap_path": bootstrap_path,
        "cloth_cross_clothes_verification": next(row for row in rep_rows if row["metric_scope"] == "different_clothes_A_C_B_C" and row["representation"] == "F_cloth"),
        "cloth_retrieval_A_C": next(row for row in retrieval_rows if row["representation"] == "F_cloth" and row["query_protocol"] == "A_gallery_C_query"),
        "visual_retrieval_A_C": next(row for row in retrieval_rows if row["representation"] == "F_visual" and row["query_protocol"] == "A_gallery_C_query"),
        "prototype_verification": verification_metrics(prototype["positive"], prototype["negative"]),
        "prototype_identity_retention": float(np.mean(prototype["retention_by_identity"])),
        "best_alpha": float(best["alpha"]),
        "best_alpha_R1": float(best["R1"]),
        "best_alpha_mAP": float(best["mAP"]),
        "alpha_1_R1": float(one["R1"]),
        "alpha_1_mAP": float(one["mAP"]),
        "alpha_best_minus_1_R1": float(best["R1"] - one["R1"]),
        "alpha_best_minus_1_mAP": float(best["mAP"] - one["mAP"]),
    }
    write_json(summary, os.path.join(seed_root(seed), "analysis_summary.json"))
    print(json.dumps({"status": "analyzed", "seed": seed, "cloth_auc": summary["cloth_cross_clothes_verification"]["ROC_AUC"], "cloth_A_C_R1": summary["cloth_retrieval_A_C"]["R1"], "best_alpha": summary["best_alpha"], "best_minus_1_R1": summary["alpha_best_minus_1_R1"], "best_minus_1_mAP": summary["alpha_best_minus_1_mAP"]}, indent=2))
    return summary


def read_csv(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("seed", "layer", "positive_count", "negative_count", "query_count", "gallery_count", "n_identity", "iterations"):
                try:
                    row[key] = int(value)
                except (TypeError, ValueError):
                    pass
            elif key not in ("scope", "representation", "metric_scope", "query_protocol"):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def aggregate_rows(rows, group_keys, value_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        out = {name: value for name, value in zip(group_keys, key)}
        for value_key in value_keys:
            numbers = np.asarray([float(row[value_key]) for row in values], dtype=np.float64)
            out[value_key + "_mean"] = float(np.mean(numbers))
            out[value_key + "_std"] = float(np.std(numbers))
        result.append(out)
    return result


def aggregate_analysis():
    h1a = sum((read_csv(os.path.join(seed_root(seed), "h1a_layer_predictability.csv")) for seed in SEEDS), [])
    h1b = sum((read_csv(os.path.join(seed_root(seed), "h1b_direct_alignment.csv")) for seed in SEEDS), [])
    identity = sum((read_csv(os.path.join(seed_root(seed), "layer_identity_discrimination.csv")) for seed in SEEDS), [])
    representation = sum((read_csv(os.path.join(seed_root(seed), "representation_verification.csv")) for seed in SEEDS), [])
    retrieval = sum((read_csv(os.path.join(seed_root(seed), "retrieval_metrics.csv")) for seed in SEEDS), [])
    alpha_v = sum((read_csv(os.path.join(seed_root(seed), "alpha_sweep_verification.csv")) for seed in SEEDS), [])
    alpha_r = sum((read_csv(os.path.join(seed_root(seed), "alpha_sweep_retrieval.csv")) for seed in SEEDS), [])
    val_h1a = [row for row in h1a if row["scope"] == "val"]
    table1 = []
    for layer in range(13):
        values = [row for row in val_h1a if row["layer"] == layer]
        by_seed = {row["seed"]: row for row in values}
        out = {"layer": layer}
        for seed in SEEDS:
            out["seed{}_cosine".format(seed)] = by_seed[seed]["cosine"]
        out["mean"] = float(np.mean([row["cosine"] for row in values]))
        out["std"] = float(np.std([row["cosine"] for row in values]))
        out["R2_mean"] = float(np.mean([row["R2"] for row in values]))
        out["NMSE_mean"] = float(np.mean([row["NMSE"] for row in values]))
        table1.append(out)
    table2 = []
    for layer in range(13):
        values = [row for row in identity if row["layer"] == layer]
        out = {"layer": layer}
        for field in ("positive_mean", "negative_mean", "D_gap", "ROC_AUC"):
            nums = [row[field] for row in values]
            out[field + "_mean"] = float(np.mean(nums))
            out[field + "_std"] = float(np.std(nums))
        table2.append(out)
    def mean_std(rows, field):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        return float(np.mean(values)), float(np.std(values))

    table3 = []
    # Table 3 deliberately joins verification and retrieval so a reader can
    # compare the same representation under both diagnostics.
    for representation_name in ("F_visual", "F_cloth"):
        same_rows = [row for row in representation if row["representation"] == representation_name and row["metric_scope"] == "same_clothes_A_B"]
        diff_rows = [row for row in representation if row["representation"] == representation_name and row["metric_scope"] == "different_clothes_A_C_B_C"]
        ret_c = [row for row in retrieval if row["representation"] == representation_name and row["query_protocol"] == "A_gallery_C_query"]
        ret_b = [row for row in retrieval if row["representation"] == representation_name and row["query_protocol"] == "A_gallery_B_query"]
        same_auc, same_auc_std = mean_std(same_rows, "ROC_AUC")
        diff_auc, diff_auc_std = mean_std(diff_rows, "ROC_AUC")
        c_r1, c_r1_std = mean_std(ret_c, "R1")
        c_map, c_map_std = mean_std(ret_c, "mAP")
        b_r1, b_r1_std = mean_std(ret_b, "R1")
        b_map, b_map_std = mean_std(ret_b, "mAP")
        table3.append({
            "representation": representation_name,
            "same_AB_ROC_AUC_mean": same_auc, "same_AB_ROC_AUC_std": same_auc_std,
            "diff_AC_BC_ROC_AUC_mean": diff_auc, "diff_AC_BC_ROC_AUC_std": diff_auc_std,
            "A_gallery_C_R1_mean": c_r1, "A_gallery_C_R1_std": c_r1_std,
            "A_gallery_C_mAP_mean": c_map, "A_gallery_C_mAP_std": c_map_std,
            "A_gallery_B_R1_mean": b_r1, "A_gallery_B_R1_std": b_r1_std,
            "A_gallery_B_mAP_mean": b_map, "A_gallery_B_mAP_std": b_map_std,
        })
    for alpha in ALPHAS:
        verification_rows = [row for row in alpha_v if abs(row["alpha"] - alpha) < 1e-9]
        retrieval_rows_c = [row for row in alpha_r if abs(row["alpha"] - alpha) < 1e-9]
        diff_auc, diff_auc_std = mean_std(verification_rows, "ROC_AUC")
        c_r1, c_r1_std = mean_std(retrieval_rows_c, "R1")
        c_map, c_map_std = mean_std(retrieval_rows_c, "mAP")
        table3.append({
            "representation": "F_residual(alpha={:.2f})".format(alpha),
            "same_AB_ROC_AUC_mean": None, "same_AB_ROC_AUC_std": None,
            "diff_AC_BC_ROC_AUC_mean": diff_auc, "diff_AC_BC_ROC_AUC_std": diff_auc_std,
            "A_gallery_C_R1_mean": c_r1, "A_gallery_C_R1_std": c_r1_std,
            "A_gallery_C_mAP_mean": c_map, "A_gallery_C_mAP_std": c_map_std,
            "A_gallery_B_R1_mean": None, "A_gallery_B_R1_std": None,
            "A_gallery_B_mAP_mean": None, "A_gallery_B_mAP_std": None,
        })
    table4 = []
    for row in aggregate_rows(alpha_r, ["alpha"], ["R1", "R5", "R10", "mAP"]):
        verification = [x for x in aggregate_rows(alpha_v, ["alpha"], ["D_gap", "ROC_AUC", "PR_AUC", "EER"]) if x["alpha"] == row["alpha"]][0]
        table4.append(dict(row, **verification))
    write_csv(table1, os.path.join(REPORT_ROOT, "table1_layerwise_cloth_predictability.csv"))
    write_csv(table2, os.path.join(REPORT_ROOT, "table2_layerwise_identity_discriminability.csv"))
    write_csv(table3, os.path.join(REPORT_ROOT, "table3_representation_diagnostic.csv"))
    write_csv(table4, os.path.join(REPORT_ROOT, "table4_alpha_sweep.csv"))
    write_csv(h1a, os.path.join(REPORT_ROOT, "h1a_layer_predictability_all_seeds.csv"))
    write_csv(h1b, os.path.join(REPORT_ROOT, "h1b_direct_alignment_all_seeds.csv"))
    write_csv(identity, os.path.join(REPORT_ROOT, "layer_identity_discrimination_all_seeds.csv"))
    write_csv(representation, os.path.join(REPORT_ROOT, "representation_verification_all_seeds.csv"))
    write_csv(retrieval, os.path.join(REPORT_ROOT, "retrieval_all_seeds.csv"))
    write_csv(alpha_v, os.path.join(REPORT_ROOT, "alpha_sweep_verification_all_seeds.csv"))
    write_csv(alpha_r, os.path.join(REPORT_ROOT, "alpha_sweep_retrieval_all_seeds.csv"))

    summaries = [read_json(os.path.join(seed_root(seed), "analysis_summary.json")) for seed in SEEDS]
    bootstrap_rows = []
    for seed in SEEDS:
        bootstrap_rows.extend(read_json(os.path.join(seed_root(seed), "bootstrap_summary.json"))["rows"])
    write_csv(bootstrap_rows, os.path.join(REPORT_ROOT, "bootstrap_summary.csv"))

    h1_val_by_seed = {seed: {row["layer"]: row for row in val_h1a if row["seed"] == seed} for seed in SEEDS}
    progression = []
    for seed in SEEDS:
        early = np.mean([h1_val_by_seed[seed][layer]["cosine"] for layer in EARLY_LAYERS])
        middle = np.mean([h1_val_by_seed[seed][layer]["cosine"] for layer in MIDDLE_LAYERS])
        late = np.mean([h1_val_by_seed[seed][layer]["cosine"] for layer in LATE_LAYERS])
        progression.append({"seed": seed, "early_predictability": float(early), "middle_predictability": float(middle), "late_predictability": float(late), "late_minus_early": float(late - early), "final_minus_early": float(h1_val_by_seed[seed][12]["cosine"] - early)})
    h1b_val = [row for row in h1b if row["scope"] == "val"]
    for row in progression:
        seed = row["seed"]
        by_layer = {x["layer"]: x for x in h1b_val if x["seed"] == seed}
        early = np.mean([by_layer[layer]["mean_cosine"] for layer in EARLY_LAYERS])
        late = np.mean([by_layer[layer]["mean_cosine"] for layer in LATE_LAYERS])
        row["early_alignment"] = float(early)
        row["late_alignment"] = float(late)
        row["late_minus_early_alignment"] = float(late - early)
        row["final_minus_early_alignment"] = float(by_layer[12]["mean_cosine"] - early)
    write_csv(progression, os.path.join(REPORT_ROOT, "h1_layer_progression.csv"))

    final_delta = np.asarray([row["final_minus_early"] for row in progression])
    late_delta = np.asarray([row["late_minus_early"] for row in progression])
    if np.all(final_delta >= 0.05) and np.all(late_delta > 0):
        h1_verdict = "STRONG SUPPORT"
    elif np.all(final_delta > 0) and np.all(late_delta > 0):
        h1_verdict = "MODERATE SUPPORT"
    else:
        h1_verdict = "NO SUPPORT"

    cloth_auc = np.asarray([row["cloth_cross_clothes_verification"]["ROC_AUC"] for row in summaries], dtype=np.float64)
    cloth_r1 = np.asarray([row["cloth_retrieval_A_C"]["R1"] for row in summaries], dtype=np.float64)
    random_r1 = 1.0 / 150.0
    if np.all(cloth_auc > 0.70):
        h2_verdict = "STRONG SUPPORT"
    elif np.all(cloth_auc >= 0.55) and np.all(cloth_r1 > random_r1):
        h2_verdict = "MODERATE SUPPORT"
    else:
        h2_verdict = "NO SUPPORT"

    best_minus_r1 = np.asarray([row["alpha_best_minus_1_R1"] for row in summaries])
    best_minus_map = np.asarray([row["alpha_best_minus_1_mAP"] for row in summaries])
    best_alphas = np.asarray([row["best_alpha"] for row in summaries])
    oversuppression = int(np.sum((best_alphas < 1.0) & ((best_minus_r1 >= 0.003) | (best_minus_map >= 0.002)))) >= 2
    if h1_verdict == "NO SUPPORT" and h2_verdict == "NO SUPPORT":
        next_method = "D. Neither — rethink bottleneck"
    elif h1_verdict != "NO SUPPORT" and h2_verdict != "NO SUPPORT":
        next_method = "A. Separated Attention + Structure Preservation"
    elif h1_verdict != "NO SUPPORT":
        next_method = "B. Separated Attention only"
    else:
        next_method = "C. Structure Preservation only"
    decision = {
        "H1_verdict": h1_verdict,
        "H2_verdict": h2_verdict,
        "next_method": next_method,
        "H1_final_minus_early_cosine_by_seed": final_delta.tolist(),
        "H1_late_minus_early_cosine_by_seed": late_delta.tolist(),
        "H2_F_cloth_cross_clothes_AUC_by_seed": cloth_auc.tolist(),
        "H2_F_cloth_A_C_R1_by_seed": cloth_r1.tolist(),
        "alpha_best_by_seed": best_alphas.tolist(),
        "alpha_best_minus_1_R1_by_seed": best_minus_r1.tolist(),
        "alpha_best_minus_1_mAP_by_seed": best_minus_map.tolist(),
        "over_suppression_signal": bool(oversuppression),
        "over_suppression_rule": "best alpha < 1.0 and best-minus-alpha1 R1 >= 0.003 or mAP >= 0.002 in at least 2/3 seeds",
        "H2_operational_rule": "strong if all three F_cloth diff-clothes AUC > 0.70; moderate if all three >= 0.55 and A/C R1 exceeds 1/150 chance",
    }
    write_json(decision, os.path.join(REPORT_ROOT, "diagnosis_decision.json"))
    write_json({"table1": table1, "table2": table2, "table3": table3, "table4": table4, "progression": progression, "summaries": summaries, "decision": decision}, os.path.join(REPORT_ROOT, "analysis_aggregate.json"))
    return decision


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h1a = read_csv(os.path.join(REPORT_ROOT, "h1a_layer_predictability_all_seeds.csv"))
    identity = read_csv(os.path.join(REPORT_ROOT, "layer_identity_discrimination_all_seeds.csv"))
    alpha_r = read_csv(os.path.join(REPORT_ROOT, "alpha_sweep_retrieval_all_seeds.csv"))
    alpha_v = read_csv(os.path.join(REPORT_ROOT, "alpha_sweep_verification_all_seeds.csv"))
    fig_dir = os.path.join(OUT_ROOT, "figures")
    colors = {0: "#1b9e77", 1: "#d95f02", 2: "#7570b3"}

    def line_by_seed(rows, field, scope=None):
        output = {}
        for seed in SEEDS:
            subset = [row for row in rows if row["seed"] == seed and (scope is None or row.get("scope") == scope)]
            output[seed] = (np.asarray([row["layer"] for row in subset]), np.asarray([row[field] for row in subset]))
        return output

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    data = line_by_seed(h1a, "cosine", "val")
    for seed, (x, y) in data.items():
        ax.plot(x, y, marker="o", alpha=0.55, color=colors[seed], label="seed{}".format(seed))
    mean = np.mean(np.stack([data[s][1] for s in SEEDS]), axis=0)
    std = np.std(np.stack([data[s][1] for s in SEEDS]), axis=0)
    x = data[0][0]
    ax.plot(x, mean, color="black", linewidth=2.2, label="mean")
    ax.fill_between(x, mean - std, mean + std, color="black", alpha=0.12)
    ax.set(xlabel="Transformer depth (Z_l)", ylabel="Linear ridge cosine predictability", title="Figure 1. PDF-defined cloth-component predictability")
    ax.legend(frameon=False, ncol=2); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure1_depth_vs_cloth_predictability.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    data_id = {}
    for seed in SEEDS:
        subset = [row for row in identity if row["seed"] == seed]
        data_id[seed] = (np.asarray([row["layer"] for row in subset]), np.asarray([row["ROC_AUC"] for row in subset]))
        ax.plot(data_id[seed][0], data_id[seed][1], marker="o", alpha=0.55, color=colors[seed], label="seed{}".format(seed))
    mean_id = np.mean(np.stack([data_id[s][1] for s in SEEDS]), axis=0)
    ax.plot(data_id[0][0], mean_id, color="black", linewidth=2.2, label="mean")
    ax.set(xlabel="Transformer depth (Z_l)", ylabel="Identity ROC-AUC (diff-clothes vs different-ID)", title="Figure 2. Layer-wise identity discriminability")
    ax.legend(frameon=False, ncol=2); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "figure2_depth_vs_identity_auc.png"), dpi=180); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.7))
    ax2 = ax1.twinx()
    for seed in SEEDS:
        ax1.plot(data_id[seed][0], data_id[seed][1], color=colors[seed], alpha=0.38, linestyle="--")
        ax2.plot(data[seed][0], data[seed][1], color=colors[seed], alpha=0.38)
    ax1.plot(data_id[0][0], mean_id, color="black", linewidth=2.2, linestyle="--", label="identity ROC-AUC mean")
    ax2.plot(x, mean, color="black", linewidth=2.2, label="cloth predictability mean")
    ax1.set_xlabel("Transformer depth (Z_l)"); ax1.set_ylabel("Identity ROC-AUC", color="black"); ax2.set_ylabel("Cloth-component predictability cosine", color="black")
    ax1.set_title("Figure 3. Information-flow trade-off"); ax1.grid(alpha=0.2)
    handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "figure3_dual_axis_information_flow.png"), dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharex=True)
    alpha_panels = [
        (alpha_r, "R1", "A/C R1 (%)", lambda value: 100.0 * value),
        (alpha_r, "mAP", "A/C mAP", lambda value: value),
        (alpha_v, "ROC_AUC", "Diff-clothes ROC-AUC", lambda value: value),
    ]
    for ax, (source_rows, field, ylabel, transform) in zip(axes, alpha_panels):
        for seed in SEEDS:
            subset = [row for row in source_rows if row["seed"] == seed]
            ax.plot([row["alpha"] for row in subset], [transform(row[field]) for row in subset], marker="o", alpha=0.35, color=colors[seed], label="seed{}".format(seed))
        groups = aggregate_rows(source_rows, ["alpha"], [field])
        ax.plot([row["alpha"] for row in groups], [transform(row[field + "_mean"]) for row in groups], marker="o", color="black", linewidth=2.2, label="mean")
        ax.set_xlabel("Offline subtraction alpha")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_title("R1")
    axes[1].set_title("mAP")
    axes[2].set_title("ROC-AUC")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 4. Subtraction trade-off", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "figure4_alpha_cross_clothes_retrieval.png"), dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for seed in SEEDS:
        path = os.path.join(seed_root(seed), "distribution_samples.npz")
        values = np.load(path)
        for ax, (left, right, title) in zip(axes, [
            ("F_visual_diff", "F_visual_neg", "F_visual"),
            ("F_cloth_diff", "F_cloth_neg", "F_cloth"),
            ("residual_best_diff", "residual_best_neg", "residual(best alpha)"),
        ]):
            ax.hist(values[left], bins=45, density=True, alpha=0.16, color=colors[seed])
            ax.hist(values[right], bins=45, density=True, histtype="step", linewidth=1.0, color=colors[seed], label="seed{} neg".format(seed))
            ax.set_title(title); ax.set_xlabel("Cosine similarity"); ax.grid(alpha=0.15)
    axes[0].set_ylabel("Density")
    fig.suptitle("Figure 5. Positive/negative verification distributions", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "figure5_representation_distributions.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"status": "figures_written", "directory": fig_dir}, indent=2))


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return ("{:.%df}" % digits).format(float(value))
    except Exception:
        return str(value)


def markdown_table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(output)


def make_report():
    decision = read_json(os.path.join(REPORT_ROOT, "diagnosis_decision.json"))
    aggregate = read_json(os.path.join(REPORT_ROOT, "analysis_aggregate.json"))
    prep = read_json(os.path.join(REPORT_ROOT, "prepare_manifest.json"))
    summaries = aggregate["summaries"]
    progression = aggregate["progression"]
    table1 = aggregate["table1"]
    table2 = aggregate["table2"]
    table3 = aggregate["table3"]
    table4 = aggregate["table4"]
    lines = []
    lines.append("# PDF Representation Leakage & Over-Suppression Diagnosis\n")
    lines.append("**Protocol:** PDF-RLOS-Diagnosis  \\  **Scope:** PRCC TRAIN only  \\  **Seeds:** R00 seed0/seed1/seed2, epoch50 final\n")
    lines.append("## Executive conclusion\n")
    lines.append("- **H1 — cloth-component predictability inside the identity stream:** **{}**. Final-minus-early validation cosine by seed: `{}`; late-minus-early: `{}`.\n".format(decision["H1_verdict"], ", ".join(fmt(x) for x in decision["H1_final_minus_early_cosine_by_seed"]), ", ".join(fmt(x) for x in decision["H1_late_minus_early_cosine_by_seed"])))
    lines.append("- **H2 — cross-clothing identity-correlated information in `F_cloth`:** **{}**. Different-clothes verification AUC by seed: `{}`; A-gallery/C-query R1 by seed: `{}`.\n".format(decision["H2_verdict"], ", ".join(fmt(x) for x in decision["H2_F_cloth_cross_clothes_AUC_by_seed"]), ", ".join(fmt(100 * x, 2) + "%" for x in decision["H2_F_cloth_A_C_R1_by_seed"])))
    lines.append("- **Recommended next method:** **{}**.\n".format(decision["next_method"]))
    lines.append("- **Over-suppression signal:** `{}` under the preregistered rule. Best alpha by seed: `{}`; best-minus-alpha=1 R1: `{}` pp; mAP: `{}` pp.\n".format("SUPPORTED" if decision["over_suppression_signal"] else "NOT SUPPORTED", ", ".join(fmt(x, 2) for x in decision["alpha_best_by_seed"]), ", ".join(fmt(100 * x, 3) for x in decision["alpha_best_minus_1_R1_by_seed"]), ", ".join(fmt(100 * x, 3) for x in decision["alpha_best_minus_1_mAP_by_seed"])))
    lines.append("\nInterpretation is intentionally bounded: H1 refers to information predictive of **PDF's own PDF-defined cloth-related component**, not ground-truth clothing leakage. H2 refers to cross-clothing identity-correlated information; without TRAIN parsing evidence it is not a claim that body structure has been proven in `F_cloth`.\n")

    lines.append("## 1. PDF code audit\n")
    lines.append("See [pdf_representation_audit.md](pdf_representation_audit.md). The actual code is a 12-block ViT-B/16 with 512-D projected visual tokens. `F_visual` is the eval BN output of visual CLS; `F_cloth` is the training-only `com_proj` route using the frozen original caption, `encode_text_irra`, cross-modal transformer/attention, EOT selection, and the same BN. Offline residuals use the exact BN-space formula `normalize(F_visual - alpha * F_cloth)`.\n")
    lines.append("## 2. Baseline checkpoint provenance\n")
    lines.append("See [baseline_checkpoint_manifest.json](baseline_checkpoint_manifest.json). All three files passed `run_id=R00`, `epoch=50`, and the actual payload role `primary epoch50 final`; seed checks also passed. Any `best_test_*` fields are recorded as auxiliary provenance only; they were not selected.\n")
    provenance_rows = []
    for entry in read_json(os.path.join(REPORT_ROOT, "baseline_checkpoint_manifest.json"))["checkpoints"]:
        provenance_rows.append({"seed": entry["seed"], "epoch": entry["epoch"], "checkpoint SHA256": entry["checkpoint_sha256"][:12] + "…", "source commit": entry["source_commit"][:12] + "…", "path": entry["checkpoint_path"]})
    lines.append(markdown_table(["seed", "epoch", "checkpoint SHA256", "source commit", "path"], provenance_rows) + "\n")

    lines.append("## 3. Train split and pair manifests\n")
    lines.append("The full extraction inventory contains `{}` PRCC TRAIN images and `{}` identities. H1 uses the copied frozen semantic split: 100 dev IDs / 600 selected images and 48 val IDs / 288 selected images, with no identity overlap. H2 and alpha-sweep pair/retrieval diagnostics use full PRCC TRAIN. Fixed pair seed is `{}`; the negative manifest contains `{}` negatives and is shared across all representations and seeds.\n".format(prep["inventory"]["count"], prep["inventory"]["person_count"], PAIR_SEED, NEGATIVE_COUNT))
    lines.append("Split details: [split_resolution.json](../outputs/pdf_representation_diagnosis/shared/split_manifest/split_resolution.json); pairs: [pair_manifest.json](../outputs/pdf_representation_diagnosis/shared/pair_manifest/pair_manifest.json); negatives: [negative_manifest.json](../outputs/pdf_representation_diagnosis/shared/negative_manifest/negative_manifest.json).\n")

    lines.append("## 4. Extraction validation\n")
    for summary in summaries:
        extraction = read_json(os.path.join(seed_root(summary["seed"]), "extraction_manifest.json"))
        lines.append("- seed{}: `{}` images, `Z_0...Z_12`, raw/L2 files; `Z_12` versus actual visual output max error `{}`; `F_visual` versus eval image-only forward max error `{}`.\n".format(summary["seed"], extraction["count"], fmt(extraction["sanity"]["max_z12_vs_visual_output_error"], 7), fmt(extraction["sanity"]["max_fvisual_vs_model_eval_image_only_error"], 7)))
    lines.append("No model parameters were updated and no PRCC TEST path was consumed. Region analysis was skipped because no reliable TRAIN parsing cache exists; see [region_analysis_status.json](region_analysis_status.json).\n")

    lines.append("## 5. H1-A — layer-wise linear predictability\n")
    lines.append("Ridge probes are fit only on the 100-ID dev images, with fixed `ridge_alpha={}` and no validation hyperparameter search. The reported H1 curve is identity-disjoint val.\n".format(RIDGE_ALPHA))
    t1_rows = []
    for row in table1:
        t1_rows.append({"layer": "Z_{}".format(row["layer"]), "seed0 cosine": fmt(row["seed0_cosine"]), "seed1 cosine": fmt(row["seed1_cosine"]), "seed2 cosine": fmt(row["seed2_cosine"]), "mean": fmt(row["mean"]), "std": fmt(row["std"]), "R2": fmt(row["R2_mean"]), "NMSE": fmt(row["NMSE_mean"])})
    lines.append(markdown_table(["layer", "seed0 cosine", "seed1 cosine", "seed2 cosine", "mean", "std", "R2", "NMSE"], t1_rows) + "\n")
    lines.append("Raw all-seed values: [h1a_layer_predictability_all_seeds.csv](h1a_layer_predictability_all_seeds.csv).\n")

    lines.append("## 6. H1-B — direct alignment\n")
    lines.append("Direct alignment uses per-sample cosine, absolute cosine, sample-wise projection energy `cosine²`, and linear CKA, separately on dev and val. `Z_0` is image-independent before the visual transformer, so its centered CKA is undefined and is recorded as NA; this is an architectural baseline, not a missing computation. Values are in [h1b_direct_alignment_all_seeds.csv](h1b_direct_alignment_all_seeds.csv).\n")

    lines.append("## 7. H1-C — layer progression\n")
    prog_rows = []
    for row in progression:
        prog_rows.append({"seed": row["seed"], "early Z1-4": fmt(row["early_predictability"]), "middle Z5-8": fmt(row["middle_predictability"]), "late Z9-12": fmt(row["late_predictability"]), "late-early": fmt(row["late_minus_early"]), "final-early": fmt(row["final_minus_early"]), "alignment late-early": fmt(row["late_minus_early_alignment"])})
    lines.append(markdown_table(["seed", "early Z1-4", "middle Z5-8", "late Z9-12", "late-early", "final-early", "alignment late-early"], prog_rows) + "\n")
    lines.append("The core visualizations are [Figure 1](../outputs/pdf_representation_diagnosis/figures/figure1_depth_vs_cloth_predictability.png), [Figure 2](../outputs/pdf_representation_diagnosis/figures/figure2_depth_vs_identity_auc.png), and [Figure 3](../outputs/pdf_representation_diagnosis/figures/figure3_dual_axis_information_flow.png).\n")

    lines.append("## 8. Layer-wise identity discrimination\n")
    t2_rows = []
    for row in table2:
        t2_rows.append({"layer": "Z_{}".format(row["layer"]), "diff-positive cosine": fmt(row["positive_mean_mean"]), "negative cosine": fmt(row["negative_mean_mean"]), "D_gap": fmt(row["D_gap_mean"]), "ROC-AUC": fmt(row["ROC_AUC_mean"])})
    lines.append(markdown_table(["layer", "diff-positive cosine", "negative cosine", "D_gap", "ROC-AUC"], t2_rows) + "\n")
    lines.append("Identity positives are fixed same-ID different-clothes A-C and B-C pairs; negatives are the shared different-ID manifest. Full values: [layer_identity_discrimination_all_seeds.csv](layer_identity_discrimination_all_seeds.csv).\n")

    lines.append("## 9. H1 verdict\n")
    lines.append("**{}**. Q1: final CLS is {} to predict `F_cloth` than the early Z1-4 average, with final-minus-early val cosine `{}` across seeds. Q2: the direction is `{}/3` seeds positive. This supports the bounded statement that later PDF identity-stream representations contain more information predictive of the PDF-defined cloth-related component; it does not prove ground-truth clothing leakage.\n".format(decision["H1_verdict"], "easier" if np.mean(decision["H1_final_minus_early_cosine_by_seed"]) > 0 else "not easier", ", ".join(fmt(x) for x in decision["H1_final_minus_early_cosine_by_seed"]), int(np.sum(np.asarray(decision["H1_final_minus_early_cosine_by_seed"]) > 0))))

    lines.append("## 10. H2-A — `F_cloth` identity verification\n")
    h2_rows = []
    for summary in summaries:
        metric = summary["cloth_cross_clothes_verification"]
        h2_rows.append({"seed": summary["seed"], "positive mean": fmt(metric["positive_mean"]), "negative mean": fmt(metric["negative_mean"]), "D_gap": fmt(metric["D_gap"]), "ROC-AUC": fmt(metric["ROC_AUC"]), "PR-AUC": fmt(metric["PR_AUC"]), "EER": fmt(metric["EER"])})
    lines.append(markdown_table(["seed", "positive mean", "negative mean", "D_gap", "ROC-AUC", "PR-AUC", "EER"], h2_rows) + "\n")
    lines.append("Same-clothes A-B and separate A-C/B-C results for `F_visual`, `F_cloth`, and residuals are in [representation_verification_all_seeds.csv](representation_verification_all_seeds.csv). Q3: `F_cloth` cross-clothes verification AUC is `{}` across the three seeds, so the evidence is `{}` under the stated 0.55/0.60/0.70 diagnostic bands.\n".format(", ".join(fmt(x) for x in decision["H2_F_cloth_cross_clothes_AUC_by_seed"]), decision["H2_verdict"]))
    def mean_pm(value, std, scale=1.0, suffix=""):
        if value is None or std is None:
            return "NA"
        return "{} ± {}{}".format(fmt(scale * value), fmt(scale * std), suffix)

    t3_rows = []
    for row in table3:
        t3_rows.append({
            "representation": row["representation"],
            "same A-B AUC": mean_pm(row["same_AB_ROC_AUC_mean"], row["same_AB_ROC_AUC_std"]),
            "diff A-C/B-C AUC": mean_pm(row["diff_AC_BC_ROC_AUC_mean"], row["diff_AC_BC_ROC_AUC_std"]),
            "A/C R1": mean_pm(row["A_gallery_C_R1_mean"], row["A_gallery_C_R1_std"], 100.0, "%"),
            "A/C mAP": mean_pm(row["A_gallery_C_mAP_mean"], row["A_gallery_C_mAP_std"]),
            "A/B R1": mean_pm(row["A_gallery_B_R1_mean"], row["A_gallery_B_R1_std"], 100.0, "%"),
            "A/B mAP": mean_pm(row["A_gallery_B_mAP_mean"], row["A_gallery_B_mAP_std"]),
        })
    lines.append("\n**Table 3 — representation diagnostic (mean ± seed std).**\n\n")
    lines.append(markdown_table(["representation", "same A-B AUC", "diff A-C/B-C AUC", "A/C R1", "A/C mAP", "A/B R1", "A/B mAP"], t3_rows) + "\n")

    lines.append("## 11. H2-B — `F_cloth` cross-clothes retrieval\n")
    ret_rows = []
    for summary in summaries:
        cloth = summary["cloth_retrieval_A_C"]
        visual = summary["visual_retrieval_A_C"]
        ret_rows.append({"seed": summary["seed"], "F_visual R1": fmt(100 * visual["R1"], 2) + "%", "F_cloth R1": fmt(100 * cloth["R1"], 2) + "%", "F_visual mAP": fmt(visual["mAP"]), "F_cloth mAP": fmt(cloth["mAP"])})
    lines.append(markdown_table(["seed", "F_visual R1", "F_cloth R1", "F_visual mAP", "F_cloth mAP"], ret_rows) + "\n")
    lines.append("Q4: `F_cloth` A-gallery/C-query retrieval is nontrivial relative to 1/150 identity chance (`{}`), with R1 `{}`. A-gallery/B-query controls and all residual retrieval rows are in [retrieval_all_seeds.csv](retrieval_all_seeds.csv).\n".format(fmt(100 / 150, 2) + "%", ", ".join(fmt(100 * x, 2) + "%" for x in decision["H2_F_cloth_A_C_R1_by_seed"])))

    lines.append("## 12. H2-C — offline alpha sweep\n")
    a_rows = []
    for row in table4:
        a_rows.append({"alpha": fmt(row["alpha"], 2), "R1 mean ± std": "{} ± {}%".format(fmt(100 * row["R1_mean"], 2), fmt(100 * row["R1_std"], 2)), "mAP mean ± std": "{} ± {}".format(fmt(row["mAP_mean"]), fmt(row["mAP_std"])), "ROC-AUC mean ± std": "{} ± {}".format(fmt(row["ROC_AUC_mean"]), fmt(row["ROC_AUC_std"])), "D_gap mean ± std": "{} ± {}".format(fmt(row["D_gap_mean"]), fmt(row["D_gap_std"]))})
    lines.append(markdown_table(["alpha", "R1 mean ± std", "mAP mean ± std", "ROC-AUC mean ± std", "D_gap mean ± std"], a_rows) + "\n")
    r1_peak = max(table4, key=lambda row: row["R1_mean"])
    map_peak = max(table4, key=lambda row: row["mAP_mean"])
    if r1_peak["alpha"] < max(row["alpha"] for row in table4) and map_peak["alpha"] == max(row["alpha"] for row in table4):
        alpha_shape = "shows a shallow R1 peak at alpha={} followed by a {:.3f} pp drop at alpha=1.50, while mAP keeps rising to alpha=1.50; it is not a joint inverted-U".format(fmt(r1_peak["alpha"], 2), 100 * (r1_peak["R1_mean"] - table4[-1]["R1_mean"]))
    else:
        alpha_shape = "does not show a joint inverted-U across R1 and mAP"
    alpha1_best = [s["seed"] for s in summaries if abs(s["best_alpha"] - 1.0) < 1e-9]
    lines.append("The trade-off plot is [Figure 4](../outputs/pdf_representation_diagnosis/figures/figure4_alpha_cross_clothes_retrieval.png). Q5: the mean curve {}. Q6: alpha=1 is the best grid point for seed{} only; it does not exceed the best point for the other seeds. Per-seed best-minus-1 deltas are reported above and in [diagnosis_decision.json](diagnosis_decision.json).\n".format(alpha_shape, ",".join(str(x) for x in alpha1_best) if alpha1_best else "none"))

    lines.append("## 13. Over-suppression analysis\n")
    lines.append("The preregistered signal is: best alpha `< 1.0`, alpha=1 lower than best by at least 0.3 percentage points R1 or 0.2 percentage points mAP, in at least 2/3 seeds. Result: **{}**. This is evidence about the frozen subtraction mechanism, not a training change.\n".format("SUPPORTED" if decision["over_suppression_signal"] else "NOT SUPPORTED"))

    lines.append("## 14. Optional region analysis\n")
    lines.append("Skipped. The environment exposes a parsing directory for PRCC TEST only and no reliable PRCC TRAIN parsing cache. No new parsing model was deployed, and no Figure 6 is generated. Consequently there is no basis here to claim head/limb/body-structure evidence in `F_cloth`.\n")

    lines.append("## 15. Three-seed consistency\n")
    lines.append("All core extraction, H1, H2, and alpha-sweep stages were run for seed0, seed1, and seed2. The report retains individual seed values rather than selecting a favorable seed.\n")
    lines.append(markdown_table(["seed", "F_cloth AUC", "F_cloth A/C R1", "best alpha", "best-1 R1 (pp)", "best-1 mAP (pp)"], [{"seed": s["seed"], "F_cloth AUC": fmt(s["cloth_cross_clothes_verification"]["ROC_AUC"]), "F_cloth A/C R1": fmt(100 * s["cloth_retrieval_A_C"]["R1"], 2) + "%", "best alpha": fmt(s["best_alpha"], 2), "best-1 R1 (pp)": fmt(100 * s["alpha_best_minus_1_R1"], 3), "best-1 mAP (pp)": fmt(100 * s["alpha_best_minus_1_mAP"], 3)} for s in summaries]) + "\n")

    lines.append("## 16. Bootstrap\n")
    lines.append("Core train-only bootstrap summaries use 10,000 resamples at the identity unit where defined: `F_cloth` cross-clothes D-gap/AUC, best-minus-alpha=1 retrieval deltas, and prototype identity retention.\n")
    bootstrap_rows = read_csv(os.path.join(REPORT_ROOT, "bootstrap_summary.csv"))
    bootstrap_md = []
    for row in bootstrap_rows:
        bootstrap_md.append({"seed": row["seed"], "metric": row["metric"], "estimate": fmt(row["estimate"]), "95% CI": "[{}, {}]".format(fmt(row["ci_low"]), fmt(row["ci_high"])), "n ID": row["n_identity"]})
    lines.append(markdown_table(["seed", "metric", "estimate", "95% CI", "n ID"], bootstrap_md) + "\n")
    lines.append("Full machine-readable output: [bootstrap_summary.csv](bootstrap_summary.csv).\n")

    lines.append("## 17. H2 verdict\n")
    lines.append("**{}**. Q7: evidence for PDF-defined representation leakage is **{}**, based on the identity-disjoint layer probe and progression. Q8: evidence for over-suppression is **{}** under the fixed alpha rule. Q3/Q4 are the H2 identity-retention checks; their conclusions are limited to identity-correlated information in `F_cloth`.\n".format(decision["H2_verdict"], decision["H1_verdict"], "SUPPORTED" if decision["over_suppression_signal"] else "NOT SUPPORTED"))

    lines.append("## 18. Final decision matrix\n")
    lines.append(markdown_table(["case", "condition", "interpretation", "route"], [
        {"case": "A", "condition": "H1 supported + H2 supported", "interpretation": "leakage and useful identity information both present", "route": "Separated Attention + Structure Preservation"},
        {"case": "B", "condition": "H1 supported + H2 not supported", "interpretation": "cloth component enters identity representation progressively", "route": "Separated Attention only"},
        {"case": "C", "condition": "H1 not supported + H2 supported", "interpretation": "suppression risks removing identity-correlated information", "route": "Structure Preservation only"},
        {"case": "D", "condition": "H1 not supported + H2 not supported", "interpretation": "hypothesized bottlenecks lack evidence", "route": "Neither — rethink bottleneck"},
    ]) + "\n")
    lines.append("Observed decision: **{} / {} -> {}**.\n".format(decision["H1_verdict"], decision["H2_verdict"], decision["next_method"]))

    lines.append("## 19. Recommended next method\n")
    lines.append("Q9: **{}**. This recommendation follows only from the TRAIN dev/val and TRAIN diagnostic evidence above; PRCC TEST was not used to select it.\n".format(decision["next_method"]))
    lines.append("## 20. Limitations\n")
    lines.append("- `F_cloth` is the PDF-defined `com_proj` path, not a ground-truth clothing annotation.\n- The caption-conditioned cloth component depends on the frozen caption/cache path and the eval-mode BN running statistics used to make extraction deterministic.\n- H1 probes use only the frozen 888-image selection; H2 uses full TRAIN pairs/retrieval, so the populations are intentionally different and explicitly reported.\n- AUC/retrieval evidence is correlational and cannot by itself establish causal information flow.\n- Region analysis is unavailable without a TRAIN parsing cache.\n- Bootstrap CIs reflect identity resampling; they do not remove all image-level dependence within identity.\n- No formal PRCC TEST benchmark was run in this phase.\n")
    lines.append("\n## Artifact index\n")
    lines.append("- Audit: [pdf_representation_audit.md](pdf_representation_audit.md)\n- Checkpoint manifest: [baseline_checkpoint_manifest.json](baseline_checkpoint_manifest.json)\n- Figures: `../outputs/pdf_representation_diagnosis/figures/`\n- Per-seed features: `../outputs/pdf_representation_diagnosis/seed{0,1,2}/`\n")
    report_path = os.path.join(REPORT_ROOT, "PDF_representation_leakage_oversuppression_report.md")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "report_written", "path": report_path, "H1": decision["H1_verdict"], "H2": decision["H2_verdict"], "next_method": decision["next_method"]}, indent=2))
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "verify", "analyze", "figures", "report", "all"])
    parser.add_argument("--seed", type=int, choices=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, REPO_ROOT)
    ensure_dirs()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "extract":
        if args.seed is None:
            raise RuntimeError("extract requires --seed")
        extract_seed(args.seed, args.device, args.batch_size, args.workers, args.force)
    elif args.stage == "verify":
        verify_extractions()
    elif args.stage == "analyze":
        verify_extractions()
        for seed in SEEDS:
            analysis_for_seed(seed, args.device)
        aggregate_analysis()
    elif args.stage == "figures":
        make_figures()
    elif args.stage == "report":
        make_report()
    else:
        prepare()
        for seed in SEEDS:
            extract_seed(seed, args.device, args.batch_size, args.workers, args.force)
        verify_extractions()
        for seed in SEEDS:
            analysis_for_seed(seed, args.device)
        aggregate_analysis()
        make_figures()
        make_report()


if __name__ == "__main__":
    main()
