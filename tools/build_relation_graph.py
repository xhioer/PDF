"""Build and freeze the RCHRL-V1 train-only relation graph.

No PRCC test directory is enumerated in this file.  The only image inventory
is ``rgb/train`` and all semantic rows are matched to that inventory before
they can be used.
"""
from __future__ import absolute_import

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.default_img import _C
from models.clip_model import build_CLIP_from_openai_pretrained, tokenize as clip_tokenize
from models.utils.simple_tokenizer import SimpleTokenizer
from tools.rchrl_common import (
    ATTRIBUTES, CAMERA_TO_INT, DATA_ROOT, K, P2_CACHE, P2_RULES,
    P2_VALIDATION, REPO_ROOT, ImagePathDataset, distribution,
    feature_transform, json_dump, jsonl_write, load_p2_cache,
    load_train_records, p2_phrase, repo_commit, sha256_file, stable_hash,
    projected_image_cls, TRAIN_ROOT,
)


def _hash_uint64(value):
    return np.uint64(int(stable_hash(value)[:16], 16))


def _metadata(row, semantic):
    result = dict(row)
    result.update(semantic[row["path"]])
    return result


def _pair_key(row):
    return "{}->{}".format(row["camera"], row["clothes_state"])


def _normalise_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def _topk_deterministic(scores, anchor_pid, pids, hash_values, k=K):
    """Exact top-k by score, stable path hash as the only tie-break."""
    scores = np.asarray(scores, dtype=np.float32).copy()
    scores[pids == anchor_pid] = -np.inf
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) < k:
        raise RuntimeError("Not enough different-ID negatives")
    candidate = valid[np.argpartition(-scores[valid], k - 1)[:k]]
    threshold = float(np.min(scores[candidate]))
    greater = valid[scores[valid] > threshold]
    equal = valid[scores[valid] == threshold]
    if len(greater) > k:
        raise RuntimeError("top-k threshold calculation failed")
    greater = greater[np.argsort(-scores[greater], kind="mergesort")]
    equal = equal[np.argsort(hash_values[equal], kind="mergesort")]
    selected = np.concatenate([greater, equal[:k - len(greater)]])
    if len(selected) != k:
        raise RuntimeError("deterministic top-k returned {} rows".format(len(selected)))
    return selected


def mine_topk(features, records, name, device, batch_rows=128):
    """Mine a train-only top-20 list from a normalized feature matrix."""
    n = len(records)
    pids = np.asarray([row["person_id"] for row in records], dtype=np.int32)
    hashes = np.asarray([_hash_uint64(row["path"]) for row in records], dtype=np.uint64)
    top_indices = np.zeros((n, K), dtype=np.int32)
    top_scores = np.zeros((n, K), dtype=np.float32)
    gpu_features = torch.from_numpy(features).to(device=device)
    for start in range(0, n, batch_rows):
        stop = min(start + batch_rows, n)
        block = torch.from_numpy(features[start:stop]).to(device=device)
        scores = torch.mm(block, gpu_features.t()).float().cpu().numpy()
        for local in range(stop - start):
            anchor = start + local
            selected = _topk_deterministic(scores[local], pids[anchor], pids, hashes)
            order = np.lexsort((hashes[selected], -scores[local, selected]))
            selected = selected[order]
            top_indices[anchor] = selected
            top_scores[anchor] = scores[local, selected]
        print("{} mining {}/{}".format(name, stop, n), flush=True)
    return top_indices, top_scores


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("epoch") != 50 or not checkpoint.get("train_only") or \
            checkpoint.get("checkpoint_selection") != "fixed epoch50 final; no test-adaptive selection":
        raise RuntimeError("V0 checkpoint provenance gate failed")
    model, _ = build_CLIP_from_openai_pretrained(
        "ViT-B/16", (384, 128), 16, 150)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().float().to(device)
    model.freeze_text_encoder()
    model.eval()
    return model, checkpoint


def extract_visual_features(model, records, device):
    config = _C.clone()
    config.defrost()
    config.DATA.HEIGHT = 384
    config.DATA.WIDTH = 128
    config.freeze()
    dataset = ImagePathDataset(records, feature_transform(config))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)
    features = np.zeros((len(records), 512), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device=device, non_blocking=True)
            batch = projected_image_cls(model, images).float()
            batch = torch.nn.functional.normalize(batch, dim=1)
            features[indices.numpy()] = batch.cpu().numpy()
    return features


def tokenize_phrases(phrases):
    tokenizer = SimpleTokenizer()
    result = torch.zeros((len(phrases), 77), dtype=torch.long)
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    for idx, phrase in enumerate(phrases):
        tokens = [sot] + tokenizer.encode(phrase) + [eot]
        if len(tokens) > 77:
            tokens = tokens[:77]
            tokens[-1] = eot
        result[idx, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


def extract_semantic_features(model, records, semantic, device):
    phrases = []
    for attr in ATTRIBUTES:
        values = sorted(set(semantic[row["path"]][attr] for row in records
                            if semantic[row["path"]][attr] != "unknown"))
        phrases.extend((attr, value, p2_phrase(attr, value)) for value in values)
    phrase_texts = [item[2] for item in phrases]
    phrase_features = {}
    with torch.no_grad():
        for start in range(0, len(phrase_texts), 64):
            token_ids = tokenize_phrases(phrase_texts[start:start + 64]).to(device)
            encoded = model.encode_text(token_ids).float()
            encoded = torch.nn.functional.normalize(encoded, dim=1)
            for idx, row in enumerate(encoded.cpu().numpy()):
                phrase_features[(phrases[start + idx][0], phrases[start + idx][1])] = row

    result = np.zeros((len(records), 512), dtype=np.float32)
    for idx, record in enumerate(records):
        vectors = []
        for attr in ATTRIBUTES:
            value = semantic[record["path"]][attr]
            if value != "unknown":
                vectors.append(phrase_features[(attr, value)])
        if vectors:
            result[idx] = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return _normalise_rows(result), {"phrase_count": len(phrase_texts),
                                    "serialization": "mean of valid per-attribute phrases",
                                    "phrase_template": "A person with {attribute}: {value}.",
                                    "unknown_values_encoded": False}


def edge_record(anchor_idx, negative_idx, records, visual_cos, semantic_cos,
                visual_rank=None, semantic_rank=None, hybrid_rank=None,
                match_level=None):
    anchor, negative = records[anchor_idx], records[negative_idx]
    row = {
        "anchor_index": int(anchor_idx),
        "negative_index": int(negative_idx),
        "anchor_path": anchor["path"],
        "negative_path": negative["path"],
        "anchor_person_id": int(anchor["person_id"]),
        "negative_person_id": int(negative["person_id"]),
        "anchor_camera": anchor["camera"],
        "negative_camera": negative["camera"],
        "anchor_clothes_id": int(anchor["clothes_id"]),
        "negative_clothes_id": int(negative["clothes_id"]),
        "anchor_clothes_key": anchor["clothes_key"],
        "negative_clothes_key": negative["clothes_key"],
        "anchor_clothes_state": anchor["clothes_state"],
        "negative_clothes_state": negative["clothes_state"],
        "visual_cosine": float(visual_cos),
        "semantic_cosine": float(semantic_cos),
    }
    if visual_rank is not None:
        row["visual_rank"] = int(visual_rank)
        row["H_visual_rank"] = float(1.0 - (visual_rank - 1) / float(K - 1))
    if semantic_rank is not None:
        row["semantic_rank"] = int(semantic_rank)
        row["H_sem_rank"] = float(1.0 - (semantic_rank - 1) / float(K - 1))
    if hybrid_rank is not None:
        row["hybrid_rank"] = int(hybrid_rank)
        row["H_hybrid"] = float(1.0 - (hybrid_rank - 1) / float(K - 1))
    if match_level is not None:
        row["match_level"] = match_level
    return row


def positive_edge_record(anchor_idx, positive_idx, records, semantic):
    anchor, positive = records[anchor_idx], records[positive_idx]
    r_conf, r_agr, r_joint = _reliability(semantic[anchor["path"]], semantic[positive["path"]])
    return {
        "edge_type": "same_id_cross_clothes",
        "anchor_index": int(anchor_idx),
        "positive_index": int(positive_idx),
        "anchor_path": anchor["path"],
        "positive_path": positive["path"],
        "person_id": int(anchor["person_id"]),
        "camera_i": anchor["camera"],
        "camera_j": positive["camera"],
        "clothes_i": int(anchor["clothes_id"]),
        "clothes_j": int(positive["clothes_id"]),
        "clothes_key_i": anchor["clothes_key"],
        "clothes_key_j": positive["clothes_key"],
        "R_conf": float(r_conf),
        "R_agr": float(r_agr),
        "R_joint": float(r_joint),
    }


def _reliability(a, b):
    conf = []
    agreement = []
    weights = {"high": 1.0, "medium": 0.5, "low": 0.0, "unknown": 0.0}
    for attr in ATTRIBUTES:
        ca = weights.get(a.get(attr + "_confidence", "low"), 0.0)
        cb = weights.get(b.get(attr + "_confidence", "low"), 0.0)
        conf.append(math.sqrt(ca * cb))
        va, vb = a.get(attr, "unknown"), b.get(attr, "unknown")
        if va != "unknown" and vb != "unknown":
            agreement.append(float(va == vb))
    r_conf = float(np.mean(conf)) if conf else 0.0
    r_agr = float(np.mean(agreement)) if agreement else 0.0
    return r_conf, r_agr, float(np.clip(r_conf * r_agr, 0.0, 1.0))


def _choose_bit(mask, anchor_path, slot, count):
    if mask == 0:
        return None
    start = int(stable_hash("{}|{}|matched-control".format(anchor_path, slot))[:16], 16) % count
    rotated = mask >> start
    if rotated:
        low = rotated & -rotated
        return start + low.bit_length() - 1
    low = mask & -mask
    return low.bit_length() - 1


def choose_matched_negative(anchor_idx, target_idx, slot, records, semantic,
                            metadata_index, exclude_indices=()):
    anchor = records[anchor_idx]
    target = records[target_idx]
    target_overlap = tuple(bool(semantic[target["path"]][attr] != "unknown" and
                                semantic[anchor["path"]][attr] != "unknown" and
                                semantic[target["path"]][attr] == semantic[anchor["path"]][attr])
                           for attr in ATTRIBUTES)
    # The anchor camera/group is fixed by construction.  We explicitly retain
    # it in the provenance and match candidate negative metadata hierarchically.
    prefixes = [4, 3, 2, 1, 0]
    chosen = None
    used = None
    excluded = metadata_index["person"].get(anchor["person_id"], 0)
    excluded |= 1 << int(target_idx)
    for index in exclude_indices:
        excluded |= 1 << int(index)
    for prefix in prefixes:
        mask = metadata_index["camera_state"].get(
            (target["camera"], target["clothes_state"]), 0)
        for attr_idx, attr in enumerate(ATTRIBUTES[:prefix]):
            anchor_value = semantic[anchor["path"]][attr]
            if anchor_value == "unknown":
                continue
            exact = metadata_index["attribute"].get((attr, anchor_value), 0)
            if target_overlap[attr_idx]:
                mask &= exact
            else:
                mask &= ~exact
        mask &= ~excluded
        candidate = _choose_bit(mask, anchor["path"], slot, len(records))
        if candidate is not None:
            chosen = candidate
            used = prefix
            break
    if chosen is None:
        # This fallback remains different-ID and is deterministic.  It should
        # be unreachable on PRCC but is retained as an explicit audit state.
        mask = metadata_index["all"] & ~excluded
        chosen = _choose_bit(mask, anchor["path"], slot, len(records))
        used = -1
    levels = {
        4: "anchor_camera+negative_camera+negative_clothes_state+gender+hair_color+hair_length+body_build",
        3: "anchor_camera+negative_camera+negative_clothes_state+gender+hair_color+hair_length",
        2: "anchor_camera+negative_camera+negative_clothes_state+gender+hair_color",
        1: "anchor_camera+negative_camera+negative_clothes_state+gender",
        0: "anchor_camera+negative_camera+negative_clothes_state",
        -1: "different_id_only",
    }
    return chosen, levels[used]


def jaccard(a, b):
    a, b = set(a), set(b)
    return float(len(a & b)) / float(len(a | b)) if a | b else 1.0


def build_graph(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records, pid_strings, clothes_keys, _ = load_train_records()
    if len(records) != 17896 or len(pid_strings) != 150:
        raise RuntimeError("Unexpected train inventory {} / {}".format(len(records), len(pid_strings)))
    semantic = load_p2_cache(P2_CACHE, records)
    if os.path.exists(P2_VALIDATION):
        with open(P2_VALIDATION) as handle:
            validation = json.load(handle)
        if not validation.get("passed", True):
            raise RuntimeError("Existing P2 cache validation report is not passed")
    else:
        validation = {"path": P2_VALIDATION, "available": False}
    checkpoint_hash = sha256_file(checkpoint_path)
    provenance_path = os.path.join(REPO_ROOT, "reports", "v0_mining_checkpoint_provenance.json")
    if not os.path.exists(provenance_path):
        raise RuntimeError("V0 provenance is missing")
    with open(provenance_path) as handle:
        provenance = json.load(handle)
    if provenance.get("checkpoint_sha256") != checkpoint_hash or \
            provenance.get("epoch") != 50 or not provenance.get("train_only") or \
            provenance.get("test_adaptive_checkpoint_selection") is not False:
        raise RuntimeError("V0 provenance/checkpoint gate failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RCHRL-V1 requires CUDA for V0 feature mining")
    model, checkpoint = load_model(checkpoint_path, device)
    t0 = time.time()
    visual_features = extract_visual_features(model, records, device)
    semantic_features, semantic_meta = extract_semantic_features(model, records, semantic, device)
    visual_features = _normalise_rows(visual_features)
    np.save(os.path.join(output_dir, "visual_features.npy"), visual_features)
    np.save(os.path.join(output_dir, "semantic_features.npy"), semantic_features)

    manifest_rows = []
    for row in records:
        manifest_rows.append(_metadata(row, semantic))
    jsonl_write(manifest_rows, os.path.join(output_dir, "train_image_manifest.jsonl"))

    visual_idx, visual_scores = mine_topk(visual_features, records, "visual", device)
    semantic_idx, semantic_scores = mine_topk(semantic_features, records, "semantic", device)

    pids = np.asarray([row["person_id"] for row in records], dtype=np.int32)
    visual_edges = []
    semantic_edges = []
    hybrid_edges = []
    visual_sets, semantic_sets, hybrid_sets = [], [], []
    for anchor_idx in range(len(records)):
        vset = set(int(x) for x in visual_idx[anchor_idx])
        sset = set(int(x) for x in semantic_idx[anchor_idx])
        visual_sets.append(vset)
        semantic_sets.append(sset)
        for rank, neg_idx in enumerate(visual_idx[anchor_idx]):
            visual_edges.append(edge_record(anchor_idx, int(neg_idx), records,
                                             visual_scores[anchor_idx, rank],
                                             float(np.dot(semantic_features[anchor_idx], semantic_features[neg_idx])),
                                             visual_rank=rank + 1))
        for rank, neg_idx in enumerate(semantic_idx[anchor_idx]):
            semantic_edges.append(edge_record(anchor_idx, int(neg_idx), records,
                                               float(np.dot(visual_features[anchor_idx], visual_features[neg_idx])),
                                               semantic_scores[anchor_idx, rank],
                                               semantic_rank=rank + 1))
        union = sorted(vset | sset)
        hybrid_raw = {}
        for neg_idx in union:
            vrank = np.flatnonzero(visual_idx[anchor_idx] == neg_idx)
            srank = np.flatnonzero(semantic_idx[anchor_idx] == neg_idx)
            hv = 1.0 - float(vrank[0]) / float(K - 1) if len(vrank) else 0.0
            hs = 1.0 - float(srank[0]) / float(K - 1) if len(srank) else 0.0
            hybrid_raw[int(neg_idx)] = 0.5 * hv + 0.5 * hs
        hybrid_order = sorted(union, key=lambda idx: (-hybrid_raw[idx], stable_hash(records[idx]["path"])))[:K]
        hybrid_sets.append(set(hybrid_order))
        for rank, neg_idx in enumerate(hybrid_order):
            visual_cos = float(np.dot(visual_features[anchor_idx], visual_features[neg_idx]))
            semantic_cos = float(np.dot(semantic_features[anchor_idx], semantic_features[neg_idx]))
            row = edge_record(anchor_idx, neg_idx, records, visual_cos, semantic_cos,
                              visual_rank=(int(np.flatnonzero(visual_idx[anchor_idx] == neg_idx)[0]) + 1
                                           if neg_idx in vset else None),
                              semantic_rank=(int(np.flatnonzero(semantic_idx[anchor_idx] == neg_idx)[0]) + 1
                                             if neg_idx in sset else None),
                              hybrid_rank=rank + 1)
            row["H_hybrid_raw"] = float(hybrid_raw[neg_idx])
            hybrid_edges.append(row)

    # Build integer bitset indexes for deterministic pseudo-random controls.
    # They make the 20-control match per anchor cheap without scanning all
    # 17,896 images for every relation.
    metadata_index = {"all": ((1 << len(records)) - 1), "camera_state": defaultdict(int),
                      "attribute": defaultdict(int), "person": defaultdict(int)}
    for idx, row in enumerate(records):
        bit = 1 << idx
        metadata_index["camera_state"][(row["camera"], row["clothes_state"])] |= bit
        metadata_index["person"][row["person_id"]] = metadata_index["person"].get(row["person_id"], 0) | bit
        for attr in ATTRIBUTES:
            value = semantic[row["path"]][attr]
            if value != "unknown":
                metadata_index["attribute"][(attr, value)] |= bit

    matched_edges = []
    matched_indices = np.zeros((len(records), K), dtype=np.int32)
    match_levels = Counter()
    for anchor_idx in range(len(records)):
        anchor = records[anchor_idx]
        hybrid_order = [row["negative_index"] for row in hybrid_edges[anchor_idx * K:(anchor_idx + 1) * K]]
        chosen_controls = set()
        for slot, target_idx in enumerate(hybrid_order):
            chosen, level = choose_matched_negative(anchor_idx, target_idx, slot, records,
                                                     semantic, metadata_index, chosen_controls)
            chosen_controls.add(chosen)
            matched_indices[anchor_idx, slot] = chosen
            match_levels[level] += 1
            vcos = float(np.dot(visual_features[anchor_idx], visual_features[chosen]))
            scos = float(np.dot(semantic_features[anchor_idx], semantic_features[chosen]))
            matched_edges.append(edge_record(anchor_idx, chosen, records, vcos, scos,
                                              hybrid_rank=slot + 1, match_level=level))

    graph_path = output_dir
    jsonl_write(visual_edges, os.path.join(graph_path, "visual_hard_negative_edges.jsonl"))
    jsonl_write(semantic_edges, os.path.join(graph_path, "semantic_hard_negative_edges.jsonl"))
    jsonl_write(hybrid_edges, os.path.join(graph_path, "hybrid_hard_negative_edges.jsonl"))
    jsonl_write(matched_edges, os.path.join(graph_path, "matched_random_negative_edges.jsonl"))

    positive_edges = []
    positive_by_anchor = [[] for _ in records]
    indices_by_pid = defaultdict(list)
    for idx, row in enumerate(records):
        indices_by_pid[row["person_id"]].append(idx)
    for anchor_idx, anchor in enumerate(records):
        if anchor["camera"] not in ("A", "B", "C"):
            continue
        for positive_idx in indices_by_pid[anchor["person_id"]]:
            positive = records[positive_idx]
            if positive_idx == anchor_idx or positive["person_id"] != anchor["person_id"]:
                continue
            if {anchor["camera"], positive["camera"]} not in ({"A", "C"}, {"B", "C"}):
                continue
            row = positive_edge_record(anchor_idx, positive_idx, records, semantic)
            positive_by_anchor[anchor_idx].append(len(positive_edges))
            positive_edges.append(row)
    jsonl_write(positive_edges, os.path.join(graph_path, "positive_edges.jsonl"))

    offsets = [0]
    positive_pos = []
    positive_conf = []
    positive_agr = []
    positive_joint = []
    for indices in positive_by_anchor:
        for edge_idx in indices:
            row = positive_edges[edge_idx]
            positive_pos.append(row["positive_index"])
            positive_conf.append(row["R_conf"])
            positive_agr.append(row["R_agr"])
            positive_joint.append(row["R_joint"])
        offsets.append(len(positive_pos))

    def scalar_matrix(rows, key):
        return np.asarray([row[key] for row in rows], dtype=np.float32).reshape(len(records), K)

    np.savez_compressed(
        os.path.join(graph_path, "relation_index.npz"),
        positive_offsets=np.asarray(offsets, dtype=np.int64),
        positive_pos=np.asarray(positive_pos, dtype=np.int32),
        positive_r_conf=np.asarray(positive_conf, dtype=np.float32),
        positive_r_agr=np.asarray(positive_agr, dtype=np.float32),
        positive_r_joint=np.asarray(positive_joint, dtype=np.float32),
        visual_neg=visual_idx.astype(np.int32), semantic_neg=semantic_idx.astype(np.int32),
        hybrid_neg=np.asarray([[row["negative_index"] for row in hybrid_edges[i * K:(i + 1) * K]]
                              for i in range(len(records))], dtype=np.int32),
        matched_neg=matched_indices,
        visual_cosine=visual_scores.astype(np.float32),
        semantic_cosine=semantic_scores.astype(np.float32),
        hybrid_cosine=np.asarray([[hybrid_edges[i * K + j]["visual_cosine"] for j in range(K)]
                                  for i in range(len(records))], dtype=np.float32),
        hybrid_semantic_cosine=np.asarray([[hybrid_edges[i * K + j]["semantic_cosine"] for j in range(K)]
                                           for i in range(len(records))], dtype=np.float32),
    )

    # Aggregate reliability and graph distributions.  Directed positive edges
    # are retained in the count because the sampler uses directed anchors.
    reliability_stats = {
        "R_conf": distribution(positive_conf),
        "R_agr": distribution(positive_agr),
        "R_joint": distribution(positive_joint),
        "by_identity": {},
        "collapse_warning_threshold": 0.80,
    }
    by_identity = defaultdict(lambda: defaultdict(list))
    for row in positive_edges:
        for key in ("R_conf", "R_agr", "R_joint"):
            by_identity[str(row["person_id"])][key].append(row[key])
    for pid, values in by_identity.items():
        reliability_stats["by_identity"][pid] = {key: distribution(vals)
                                                   for key, vals in values.items()}
    reliability_stats["collapse_warning"] = {
        key: bool(stat["unique_count"] == 1 or
                  (max(Counter(values).values()) / float(len(values)) > .80 if values else False))
        for key, stat, values in [
            ("R_conf", reliability_stats["R_conf"], positive_conf),
            ("R_agr", reliability_stats["R_agr"], positive_agr),
            ("R_joint", reliability_stats["R_joint"], positive_joint),
        ]
    }

    overlap_stats = {
        "visual_semantic": float(np.mean([jaccard(visual_sets[i], semantic_sets[i]) for i in range(len(records))])),
        "visual_hybrid": float(np.mean([jaccard(visual_sets[i], hybrid_sets[i]) for i in range(len(records))])),
        "semantic_hybrid": float(np.mean([jaccard(semantic_sets[i], hybrid_sets[i]) for i in range(len(records))])),
        "semantic_not_visual_top20_rate": float(np.mean([
            len(semantic_sets[i] - visual_sets[i]) / float(K) for i in range(len(records))])),
    }
    negative_stats = {}
    for name, idx_array in (("matched_random", matched_indices), ("visual", visual_idx),
                             ("semantic", semantic_idx),
                             ("hybrid", np.asarray([[row["negative_index"] for row in hybrid_edges[i * K:(i + 1) * K]]
                                                      for i in range(len(records))]))):
        pair_camera = Counter()
        pair_clothes = Counter()
        state_counter = Counter()
        attr_counter = Counter()
        visual_cos = []
        semantic_cos = []
        for anchor_idx in range(len(records)):
            anchor = records[anchor_idx]
            for neg_idx in idx_array[anchor_idx]:
                neg = records[int(neg_idx)]
                pair_camera["{}->{}".format(anchor["camera"], neg["camera"])] += 1
                pair_clothes["{}->{}".format(anchor["clothes_state"], neg["clothes_state"])] += 1
                state_counter[neg["clothes_state"]] += 1
                for attr in ATTRIBUTES:
                    if semantic[anchor["path"]][attr] != "unknown" and \
                            semantic[neg["path"]][attr] != "unknown" and \
                            semantic[anchor["path"]][attr] == semantic[neg["path"]][attr]:
                        attr_counter[attr + "_overlap"] += 1
                visual_cos.append(float(np.dot(visual_features[anchor_idx], visual_features[int(neg_idx)])))
                semantic_cos.append(float(np.dot(semantic_features[anchor_idx], semantic_features[int(neg_idx)])))
        total = float(len(visual_cos))
        negative_stats[name] = {
            "negative_count": int(total),
            "negative_count_per_anchor": {"min": int(K), "max": int(K), "mean": float(K)},
            "camera_pair_distribution": dict(pair_camera),
            "clothes_pair_distribution": dict(pair_clothes),
            "same_clothes_negative_ratio": float(state_counter["same_clothes"] / total),
            "gender_overlap": float(attr_counter["gender_overlap"] / total),
            "hair_color_overlap": float(attr_counter["hair_color_overlap"] / total),
            "hair_length_overlap": float(attr_counter["hair_length_overlap"] / total),
            "body_build_overlap": float(attr_counter["body_build_overlap"] / total),
            "mean_visual_cosine": float(np.mean(visual_cos)),
            "mean_semantic_cosine": float(np.mean(semantic_cos)),
        }
    negative_stats["matched_random"]["match_level_distribution"] = dict(match_levels)
    semantic_stats = {
        "semantic_hard_mean_visual_cosine": negative_stats["semantic"]["mean_visual_cosine"],
        "visual_hard_mean_visual_cosine": negative_stats["visual"]["mean_visual_cosine"],
        "semantic_hard_mean_semantic_cosine": negative_stats["semantic"]["mean_semantic_cosine"],
        "visual_hard_mean_semantic_cosine": negative_stats["visual"]["mean_semantic_cosine"],
        "interpretation": "semantic-hard is a semantic-nearest different-ID set; compare its visual cosine to visual-hard and its semantic cosine to visual-hard to test whether it adds visually non-nearest semantic confusers",
        "semantic_hard_is_visually_not_nearest_but_semantically_similar": bool(
            negative_stats["semantic"]["mean_semantic_cosine"] > negative_stats["visual"]["mean_semantic_cosine"] and
            negative_stats["semantic"]["mean_visual_cosine"] < negative_stats["visual"]["mean_visual_cosine"]),
    }

    stats = {
        "experiment": "RCHRL-V1",
        "frozen": True,
        "train_only": True,
        "train_images": len(records),
        "train_ids": len(pid_strings),
        "positive_edge_count_directed": len(positive_edges),
        "positive_definition": "same person and A-C or B-C only; A-B excluded",
        "K": K,
        "hardness_definition": "per-anchor rank; rank1=1, rank20=0; path stable hash tie-break",
        "reliability": reliability_stats,
        "overlap": overlap_stats,
        "negative_sets": negative_stats,
        "semantic_confuser_analysis": semantic_stats,
        "semantic_representation": semantic_meta,
        "semantic_zero_feature_count": int(np.sum(np.linalg.norm(semantic_features, axis=1) == 0.0)),
        "semantic_zero_feature_rate": float(np.mean(np.linalg.norm(semantic_features, axis=1) == 0.0)),
        "p2_cache_validation": validation,
        "v0_checkpoint_sha256": checkpoint_hash,
        "v0_checkpoint_path": checkpoint_path,
        "test_data_used": False,
        "no_test_adaptive_checkpoint_selection_was_used_for_relation_mining": True,
        "build_seconds": time.time() - t0,
    }
    json_dump(stats, os.path.join(graph_path, "relation_graph_stats.json"))

    # Extreme audit: five ranks from each frozen mining set for the first 100
    # deterministic train anchors, retaining all requested metadata.
    audit_path = os.path.join(REPO_ROOT, "reports", "extreme_hard_negative_audit.csv")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    fields = ["set", "rank", "anchor_path", "negative_path", "anchor_person_id",
              "negative_person_id", "visual_cosine", "semantic_cosine", "hybrid_rank",
              "anchor_camera", "negative_camera", "anchor_clothes_key", "negative_clothes_key"] + \
             [x for attr in ATTRIBUTES for x in ("anchor_" + attr, "negative_" + attr)]
    with open(audit_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for anchor_idx in range(min(100, len(records))):
            for set_name, indices in (("visual", visual_idx), ("semantic", semantic_idx)):
                for rank in range(5):
                    neg_idx = int(indices[anchor_idx, rank])
                    row = edge_record(anchor_idx, neg_idx, records,
                                      float(np.dot(visual_features[anchor_idx], visual_features[neg_idx])),
                                      float(np.dot(semantic_features[anchor_idx], semantic_features[neg_idx])),
                                      hybrid_rank=None)
                    row["set"] = set_name
                    row["rank"] = rank + 1
                    for attr in ATTRIBUTES:
                        row["anchor_" + attr] = semantic[records[anchor_idx]["path"]][attr]
                        row["negative_" + attr] = semantic[records[neg_idx]["path"]][attr]
                    writer.writerow({field: row.get(field, "") for field in fields})
            rows = hybrid_edges[anchor_idx * K:(anchor_idx + 1) * K]
            for rank in range(5):
                row = dict(rows[rank])
                row["set"] = "hybrid"
                row["rank"] = rank + 1
                for attr in ATTRIBUTES:
                    row["anchor_" + attr] = semantic[records[anchor_idx]["path"]][attr]
                    row["negative_" + attr] = semantic[records[row["negative_index"]]["path"]][attr]
                writer.writerow({field: row.get(field, "") for field in fields})

    payload_files = [
        "positive_edges.jsonl", "matched_random_negative_edges.jsonl",
        "visual_hard_negative_edges.jsonl", "semantic_hard_negative_edges.jsonl",
        "hybrid_hard_negative_edges.jsonl", "relation_graph_stats.json",
        "train_image_manifest.jsonl", "relation_index.npz", "visual_features.npy",
        "semantic_features.npy",
    ]
    file_hashes = {name: sha256_file(os.path.join(graph_path, name)) for name in payload_files}
    manifest = {
        "experiment": "RCHRL-V1",
        "graph_version": "RCHRL-V1-frozen-train-graph-1",
        "frozen": True,
        "train_only": True,
        "source_commit": repo_commit(REPO_ROOT),
        "train_root": TRAIN_ROOT,
        "train_image_count": len(records),
        "train_id_count": len(pid_strings),
        "p2_cache_path": P2_CACHE,
        "p2_cache_sha256": sha256_file(P2_CACHE),
        "p2_normalization_rules_path": P2_RULES,
        "p2_normalization_rules_sha256": sha256_file(P2_RULES),
        "v0_provenance_path": provenance_path,
        "v0_provenance_sha256": sha256_file(provenance_path),
        "v0_checkpoint_path": checkpoint_path,
        "v0_checkpoint_sha256": checkpoint_hash,
        "v0_checkpoint_epoch": 50,
        "v0_training_seed": provenance.get("training_seed"),
        "no_test_adaptive_checkpoint_selection_was_used_for_relation_mining": True,
        "test_data_used": False,
        "positive_definition": "same-ID A-C and B-C directed relations; A-B excluded",
        "semantic_allowed_attributes": list(ATTRIBUTES),
        "semantic_disallowed_fields": ["S2", "S4", "raw_model_output", "viewpoint", "face_visibility",
                                        "occlusion", "image_quality", "upper_clothing", "lower_clothing",
                                        "shoes", "accessories"],
        "semantic_serialization": semantic_meta,
        "visual_feature_definition": "V0 model encode_image projected CLS; L2 normalized; no flip",
        "hardness_definition": "per-anchor rank normalization K=20; stable path hash tie-break",
        "hybrid_definition": "0.5 H_visual_rank + 0.5 H_sem_rank over visual/semantic union, final rank top20",
        "matched_random_definition": "different-ID controls matched hierarchically to hybrid negative camera, clothes state, and P2 overlap indicators",
        "payload_sha256": file_hashes,
    }
    manifest_path = os.path.join(graph_path, "relation_graph_manifest.json")
    json_dump(manifest, manifest_path)
    ledger_files = dict(file_hashes)
    ledger_files["relation_graph_manifest.json"] = sha256_file(manifest_path)
    ledger_path = os.path.join(graph_path, "relation_graph_hashes.json")
    json_dump({"frozen": True, "files": ledger_files,
               "ledger_sha256_sidecar": "relation_graph_hashes.sha256"}, ledger_path)
    with open(os.path.join(graph_path, "relation_graph_hashes.sha256"), "w") as handle:
        handle.write("{}  relation_graph_hashes.json\n".format(sha256_file(ledger_path)))
    print(json.dumps({"status": "complete", "graph_dir": graph_path,
                      "positive_edges": len(positive_edges), "negative_edges_per_set": len(records) * K,
                      "elapsed_seconds": time.time() - t0}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_graph(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
