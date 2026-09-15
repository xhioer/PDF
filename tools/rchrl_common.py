"""Common, auditable utilities for RCHRL-V1.

This module deliberately keeps the research additions outside the frozen PDF
implementation.  In particular, it contains no test-set discovery and no
model/inference changes.
"""
from __future__ import absolute_import

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/data/datasets/PRCC"
TRAIN_ROOT = os.path.join(DATA_ROOT, "prcc", "rgb", "train")
P2_CACHE = "/data/projects/ccreid-semantic-consistency-exp0/outputs/exp1_full_prcc_mllm/qwen3vl32b_p2_17896images.jsonl"
P2_RULES = "/data/projects/ccreid-semantic-consistency-exp0/configs/normalization_rules_v2.json"
P2_VALIDATION = "/data/projects/PDF-worktrees/pdf-reliability-ablation/reports/prcc_semantic_cache_validation.json"
# Confirmatory runs may pin the original PDF caption file by absolute path so
# the protocol snapshot is byte-for-byte comparable across worktrees.  The
# default remains the local repository copy for ordinary RCHRL-V1 runs.
ORIGINAL_CAPTION = os.environ.get(
    "RCHRL_ORIGINAL_CAPTION",
    os.path.join(REPO_ROOT, "data", "captions", "prcc.json"),
)
LOCAL_IO_ROOT = os.environ.get("RCHRL_LOCAL_PRCC_ROOT", "")
ATTRIBUTES = ("gender", "hair_color", "hair_length", "body_build")
CONFIDENCE_VALUES = {"high": 1.0, "medium": 0.5, "low": 0.0, "unknown": 0.0}
CAMERA_TO_INT = {"A": 0, "B": 1, "C": 2}
K = 20
EPS = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def json_dump(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)


def jsonl_write(rows, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def normalize_global(value):
    if value is None:
        return "unknown"
    value = str(value).lower().strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value)


def load_normalization_rules(path=P2_RULES):
    with open(path) as handle:
        rules = json.load(handle)
    invalid = set(normalize_global(x) for x in rules["global"]["invalid_tokens"])
    mappings = {}
    for attr, values in rules["attribute_rules"].items():
        mappings[attr] = {normalize_global(k): normalize_global(v) for k, v in values.items()}
    return invalid, mappings


_INVALID_TOKENS, _ATTRIBUTE_MAPPINGS = load_normalization_rules()


def normalize_attribute(attribute, value):
    value = normalize_global(value)
    if value in _INVALID_TOKENS:
        return "unknown"
    return _ATTRIBUTE_MAPPINGS.get(attribute, {}).get(value, value)


def normalize_confidence(attribute, value):
    # The frozen rules use one mapping per attribute confidence field.
    key = attribute + "_confidence"
    value = normalize_global(value)
    if value in _INVALID_TOKENS:
        return "low"
    return _ATTRIBUTE_MAPPINGS.get(key, {}).get(value, "low")


def p2_phrase(attribute, value):
    if value == "unknown":
        raise ValueError("unknown values must not be serialized")
    return "A person with {}: {}.".format(attribute.replace("_", " "), value)


def projected_image_cls(model, images):
    """Return the projected ViT CLS feature used by the frozen PDF loss.

    This repository's ``CLIP.encode_image`` helper predates the local
    VisionTransformer return signature (``(projected_tokens, tokens)``), so
    calling it directly would attempt tuple indexing.  Keeping this adapter
    in the RCHRL tools avoids modifying the frozen model/inference files and
    makes the relation feature definition explicit and auditable.
    """
    visual_output = model.visual(images.type(model.dtype))
    if isinstance(visual_output, (tuple, list)):
        visual_output = visual_output[0]
    if visual_output.dim() == 3:
        visual_output = visual_output[:, 0, :]
    if visual_output.dim() != 2:
        raise RuntimeError("unexpected projected image feature shape: {}".format(
            tuple(visual_output.shape)))
    return visual_output.float()


def _canonical_path(path):
    # Do not call Path.resolve() here: the PRCC mount is an object-backed
    # filesystem and resolving every image causes avoidable metadata walks.
    return os.path.normpath(str(path))


def image_io_path(path):
    """Map only image reads to an optional local byte-for-byte staging copy."""
    path = _canonical_path(path)
    if LOCAL_IO_ROOT and path.startswith(DATA_ROOT + os.sep):
        return _canonical_path(LOCAL_IO_ROOT + path[len(DATA_ROOT):])
    return path


def _train_clothes_key(pid_string, camera):
    return pid_string if camera in ("A", "B") else pid_string + camera


def load_train_records(train_root=TRAIN_ROOT):
    """Reproduce PRCC's train parsing without touching val or test.

    The cache validation already establishes a one-to-one mapping between the
    17,896 cache paths and PRCC TRAIN.  Reading those path strings avoids a
    very slow per-directory object-store walk; no semantic cache field is
    inspected here.
    """
    cached_paths = []
    if os.path.exists(P2_CACHE):
        with open(P2_CACHE) as handle:
            for line in handle:
                row = json.loads(line)
                path = _canonical_path(row["image_path"])
                if not path.startswith(_canonical_path(train_root) + os.sep):
                    raise RuntimeError("Non-train path encountered in train inventory: {}".format(path))
                cached_paths.append(path)
    if len(cached_paths) == 17896 and len(set(cached_paths)) == 17896:
        raw_paths = sorted(cached_paths)
        pid_strings = sorted(set(os.path.basename(os.path.dirname(path)) for path in raw_paths),
                             key=lambda x: int(x))
        person_dirs = [os.path.join(train_root, name) for name in pid_strings]
    else:
        person_dirs = sorted(
            os.path.join(train_root, name)
            for name in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, name))
        )
        pid_strings = [os.path.basename(path) for path in person_dirs]
        pid_strings.sort(key=lambda x: int(x))
    pid_strings.sort(key=lambda x: int(x))
    pid_to_label = {pid: idx for idx, pid in enumerate(pid_strings)}

    clothes_keys = set()
    raw = []
    cached_paths_by_person = defaultdict(list)
    if cached_paths:
        for path in raw_paths:
            cached_paths_by_person[os.path.dirname(path)].append(path)
    for person_dir in person_dirs:
        pid_string = os.path.basename(person_dir)
        if cached_paths:
            paths = cached_paths_by_person.get(_canonical_path(person_dir), [])
        else:
            paths = sorted(os.path.join(person_dir, name) for name in os.listdir(person_dir)
                           if name.lower().endswith(".jpg"))
        for path in paths:
            name = os.path.basename(path)
            camera = name[0]
            if camera not in CAMERA_TO_INT:
                raise ValueError("Unexpected PRCC train camera in {}".format(path))
            clothes_key = _train_clothes_key(pid_string, camera)
            clothes_keys.add(clothes_key)
            raw.append((path, pid_string, camera, clothes_key))
    clothes_keys = sorted(clothes_keys)
    clothes_to_label = {name: idx for idx, name in enumerate(clothes_keys)}

    records = []
    pid2clothes = np.zeros((len(pid_strings), len(clothes_keys)), dtype=np.float32)
    for path, pid_string, camera, clothes_key in raw:
        pid = pid_to_label[pid_string]
        clothes_id = clothes_to_label[clothes_key]
        records.append({
            "index": len(records),
            "path": _canonical_path(path),
            "person_id": pid,
            "person_id_raw": pid_string,
            "camera": camera,
            "camera_id": CAMERA_TO_INT[camera],
            "clothes_id": clothes_id,
            "clothes_key": clothes_key,
            "clothes_state": "same_clothes" if camera in ("A", "B") else "different_clothes",
        })
        pid2clothes[pid, clothes_id] = 1.0
    records.sort(key=lambda x: x["path"])
    for idx, row in enumerate(records):
        row["index"] = idx
    return records, pid_strings, clothes_keys, pid2clothes


def load_p2_cache(cache_path=P2_CACHE, train_records=None):
    """Read only the four permitted P2 fields and their confidence fields."""
    if train_records is None:
        train_records, _, _, _ = load_train_records()
    expected = {_canonical_path(row["path"]): row for row in train_records}
    result = {}
    total = 0
    bad = []
    with open(cache_path) as handle:
        for line_no, line in enumerate(handle, 1):
            total += 1
            try:
                row = json.loads(line)
                image_path = _canonical_path(row["image_path"])
                description = row["description"]
                if image_path not in expected:
                    bad.append((line_no, "extra_path", image_path))
                    continue
                # Deliberately do not access raw_model_output, S2/S4, or any
                # other semantic field.
                values = {}
                for attr in ATTRIBUTES:
                    values[attr] = normalize_attribute(attr, description.get(attr))
                    values[attr + "_confidence"] = normalize_confidence(
                        attr, description.get(attr + "_confidence"))
                result[image_path] = values
            except (TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
                bad.append((line_no, "parse", str(exc)))
    missing = sorted(set(expected) - set(result))
    duplicate_count = total - len(result) - len(bad) - len(missing)
    if total != len(expected) or missing or bad or duplicate_count != 0:
        raise RuntimeError("P2 train-only cache mapping failed: total={} expected={} missing={} bad={} duplicate={}".format(
            total, len(expected), len(missing), len(bad), duplicate_count))
    return result


def reliability_pair(a, b):
    """Return R_conf, R_agr, R_joint for one positive edge."""
    conf_values = []
    agreement = []
    for attr in ATTRIBUTES:
        ca = CONFIDENCE_VALUES.get(a.get(attr + "_confidence", "low"), 0.0)
        cb = CONFIDENCE_VALUES.get(b.get(attr + "_confidence", "low"), 0.0)
        conf_values.append(math.sqrt(ca * cb))
        va = a.get(attr, "unknown")
        vb = b.get(attr, "unknown")
        if va != "unknown" and vb != "unknown":
            agreement.append(1.0 if va == vb else 0.0)
    r_conf = float(np.mean(conf_values)) if conf_values else 0.0
    r_agr = float(np.mean(agreement)) if agreement else 0.0
    return r_conf, r_agr, float(np.clip(r_conf * r_agr, 0.0, 1.0))


def attr_overlap(a, b):
    result = {}
    for attr in ATTRIBUTES:
        va, vb = a.get(attr, "unknown"), b.get(attr, "unknown")
        result[attr + "_overlap"] = bool(va != "unknown" and vb != "unknown" and va == vb)
    return result


def distribution(values, bins=10):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "q1": 0.0,
                "q3": 0.0, "min": 0.0, "max": 0.0, "zero_rate": 0.0,
                "unique_count": 0, "histogram": {}}
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = {"[{:.3f},{:.3f})".format(edges[i], edges[i + 1]): int(counts[i])
                 for i in range(len(counts))}
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
        "q3": float(np.quantile(values, .75)), "min": float(values.min()),
        "max": float(values.max()), "zero_rate": float(np.mean(values == 0.0)),
        "unique_count": int(np.unique(values).size), "histogram": histogram,
    }


def feature_transform(config):
    from data.img_transforms import Compose, Resize, ToTensor, Normalize
    return Compose([
        Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(image_io_path(row["path"])) as image:
            image = image.convert("RGB")
        return self.transform(image), index


class TrainCaptionDataset(Dataset):
    """Original PDF caption mechanism, with train-only path mapping."""
    def __init__(self, records, transform, caption_path=ORIGINAL_CAPTION):
        self.records = records
        self.transform = transform
        with open(caption_path) as handle:
            payload = json.load(handle)
        self.captions = {}
        for key, values in payload.items():
            self.captions[_canonical_path(key)] = values[0]
        # The frozen caption JSON uses data/prcc/... keys, while the dataset is
        # mounted at /data/datasets/PRCC/prcc/....
        for row in records:
            marker = "/prcc/"
            pos = row["path"].find(marker)
            if pos < 0:
                raise ValueError("Cannot map PRCC caption path: {}".format(row["path"]))
            key = "data/prcc/" + row["path"][pos + len(marker):]
            if key not in self.captions:
                raise KeyError("Caption missing for {}".format(key))
            row["caption_key"] = key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(image_io_path(row["path"])) as image:
            image = image.convert("RGB")
        return (self.transform(image), row["person_id"], row["camera_id"],
                row["clothes_id"], self.captions[row["caption_key"]])


def make_train_tuple(row, semantic):
    values = {attr: semantic[row["path"]][attr] for attr in ATTRIBUTES}
    return values


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def config_snapshot(config):
    return config.dump() if hasattr(config, "dump") else str(config)


def repo_commit(cwd=REPO_ROOT):
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()


def source_hashes(cwd=REPO_ROOT):
    paths = ["train.py", "models/clip_model.py", "test.py", "configs/default_img.py",
             "data/__init__.py", "data/dataset_loader.py", "losses/contrastive_loss.py",
             "losses/orthogonal_loss.py"]
    return {path: sha256_file(os.path.join(cwd, path)) for path in paths}
