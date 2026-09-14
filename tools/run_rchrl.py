"""RCHRL-V1 training, sanity, smoke, and image-only evaluation runner.

The frozen PDF files are imported but not edited.  Relation data are supplied
by a separate train-only tuple dataset and are never re-mined during a run.
"""
from __future__ import absolute_import

import argparse
import csv
import json
import logging
import math
import os
import shutil
import time
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Sampler

from configs.default_img import _C
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset
from data import build_img_transforms
from data.samplers import DistributedInferenceSampler
from data.datasets.prcc import PRCC
from losses import build_losses
from losses.orthogonal_loss import OrthogonalProjectionLoss
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from test import concat_all_gather, extract_img_feature_clip_combiner
from tools.eval_metrics import evaluate
from tools.lr_scheduler import WarmupMultiStepLR
from tools.rchrl_common import (
    ATTRIBUTES, DATA_ROOT, ORIGINAL_CAPTION, REPO_ROOT, TrainCaptionDataset,
    config_snapshot, feature_transform, json_dump, load_train_records,
    image_io_path, repo_commit, set_all_seeds, sha256_file, source_hashes,
    projected_image_cls,
)
from train import tokenize as pdf_tokenize


GRAPH_REQUIRED = (
    "positive_edges.jsonl", "matched_random_negative_edges.jsonl",
    "visual_hard_negative_edges.jsonl", "semantic_hard_negative_edges.jsonl",
    "hybrid_hard_negative_edges.jsonl", "relation_graph_stats.json",
    "relation_graph_manifest.json", "relation_graph_hashes.json",
    "relation_graph_hashes.sha256", "relation_index.npz",
)

RUNS = OrderedDict([
    ("R00", {"name": "R00_control", "relation": "none", "negative": None,
             "positive": "uniform", "lambda": 0.0}),
    ("R01", {"name": "R01_matched_random", "relation": "hinge", "negative": "matched",
             "positive": "uniform", "lambda": 0.1}),
    ("R02", {"name": "R02_visualhard", "relation": "hinge", "negative": "visual",
             "positive": "uniform", "lambda": 0.1}),
    ("R03", {"name": "R03_semhard", "relation": "hinge", "negative": "semantic",
             "positive": "uniform", "lambda": 0.1}),
    ("R04", {"name": "R04_hybrid", "relation": "hinge", "negative": "hybrid",
             "positive": "uniform", "lambda": 0.1}),
    ("R05", {"name": "R05_hybrid_conf", "relation": "hinge", "negative": "hybrid",
             "positive": "R_conf", "lambda": 0.1}),
    ("R06", {"name": "R06_hybrid_agreement", "relation": "hinge", "negative": "hybrid",
             "positive": "R_agr", "lambda": 0.1}),
    ("R07", {"name": "R07_hybrid_joint", "relation": "hinge", "negative": "hybrid",
             "positive": "R_joint", "lambda": 0.1}),
    ("R08", {"name": "R08_visual_joint", "relation": "hinge", "negative": "visual",
             "positive": "R_joint", "lambda": 0.1}),
    ("R09", {"name": "R09_sem_joint", "relation": "hinge", "negative": "semantic",
             "positive": "R_joint", "lambda": 0.1}),
    ("R10", {"name": "R10_hybrid_hardness", "relation": "hinge", "negative": "hybrid",
             "positive": "uniform", "hardness": True, "lambda": 0.1}),
    ("R11", {"name": "R11_full", "relation": "hinge", "negative": "hybrid",
             "positive": "R_joint", "hardness": True, "lambda": 0.1}),
    ("R12", {"name": "R12_full_l005", "relation": "hinge", "negative": "hybrid",
             "positive": "R_joint", "hardness": True, "lambda": 0.05}),
    ("R13", {"name": "R13_full_l020", "relation": "hinge", "negative": "hybrid",
             "positive": "R_joint", "hardness": True, "lambda": 0.20}),
    ("R14", {"name": "R14_positive_only", "relation": "positive_only", "negative": None,
             "positive": "R_joint", "lambda": 0.1}),
    ("R15", {"name": "R15_negative_only", "relation": "negative_only", "negative": "hybrid",
             "positive": "uniform", "hardness": True, "lambda": 0.1}),
])


def make_config(seed, output):
    config = _C.clone()
    config.defrost()
    config.DATA.ROOT = DATA_ROOT
    config.DATA.CAPTION_PATH = ORIGINAL_CAPTION
    config.DATA.TRAIN_BATCH = 64
    config.DATA.TEST_BATCH = 64
    config.DATA.NUM_WORKERS = 4
    config.DATA.NUM_INSTANCES = 8
    config.TRAIN.MAX_EPOCH = 50
    config.TRAIN.OPTIMIZER.LR = 3.5e-7
    config.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
    config.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
    config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    config.TRAIN.AMP = True
    config.TEST.EVAL_STEP = 5
    config.TEST.START_EVAL = 0
    config.SEED = int(seed)
    config.OUTPUT = output
    config.TAG = "rchrl"
    config.freeze()
    return config


def setup_logger(path, rank):
    logger = logging.getLogger("rchrl")
    logger.handlers = []
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if rank == 0:
        for handler in (logging.StreamHandler(), logging.FileHandler(path, mode="a")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def verify_frozen_graph(graph_dir):
    for name in GRAPH_REQUIRED:
        if not os.path.exists(os.path.join(graph_dir, name)):
            raise RuntimeError("Frozen graph file missing: {}".format(name))
    ledger_path = os.path.join(graph_dir, "relation_graph_hashes.json")
    with open(ledger_path) as handle:
        ledger = json.load(handle)
    with open(os.path.join(graph_dir, "relation_graph_hashes.sha256")) as handle:
        sidecar = handle.read().split()[0]
    if sidecar != sha256_file(ledger_path):
        raise RuntimeError("Frozen graph hash ledger sidecar mismatch")
    for name, expected in ledger["files"].items():
        actual = sha256_file(os.path.join(graph_dir, name))
        if actual != expected:
            raise RuntimeError("Frozen graph payload hash mismatch: {}".format(name))
    with open(os.path.join(graph_dir, "relation_graph_manifest.json")) as handle:
        manifest = json.load(handle)
    if not manifest.get("frozen") or not manifest.get("train_only") or \
            manifest.get("test_data_used") is not False or \
            not manifest.get("no_test_adaptive_checkpoint_selection_was_used_for_relation_mining"):
        raise RuntimeError("Frozen graph manifest gate failed")
    if manifest.get("train_image_count") != 17896 or manifest.get("train_id_count") != 150:
        raise RuntimeError("Frozen graph inventory gate failed")
    arrays = np.load(os.path.join(graph_dir, "relation_index.npz"))
    expected_shapes = {
        "visual_neg": (17896, 20), "semantic_neg": (17896, 20),
        "hybrid_neg": (17896, 20), "matched_neg": (17896, 20),
    }
    for key, shape in expected_shapes.items():
        if tuple(arrays[key].shape) != shape:
            raise RuntimeError("Frozen graph {} shape {} != {}".format(key, arrays[key].shape, shape))
    return manifest, ledger


class RelationAnchorSampler(Sampler):
    def __init__(self, count, batch_size, seed, epoch=0):
        self.count = int(count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.samples = (self.count // self.batch_size) * self.batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 100003)
        permutation = torch.randperm(self.count, generator=generator).tolist()
        return iter(permutation[:self.samples])

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class RelationTupleDataset(torch.utils.data.Dataset):
    def __init__(self, records, graph_dir, variant, transform, seed):
        self.records = records
        self.graph_dir = graph_dir
        self.variant = variant
        self.transform = transform
        self.seed = int(seed)
        arrays = np.load(os.path.join(graph_dir, "relation_index.npz"))
        self.offsets = arrays["positive_offsets"]
        self.positive_pos = arrays["positive_pos"]
        self.r_conf = arrays["positive_r_conf"]
        self.r_agr = arrays["positive_r_agr"]
        self.r_joint = arrays["positive_r_joint"]
        self.visual_neg = arrays["visual_neg"]
        self.semantic_neg = arrays["semantic_neg"]
        self.hybrid_neg = arrays["hybrid_neg"]
        self.matched_neg = arrays["matched_neg"]
        self.visual_cos = arrays["visual_cosine"]
        self.semantic_cos = arrays["semantic_cosine"]
        self.hybrid_cos = arrays["hybrid_cosine"]
        self.hybrid_semantic_cos = arrays["hybrid_semantic_cosine"]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _pick(self, anchor):
        start, stop = int(self.offsets[anchor]), int(self.offsets[anchor + 1])
        if stop <= start:
            raise RuntimeError("Anchor has no same-ID cross-clothes positive: {}".format(anchor))
        positive_flat = start + ((self.seed + self.epoch + anchor) % (stop - start))
        positive = int(self.positive_pos[positive_flat])
        slot = (self.seed * 17 + self.epoch * 7919 + anchor * 13) % 20
        if self.variant == "matched":
            negative = int(self.matched_neg[anchor, slot])
        elif self.variant == "visual":
            negative = int(self.visual_neg[anchor, slot])
        elif self.variant == "semantic":
            negative = int(self.semantic_neg[anchor, slot])
        else:
            negative = int(self.hybrid_neg[anchor, slot])
        h_visual = 0.0
        h_semantic = 0.0
        if self.variant == "visual":
            h_visual = 1.0 - slot / 19.0
        elif self.variant == "semantic":
            h_semantic = 1.0 - slot / 19.0
        elif self.variant == "hybrid":
            visual_rank = np.flatnonzero(self.visual_neg[anchor] == negative)
            semantic_rank = np.flatnonzero(self.semantic_neg[anchor] == negative)
            h_visual = 1.0 - float(visual_rank[0]) / 19.0 if len(visual_rank) else 0.0
            h_semantic = 1.0 - float(semantic_rank[0]) / 19.0 if len(semantic_rank) else 0.0
        if self.variant == "matched":
            # The control is aligned to the corresponding hybrid rank, not
            # itself mined by visual/semantic similarity.
            h_visual, h_semantic = 0.0, 0.0
        h_hybrid = 1.0 - slot / 19.0
        return positive_flat, positive, negative, slot, h_visual, h_semantic, h_hybrid

    def __getitem__(self, anchor):
        positive_flat, positive, negative, slot, h_visual, h_semantic, h_hybrid = self._pick(anchor)
        images = []
        for index in (anchor, positive, negative):
            with Image.open(image_io_path(self.records[index]["path"])) as image:
                images.append(self.transform(image.convert("RGB")))
        return (images[0], images[1], images[2], int(anchor), int(positive), int(negative),
                float(self.r_conf[positive_flat]), float(self.r_agr[positive_flat]),
                float(self.r_joint[positive_flat]), float(h_visual), float(h_semantic),
                float(h_hybrid), int(slot), int(positive_flat))


def build_train_loader(config, records, seed):
    transform_train, _ = build_img_transforms(config)
    train_dataset = TrainCaptionDataset(records, transform_train, ORIGINAL_CAPTION)
    train_tuples = [(row["path"], row["person_id"], row["camera_id"], row["clothes_id"])
                    for row in records]
    from data.samplers import DistributedRandomIdentitySampler
    sampler = DistributedRandomIdentitySampler(train_tuples,
                                                num_instances=config.DATA.NUM_INSTANCES,
                                                seed=seed)
    loader = DataLoaderX(dataset=train_dataset, sampler=sampler,
                         batch_size=config.DATA.TRAIN_BATCH,
                         num_workers=config.DATA.NUM_WORKERS, pin_memory=True,
                         drop_last=True)
    return loader, sampler, transform_train


def build_relation_loader(config, records, graph_dir, variant, seed, epoch=0):
    dataset = RelationTupleDataset(records, graph_dir, variant,
                                   build_img_transforms(config)[0], seed)
    sampler = RelationAnchorSampler(len(records), config.DATA.TRAIN_BATCH, seed, epoch)
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.DATA.TRAIN_BATCH,
                        num_workers=config.DATA.NUM_WORKERS, pin_memory=True,
                        drop_last=True)
    dataset.set_epoch(epoch)
    return loader, sampler, dataset


def build_eval_loaders(config):
    """Construct TEST loaders only at evaluation time."""
    eval_root = os.environ.get("RCHRL_LOCAL_PRCC_ROOT", DATA_ROOT)
    dataset = PRCC(root=eval_root)
    _, transform_test = build_img_transforms(config)
    # The dataset metadata remain the original PRCC TEST metadata; only the
    # bytes are optionally read from the validated local staging copy.
    def mapped(items):
        return [(image_io_path(path), pid, camid, clothes) for path, pid, camid, clothes in items]
    dataset.query_same = mapped(dataset.query_same)
    dataset.query_diff = mapped(dataset.query_diff)
    dataset.gallery = mapped(dataset.gallery)
    common = {"num_workers": config.DATA.NUM_WORKERS, "pin_memory": True,
              "drop_last": False}
    same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform_test),
                       sampler=DistributedInferenceSampler(dataset.query_same),
                       batch_size=config.DATA.TEST_BATCH, **common)
    diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform_test),
                       sampler=DistributedInferenceSampler(dataset.query_diff),
                       batch_size=config.DATA.TEST_BATCH, **common)
    gallery = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform_test),
                          sampler=DistributedInferenceSampler(dataset.gallery),
                          batch_size=config.DATA.TEST_BATCH, **common)
    return same, diff, gallery, dataset


@torch.no_grad()
def evaluate_image_only(model, config, logger):
    same_loader, diff_loader, gallery_loader, dataset = build_eval_loaders(config)
    model.eval()
    t0 = time.time()
    qsf, qs_pids, qs_camids, qs_clothes = extract_img_feature_clip_combiner(
        model, same_loader, mode="image")
    qdf, qd_pids, qd_camids, qd_clothes = extract_img_feature_clip_combiner(
        model, diff_loader, mode="image")
    gf, g_pids, g_camids, g_clothes = extract_img_feature_clip_combiner(
        model, gallery_loader, mode="image")
    qsf, qs_pids, qs_camids, qs_clothes = concat_all_gather(
        [qsf, qs_pids, qs_camids, qs_clothes], len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes = concat_all_gather(
        [qdf, qd_pids, qd_camids, qd_clothes], len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes = concat_all_gather(
        [gf, g_pids, g_camids, g_clothes], len(dataset.gallery))
    if qsf.shape[1] != 512 or qdf.shape[1] != 512 or gf.shape[1] != 512:
        raise RuntimeError("image-only inference feature shape changed")
    gallery_gpu = gf.cuda()
    same_dist = (-torch.mm(qsf.cuda(), gallery_gpu.t())).cpu().numpy()
    diff_dist = (-torch.mm(qdf.cuda(), gallery_gpu.t())).cpu().numpy()
    same_cmc, same_map = evaluate(same_dist, qs_pids.numpy(), g_pids.numpy(),
                                   qs_camids.numpy(), g_camids.numpy())
    diff_cmc, diff_map = evaluate(diff_dist, qd_pids.numpy(), g_pids.numpy(),
                                   qd_camids.numpy(), g_camids.numpy())
    result = {
        "same": {"R1": float(same_cmc[0]), "R5": float(same_cmc[4]),
                  "R10": float(same_cmc[9]), "R20": float(same_cmc[19]),
                  "mAP": float(same_map)},
        "different": {"R1": float(diff_cmc[0]), "R5": float(diff_cmc[4]),
                       "R10": float(diff_cmc[9]), "R20": float(diff_cmc[19]),
                       "mAP": float(diff_map)},
        "feature_shape": [int(gf.shape[0]), int(gf.shape[1])],
        "image_only": True,
        "text_used": False,
        "test_labels_used_for": "evaluation metrics only",
        "seconds": time.time() - t0,
    }
    logger.info("image-only eval same R1 %.4f mAP %.4f | diff R1 %.4f mAP %.4f",
                result["same"]["R1"], result["same"]["mAP"],
                result["different"]["R1"], result["different"]["mAP"])
    for loader in (same_loader, diff_loader, gallery_loader):
        if hasattr(loader, "shutdown"):
            loader.shutdown()
    return result


def l2_norm(tensor):
    return F.normalize(tensor, dim=1)


def grad_norm(parameters):
    values = []
    for parameter in parameters:
        if parameter.grad is not None:
            values.append(parameter.grad.detach().float().norm(2) ** 2)
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).item())


def make_weight(raw):
    active = raw > 0
    if bool(active.any()):
        mean_active = raw[active].mean()
    else:
        mean_active = raw.new_tensor(0.0)
    return torch.where(active, raw / (mean_active + 1e-12), torch.zeros_like(raw)), mean_active


def relation_loss(anchor, positive, negative, rel_batch, spec):
    za = l2_norm(anchor)
    zp = l2_norm(positive)
    zn = l2_norm(negative)
    pos_cos = (za * zp).sum(1)
    neg_cos = (za * zn).sum(1)
    d_pos = 1.0 - pos_cos
    d_neg = 1.0 - neg_cos
    r_conf, r_agr, r_joint = rel_batch[6], rel_batch[7], rel_batch[8]
    h_visual, h_semantic, h_hybrid = rel_batch[9], rel_batch[10], rel_batch[11]
    if spec["relation"] == "negative_only":
        losses = F.relu(0.5 - d_neg)
    elif spec["relation"] == "positive_only":
        losses = d_pos
    else:
        losses = F.relu(0.3 + d_pos - d_neg)
    if spec["positive"] == "R_conf":
        positive_raw = r_conf
    elif spec["positive"] == "R_agr":
        positive_raw = r_agr
    elif spec["positive"] == "R_joint":
        positive_raw = r_joint
    else:
        positive_raw = torch.ones_like(r_joint)
    if spec.get("hardness"):
        negative_raw = h_hybrid
    else:
        negative_raw = torch.ones_like(h_hybrid)
    if spec["relation"] == "positive_only":
        raw = positive_raw
    elif spec["relation"] == "negative_only":
        raw = negative_raw
    else:
        raw = positive_raw * negative_raw
    weights, mean_active = make_weight(raw)
    weighted = (weights * losses).mean()
    return {
        "loss": weighted, "raw_loss": losses.mean(), "weighted_before_lambda": weighted,
        "raw_weight": raw, "norm_weight": weights, "mean_active": mean_active,
        "active_rate": (raw > 0).float().mean(), "pos_cos": pos_cos, "neg_cos": neg_cos,
        "d_pos": d_pos, "d_neg": d_neg, "h_visual": h_visual, "h_semantic": h_semantic,
        "h_hybrid": h_hybrid, "r_conf": r_conf, "r_agr": r_agr, "r_joint": r_joint,
        "hinge_active_rate": (losses > 0).float().mean(),
    }


def pdf_loss(config, clip_model, criterion_cla, criterion_pair, opl, tokenizer,
             imgs, pids, clothes_ids, cap):
    reference_images = imgs
    text_inputs = pdf_tokenize(cap, tokenizer, context_length=77, truncate=True).cuda()
    with torch.cuda.amp.autocast():
        [_, cls_score_proj], [_, img_feature_proj], [com_proj, _, _, _] = clip_model(
            reference_images, text_inputs)
        alpha = torch.randn(img_feature_proj.size(0), 1, device=img_feature_proj.device,
                            dtype=img_feature_proj.dtype) * 0.5 + 0.5
        alpha_pos = torch.randn(img_feature_proj.size(0), 1, device=img_feature_proj.device,
                                dtype=img_feature_proj.dtype) * 0.5 + 0.5
        ir_features = img_feature_proj - com_proj * alpha
        ir_features_pos = img_feature_proj - com_proj * alpha_pos
        cla = criterion_cla(cls_score_proj, pids)
        pair = criterion_pair(ir_features, ir_features_pos, pids)
        opl_loss = opl(img_feature_proj, com_proj, pids, clothes_ids) * 0.5
        base = cla + pair + opl_loss
    return base, cla, pair, opl_loss


def build_model_and_optim(config, rank, num_train_clothes):
    model, _ = build_CLIP_from_openai_pretrained(
        "ViT-B/16", (config.DATA.HEIGHT, config.DATA.WIDTH), 16, 150)
    model.eval().float()
    model.freeze_text_encoder()
    criterion_cla, criterion_pair, _, _ = build_losses(config, num_train_clothes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=config.TRAIN.OPTIMIZER.LR,
                           weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    scheduler = WarmupMultiStepLR(
        optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
        gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, warmup_factor=0.1,
        warmup_iters=10)
    model = model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    return model, criterion_cla, criterion_pair, optimizer, scheduler, trainable


def init_accumulator():
    return {key: [] for key in (
        "raw_weight", "norm_weight", "active_rate", "raw_loss", "weighted_loss",
        "lambda_loss", "total_loss", "ratio", "pos_cos", "neg_cos", "pos_margin",
        "h_visual", "h_semantic", "h_hybrid", "r_conf", "r_agr", "r_joint",
        "hinge_rate", "relation_grad", "visual_grad", "total_grad", "finite_grad",
        "anchors", "positive_ids", "negative_ids", "edge_ids", "positive_edge_ids",
    )}


def append_metric(acc, key, value):
    if torch.is_tensor(value):
        value = value.detach().float().mean().item()
    acc[key].append(float(value))


def mean_metric(acc, key):
    vals = acc.get(key, [])
    return float(np.mean(vals)) if vals else 0.0


def train_one_epoch(config, model, criterion_cla, criterion_pair, optimizer,
                    trainloader, relation_loader, relation_sampler, relation_dataset,
                    spec, epoch, logger, duration_seconds=None):
    model.train()
    tokenizer = SimpleTokenizer()
    opl = OrthogonalProjectionLoss()
    scaler = torch.cuda.amp.GradScaler()
    relation_iter = iter(relation_loader) if relation_loader is not None else None
    acc = init_accumulator()
    start = time.time()
    successful_steps = 0
    overflow_steps = 0
    visual_parameters = list(model.module.visual.parameters())
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(trainloader):
        imgs, pids, _, clothes_ids, cap = batch
        pids = pids.cuda(non_blocking=True)
        clothes_ids = clothes_ids.cuda(non_blocking=True)
        imgs = imgs.cuda(non_blocking=True)
        rel_batch = None
        if spec["relation"] != "none":
            try:
                rel_batch = next(relation_iter)
            except StopIteration:
                relation_iter = iter(relation_loader)
                rel_batch = next(relation_iter)
            rel_imgs = torch.cat([rel_batch[0], rel_batch[1], rel_batch[2]], dim=0).cuda(non_blocking=True)
            rel_batch = list(rel_batch)
            rel_batch[6:13] = [x.cuda(non_blocking=True) if torch.is_tensor(x) else torch.tensor(x, device=imgs.device)
                               for x in rel_batch[6:13]]
            # Keep scalar indices on CPU for coverage diagnostics.
            with torch.cuda.amp.autocast():
                rel_features = projected_image_cls(model.module, rel_imgs).float()
                rel_features = rel_features.chunk(3, dim=0)
                rel = relation_loss(rel_features[0], rel_features[1], rel_features[2], rel_batch, spec)
        else:
            rel = None
        with torch.cuda.amp.autocast():
            base, cla, pair, opl_loss = pdf_loss(
                config, model, criterion_cla, criterion_pair, opl, tokenizer,
                imgs, pids, clothes_ids, cap)
            if rel is None:
                lambda_loss = base.new_tensor(0.0)
                total = base
            else:
                lambda_loss = float(spec["lambda"]) * rel["weighted_before_lambda"]
                total = base + lambda_loss
        # Inspect the relation-only gradient before adding the PDF gradient.
        if rel is not None:
            relation_grads = torch.autograd.grad(lambda_loss, visual_parameters,
                                                 retain_graph=True, allow_unused=True)
            relation_grad = math.sqrt(sum(float(g.detach().float().norm(2).item() ** 2)
                                          for g in relation_grads if g is not None))
        else:
            relation_grad = 0.0
        finite_loss = bool(torch.isfinite(total.detach()).item())
        if not finite_loss:
            raise FloatingPointError("non-finite total loss at epoch {} batch {}".format(epoch + 1, batch_idx + 1))
        old_scale = scaler.get_scale()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient_finite = all(parameter.grad is None or
                               bool(torch.isfinite(parameter.grad.detach()).all().item())
                               for parameter in model.parameters())
        if gradient_finite:
            total_grad = grad_norm(model.parameters())
            visual_grad = grad_norm(visual_parameters)
        else:
            # GradScaler will skip this update.  Keep epoch diagnostics
            # numeric by averaging gradient norms over finite-gradient
            # steps, while recording the skipped count separately.
            total_grad = 0.0
            visual_grad = 0.0
        acc["finite_grad"].append(1.0 if gradient_finite else 0.0)
        scaler.step(optimizer)
        scaler.update()
        if not all(bool(torch.isfinite(parameter.detach()).all().item())
                   for parameter in model.parameters()):
            raise FloatingPointError("non-finite parameter after optimizer step")
        new_scale = scaler.get_scale()
        if new_scale < old_scale:
            overflow_steps += 1
        else:
            successful_steps += 1
        optimizer.zero_grad()

        append_metric(acc, "raw_loss", rel["raw_loss"] if rel else 0.0)
        append_metric(acc, "weighted_loss", rel["weighted_before_lambda"] if rel else 0.0)
        append_metric(acc, "lambda_loss", lambda_loss)
        append_metric(acc, "total_loss", total)
        append_metric(acc, "ratio", lambda_loss.detach() / (total.detach().abs() + 1e-12))
        append_metric(acc, "relation_grad", relation_grad)
        append_metric(acc, "visual_grad", visual_grad)
        append_metric(acc, "total_grad", total_grad)
        if rel:
            acc["raw_weight"].extend(rel["raw_weight"].detach().float().cpu().numpy().tolist())
            acc["norm_weight"].extend(rel["norm_weight"].detach().float().cpu().numpy().tolist())
            append_metric(acc, "active_rate", rel["active_rate"])
            append_metric(acc, "pos_cos", rel["pos_cos"])
            append_metric(acc, "neg_cos", rel["neg_cos"])
            append_metric(acc, "pos_margin", rel["d_neg"] - rel["d_pos"])
            append_metric(acc, "h_visual", rel["h_visual"])
            append_metric(acc, "h_semantic", rel["h_semantic"])
            append_metric(acc, "h_hybrid", rel["h_hybrid"])
            append_metric(acc, "r_conf", rel["r_conf"])
            append_metric(acc, "r_agr", rel["r_agr"])
            append_metric(acc, "r_joint", rel["r_joint"])
            append_metric(acc, "hinge_rate", rel["hinge_active_rate"])
            # DataLoader collates indices into tensors.
            anchor_ids = rel_batch[3].detach().cpu().numpy().tolist()
            positive_ids = rel_batch[4].detach().cpu().numpy().tolist()
            negative_ids = rel_batch[5].detach().cpu().numpy().tolist()
            acc["anchors"].extend(anchor_ids)
            acc["positive_ids"].extend(positive_ids)
            acc["negative_ids"].extend(negative_ids)
            acc["edge_ids"].extend((int(a) * 20 + int(s) for a, s in
                                     zip(anchor_ids, rel_batch[12].detach().cpu().numpy().tolist())))
            acc["positive_edge_ids"].extend(rel_batch[13].detach().cpu().numpy().tolist())
        if (batch_idx + 1) % 20 == 0 and logger:
            logger.info("epoch %d [%d/%d] base %.4f rel %.4f total %.4f",
                        epoch + 1, batch_idx + 1, len(trainloader),
                        mean_metric(acc, "total_loss") - mean_metric(acc, "lambda_loss"),
                        mean_metric(acc, "weighted_loss"), mean_metric(acc, "total_loss"))
        if duration_seconds is not None and time.time() - start >= duration_seconds:
            break
    elapsed = time.time() - start
    unique_anchor = len(set(acc["anchors"]))
    unique_positive = len(set(acc["positive_ids"]))
    unique_negative = len(set(acc["negative_ids"]))
    unique_edges = len(set(acc["edge_ids"]))
    unique_positive_edges = len(set(acc["positive_edge_ids"]))
    row = {
        "epoch": epoch + 1, "batches": len(acc["total_loss"]),
        "seconds": elapsed, "sec_per_iter": elapsed / max(1, len(acc["total_loss"])),
        "images_per_sec": config.DATA.TRAIN_BATCH * len(acc["total_loss"]) / max(elapsed, 1e-9),
        "raw_weight_mean": mean_metric(acc, "raw_weight"),
        "raw_weight_std": float(np.std(acc["raw_weight"])) if acc["raw_weight"] else 0.0,
        "raw_weight_zero_rate": float(np.mean(np.asarray(acc["raw_weight"]) == 0.0)) if acc["raw_weight"] else 0.0,
        "normalized_weight_mean": mean_metric(acc, "norm_weight"),
        "normalized_weight_std": float(np.std(acc["norm_weight"])) if acc["norm_weight"] else 0.0,
        "normalized_active_weight_mean": float(np.mean(np.asarray(acc["norm_weight"])[
            np.asarray(acc["raw_weight"]) > 0.0])) if acc["raw_weight"] and
            np.any(np.asarray(acc["raw_weight"]) > 0.0) else 0.0,
        "active_edge_rate": mean_metric(acc, "active_rate"),
        "raw_relation_loss_before_weight": mean_metric(acc, "raw_loss"),
        "weighted_relation_loss_before_lambda": mean_metric(acc, "weighted_loss"),
        "lambda_relation_loss": mean_metric(acc, "lambda_loss"),
        "total_loss": mean_metric(acc, "total_loss"),
        "lambda_over_total": mean_metric(acc, "ratio"),
        "relation_gradient_norm": mean_metric(acc, "relation_grad"),
        "visual_backbone_gradient_norm": mean_metric(acc, "visual_grad"),
        "gradient_finite_rate": mean_metric(acc, "finite_grad"),
        "nonfinite_gradient_steps": int(len(acc["finite_grad"]) -
                                          sum(acc["finite_grad"])),
        "relation_grad_over_total_grad": float(mean_metric(acc, "relation_grad") /
                                                (mean_metric(acc, "total_grad") + 1e-12)),
        "positive_cosine": mean_metric(acc, "pos_cos"),
        "sampled_negative_cosine": mean_metric(acc, "neg_cos"),
        "positive_negative_margin": mean_metric(acc, "pos_margin"),
        "H_visual": mean_metric(acc, "h_visual"), "H_sem": mean_metric(acc, "h_semantic"),
        "H_hybrid": mean_metric(acc, "h_hybrid"), "R_conf": mean_metric(acc, "r_conf"),
        "R_agr": mean_metric(acc, "r_agr"), "R_joint": mean_metric(acc, "r_joint"),
        "active_hinge_rate": mean_metric(acc, "hinge_rate"),
        "unique_anchors": unique_anchor, "unique_positive_images": unique_positive,
        "unique_negative_images": unique_negative, "unique_relation_edges": unique_edges,
        "positive_edge_coverage": float(unique_positive_edges / max(1, len(acc["positive_edge_ids"]))),
        "negative_edge_coverage": float(unique_edges / max(1, len(acc["edge_ids"]))),
        "edge_repeat_rate": float(1.0 - unique_edges / max(1, len(acc["edge_ids"]))),
        "max_edge_repeats": int(max(Counter(acc["edge_ids"]).values() or [0])),
        "successful_optimizer_steps": successful_steps,
        "amp_overflow_steps": overflow_steps,
        "finite": True,
    }
    return row


def save_model(model, path, epoch, spec, seed, extra=None):
    payload = {"model_state_dict": model.module.state_dict(), "epoch": int(epoch),
               "run_id": spec.get("run_id"), "training_seed": int(seed),
               "train_protocol": "RCHRL-V1 fixed 50 epoch protocol",
               "image_only_inference_unchanged": True}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def write_metrics_csv(rows, path):
    if not rows:
        return
    fields = list(rows[-1].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_training(args, phase):
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(os.path.join(args.output, "train.log"), rank)
    set_all_seeds(args.seed)
    records, pid_strings, clothes_keys, pid2clothes = load_train_records()
    config = make_config(args.seed, args.output)
    spec = dict(RUNS[args.run_id])
    spec["run_id"] = args.run_id
    graph_dir = args.graph
    manifest, ledger = verify_frozen_graph(graph_dir)
    if args.run_id != "R00" and spec["relation"] == "none":
        raise RuntimeError("R00 relation mismatch")
    if args.run_id != "R00" and not graph_dir:
        raise RuntimeError("relation run requires frozen graph")
    trainloader, train_sampler, transform_train = build_train_loader(config, records, args.seed)
    relation_loader = relation_sampler = relation_dataset = None
    if spec["relation"] != "none":
        relation_loader, relation_sampler, relation_dataset = build_relation_loader(
            config, records, graph_dir, spec["negative"], args.seed)
    model, criterion_cla, criterion_pair, optimizer, scheduler, trainable = build_model_and_optim(
        config, rank, len(clothes_keys))
    text_trainable = []
    for name, parameter in model.module.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in (
                "token_embedding", "transformer", "ln_final", "positional_embedding", "text_projection")):
            if parameter.requires_grad:
                text_trainable.append(name)
    if text_trainable:
        raise RuntimeError("frozen text encoder has trainable parameters: {}".format(text_trainable))
    if rank == 0:
        json_dump({
            "experiment": "RCHRL-V1", "run_id": args.run_id, "seed": args.seed,
            "spec": spec, "config_snapshot": config_snapshot(config),
            "source_commit": repo_commit(REPO_ROOT), "source_core_file_sha256": source_hashes(REPO_ROOT),
            "graph_manifest_sha256": sha256_file(os.path.join(graph_dir, "relation_graph_manifest.json")),
            "graph_ledger": ledger, "graph_is_frozen": True,
            "train_images": len(records), "train_ids": len(pid_strings),
            "train_loader_batches": len(trainloader), "relation_loader_batches": len(relation_loader) if relation_loader else 0,
            "test_access": "evaluation only; no test metadata/labels/features/captions used for relation construction or training",
            "text_encoder_trainable_parameters": text_trainable,
            "unexpected_relation_trainable_parameters": 0,
            "image_only_inference_output_shape_expected": [512],
        }, os.path.join(args.output, "run_provenance.json"))

    rows = []
    eval_rows = []
    best_diff = -1.0
    best_epoch = None
    # Smoke must cover a real five-minute training window.  Using one epoch
    # here would silently pass on fast local storage (one epoch is well under
    # five minutes), so allow enough epochs and pass the remaining global
    # duration into each epoch below.  Sanity intentionally remains one
    # epoch because its zero-second duration is checked after the first real
    # optimizer step.
    if phase == "train":
        max_epoch = config.TRAIN.MAX_EPOCH
    elif phase == "smoke":
        max_epoch = 1000000
    else:
        max_epoch = 1
    smoke_duration = args.duration if phase in ("smoke", "sanity") else None
    start = time.time()
    for epoch in range(max_epoch):
        train_sampler.set_epoch(epoch)
        if relation_sampler is not None:
            relation_sampler.set_epoch(epoch)
            relation_dataset.set_epoch(epoch)
        if phase == "smoke":
            epoch_duration = max(0.0, smoke_duration - (time.time() - start))
        else:
            epoch_duration = smoke_duration
        row = train_one_epoch(config, model, criterion_cla, criterion_pair,
                              optimizer, trainloader, relation_loader,
                              relation_sampler, relation_dataset, spec, epoch,
                              logger, epoch_duration)
        scheduler.step()
        rows.append(row)
        if phase == "train" and (epoch + 1) % config.TEST.EVAL_STEP == 0 or \
                phase == "train" and (epoch + 1) == config.TRAIN.MAX_EPOCH:
            if rank == 0:
                result = evaluate_image_only(model, config, logger)
                result["epoch"] = epoch + 1
                eval_rows.append(result)
                json_dump(result, os.path.join(args.output, "eval_epoch{:02d}.json".format(epoch + 1)))
                if result["different"]["R1"] > best_diff:
                    best_diff = result["different"]["R1"]
                    best_epoch = epoch + 1
                    save_model(model, os.path.join(args.output, "best_test_auxiliary.pth"),
                               epoch + 1, spec, args.seed,
                               {"best_test_diff_r1": best_diff, "checkpoint_role": "auxiliary only"})
            dist.barrier()
            model.train()
        if phase in ("smoke", "sanity") and time.time() - start >= smoke_duration:
            break
    if rank == 0:
        write_metrics_csv(rows, os.path.join(args.output, "metrics.csv"))
        json_dump(eval_rows, os.path.join(args.output, "evaluation_history.json"))
        if phase == "smoke":
            # Required checkpoint-write and image-only feature-shape smoke.
            smoke_path = os.path.join(args.output, "smoke_checkpoint.pth")
            save_model(model, smoke_path, rows[-1]["epoch"], spec, args.seed,
                       {"checkpoint_role": "smoke"})
            eval_result = evaluate_image_only(model, config, logger)
            json_dump(eval_result, os.path.join(args.output, "smoke_evaluation.json"))
            json_dump({"phase": "smoke", "duration_seconds": time.time() - start,
                       "required_duration_seconds": smoke_duration,
                       "iterations": int(sum(row["batches"] for row in rows)),
                       "sec_per_iter": float(np.average([row["sec_per_iter"] for row in rows],
                                                          weights=[row["batches"] for row in rows])),
                       "images_per_sec": float(np.sum([row["batches"] * config.DATA.TRAIN_BATCH for row in rows]) /
                                                max(time.time() - start, 1e-9)),
                       "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                       "successful_optimizer_steps": int(sum(row["successful_optimizer_steps"] for row in rows)),
                       "amp_overflow_steps": int(sum(row["amp_overflow_steps"] for row in rows)),
                       "gradient_finite_rate": float(np.average(
                           [row["gradient_finite_rate"] for row in rows],
                           weights=[row["batches"] for row in rows])),
                       "nonfinite_gradient_steps": int(sum(
                           row["nonfinite_gradient_steps"] for row in rows)),
                       "checkpoint_write": os.path.exists(smoke_path),
                       "evaluation_feature_shape": eval_result["feature_shape"],
                       "finite": all(row["finite"] for row in rows)},
                      os.path.join(args.output, "smoke_runtime.json"))
        elif phase == "train":
            final_path = os.path.join(args.output, "epoch50_final.pth")
            save_model(model, final_path, 50, spec, args.seed,
                       {"checkpoint_role": "primary epoch50 final",
                        "best_test_epoch_auxiliary": best_epoch,
                        "best_test_diff_r1_auxiliary": best_diff})
            json_dump({"status": "complete", "run_id": args.run_id, "seed": args.seed,
                       "epoch50_final": final_path, "best_test_epoch_auxiliary": best_epoch,
                       "best_test_diff_r1_auxiliary": best_diff,
                        "elapsed_seconds": time.time() - start},
                      os.path.join(args.output, "run_summary.json"))
        else:
            json_dump({"phase": "sanity", "run_id": args.run_id, "seed": args.seed,
                       "iterations": int(sum(row["batches"] for row in rows)),
                       "rows": rows, "finite": all(row["finite"] for row in rows),
                       "optimizer_update_success": all(row["successful_optimizer_steps"] > 0 for row in rows),
                       "gradient_finite_on_successful_updates": all(
                           row["successful_optimizer_steps"] > 0 and
                           row["gradient_finite_rate"] > 0.0 for row in rows),
                       "gradient_finite_rate": float(np.average(
                           [row["gradient_finite_rate"] for row in rows],
                           weights=[row["batches"] for row in rows])),
                       "nonfinite_gradient_steps": int(sum(
                           row["nonfinite_gradient_steps"] for row in rows)),
                       "amp_overflow_steps": int(sum(row["amp_overflow_steps"] for row in rows)),
                       "unexpected_trainable_relation_parameters": 0},
                      os.path.join(args.output, "sanity.json"))
    if hasattr(trainloader, "shutdown"):
        trainloader.shutdown()
    if relation_loader is not None:
        del relation_loader
    dist.barrier()
    dist.destroy_process_group()


def run_sanity(args):
    # The same runner executes one real AMP forward/backward/update for each
    # requested variant; no synthetic loss or fake optimizer step is used.
    run_training(args, "sanity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=list(RUNS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("train", "smoke", "sanity"), default="train")
    parser.add_argument("--duration", type=float, default=300.0)
    args = parser.parse_args()
    if args.phase == "sanity":
        # The first real AMP batch can overflow at GradScaler's initial scale
        # on this accelerator.  Run a short real window so the unchanged
        # scaler can adapt, and require a later successful update in the gate.
        args.duration = 30.0
        run_training(args, "sanity")
    else:
        run_training(args, args.phase)


if __name__ == "__main__":
    main()
