"""Train the fixed, train-only Original PDF epoch-50 mining checkpoint.

The script intentionally does not instantiate PRCC (whose constructor also
parses test metadata), does not import the evaluator, and performs no test
evaluation or checkpoint selection.
"""
from __future__ import absolute_import

import argparse
import datetime
import json
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from configs.default_img import _C
from data.dataloader import DataLoaderX
from data.samplers import DistributedRandomIdentitySampler
from losses import build_losses
from models.clip_model import build_CLIP_from_openai_pretrained
from tools.lr_scheduler import WarmupMultiStepLR
from train import train_clip_combiner
from tools.rchrl_common import (
    DATA_ROOT, ORIGINAL_CAPTION, REPO_ROOT, TrainCaptionDataset,
    config_snapshot, json_dump, load_train_records, repo_commit,
    set_all_seeds, sha256_file, source_hashes,
)


def make_config(seed, output, max_epoch=50):
    config = _C.clone()
    config.defrost()
    config.DATA.ROOT = DATA_ROOT
    config.DATA.CAPTION_PATH = ORIGINAL_CAPTION
    config.DATA.TRAIN_BATCH = 64
    config.DATA.TEST_BATCH = 64
    config.DATA.NUM_WORKERS = 4
    config.DATA.NUM_INSTANCES = 8
    config.TRAIN.MAX_EPOCH = int(max_epoch)
    config.TRAIN.OPTIMIZER.LR = 3.5e-7
    config.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
    config.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
    config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
    config.TRAIN.AMP = True
    config.SEED = int(seed)
    config.OUTPUT = output
    config.TAG = "v0_mining_epoch50_final"
    config.freeze()
    return config


def configure_logging(log_path):
    logger = logging.getLogger("cir_reid.train")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epoch", type=int, default=50)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    os.makedirs(args.output, exist_ok=True)
    logger = configure_logging(os.path.join(args.output, "v0_training.log")) if rank == 0 else None
    set_all_seeds(args.seed)

    records, pid_strings, clothes_keys, pid2clothes = load_train_records()
    if len(records) != 17896 or len(pid_strings) != 150:
        raise RuntimeError("Unexpected train inventory: {} images / {} IDs".format(
            len(records), len(pid_strings)))
    config = make_config(args.seed, args.output, args.max_epoch)
    transform_train, _ = __import__("data", fromlist=["build_img_transforms"]).build_img_transforms(config)
    train_dataset = TrainCaptionDataset(records, transform_train, ORIGINAL_CAPTION)
    train_tuples = [(row["path"], row["person_id"], row["camera_id"], row["clothes_id"])
                    for row in records]
    train_sampler = DistributedRandomIdentitySampler(
        train_tuples, num_instances=config.DATA.NUM_INSTANCES, seed=config.SEED)
    trainloader = DataLoaderX(
        dataset=train_dataset, sampler=train_sampler,
        batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, drop_last=True)

    clip_model, _ = build_CLIP_from_openai_pretrained(
        "ViT-B/16", (config.DATA.HEIGHT, config.DATA.WIDTH), 16, len(pid_strings))
    clip_model.eval().float()
    clip_model.freeze_text_encoder()
    criterion_cla, criterion_pair, _, _ = build_losses(config, len(clothes_keys))
    trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.TRAIN.OPTIMIZER.LR,
                           weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    scheduler = WarmupMultiStepLR(
        optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
        gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, warmup_factor=0.1,
        warmup_iters=10)
    clip_model = clip_model.cuda(rank)
    clip_model = nn.parallel.DistributedDataParallel(
        clip_model, device_ids=[rank], output_device=rank)

    start = time.time()
    if rank == 0:
        logger.info("RCHRL-V1 V0 mining training: train-only, no test metadata/evaluation")
        logger.info("train_images=%d train_ids=%d train_batches=%d", len(records),
                    len(pid_strings), len(trainloader))
    for epoch in range(config.TRAIN.MAX_EPOCH):
        train_sampler.set_epoch(epoch)
        train_clip_combiner(config, epoch, clip_model, criterion_cla,
                            criterion_pair, optimizer, trainloader,
                            torch.from_numpy(pid2clothes))
        scheduler.step()
        if rank == 0:
            logger.info("V0 epoch %d/%d completed", epoch + 1, config.TRAIN.MAX_EPOCH)

    if rank == 0:
        checkpoint_name = "epoch50_final.pth" if args.max_epoch == 50 else "epoch{}_test.pth".format(args.max_epoch)
        checkpoint_path = os.path.join(args.output, checkpoint_name)
        torch.save({
            "model_state_dict": clip_model.module.state_dict(),
            "epoch": int(args.max_epoch),
            "training_seed": int(args.seed),
            "train_only": True,
            "checkpoint_selection": "fixed epoch50 final; no test-adaptive selection" if args.max_epoch == 50 else "development smoke; not valid for mining",
            "source_commit": repo_commit(REPO_ROOT),
            "config_snapshot": config_snapshot(config),
            "train_protocol": {
                "dataset": "PRCC train only",
                "backbone": "ViT-B/16",
                "image_size": [384, 128],
                "batch": 64,
                "world_size": 1,
                "epochs": int(args.max_epoch),
                "optimizer": "Adam",
                "lr": 3.5e-7,
                "weight_decay": 5e-4,
                "milestones": [20, 40],
                "gamma": 0.1,
                "caption_mechanism": "frozen current Original PDF captions",
                "relation_loss": "none",
                "evaluation": "not run",
            },
        }, checkpoint_path)
        if args.max_epoch != 50:
            json_dump({"status": "development_only", "checkpoint": checkpoint_path,
                       "epoch": int(args.max_epoch)},
                      os.path.join(args.output, "v0_development_summary.json"))
        else:
            provenance = {
                "experiment": "RCHRL-V1",
                "purpose": "fixed Original PDF epoch50 final checkpoint for train-only relation mining",
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_commit": repo_commit(REPO_ROOT),
                "source_core_file_sha256": source_hashes(REPO_ROOT),
                "training_seed": int(args.seed),
                "epoch": 50,
                "train_only": True,
                "test_adaptive_checkpoint_selection": False,
                "no_test_adaptive_checkpoint_selection_was_used_for_relation_mining": True,
                "test_metadata_loaded_during_training": False,
                "test_features_loaded_during_training": False,
                "test_captions_loaded_during_training": False,
                "test_metrics_loaded_during_training": False,
                "train_inventory": {"images": len(records), "ids": len(pid_strings)},
                "config_snapshot": config_snapshot(config),
                "train_protocol": {
                    "dataset": "PRCC TRAIN",
                    "backbone": "ViT-B/16",
                    "resolution": "384x128",
                    "batch": 64,
                    "world": 1,
                    "epochs": 50,
                    "optimizer": "Adam",
                    "lr": 3.5e-7,
                    "weight_decay": 5e-4,
                    "milestones": [20, 40],
                    "gamma": 0.1,
                    "same_augmentation": True,
                    "amp_grad_scaler": True,
                    "loss_family": "current Original PDF L_PDF",
                    "relation_loss": None,
                    "primary_checkpoint": "epoch50_final.pth",
                },
                "wall_clock_seconds": time.time() - start,
            }
            json_dump(provenance, os.path.join(REPO_ROOT, "reports", "v0_mining_checkpoint_provenance.json"))
        summary = {"status": "complete" if args.max_epoch == 50 else "development_only",
                   "checkpoint": checkpoint_path, "epoch": int(args.max_epoch),
                   "train_only": True, "elapsed_seconds": time.time() - start}
        if args.max_epoch == 50:
            summary["checkpoint_sha256"] = provenance["checkpoint_sha256"]
        json_dump(summary, os.path.join(args.output, "v0_training_summary.json"))
        logger.info("V0 epoch50 final written: %s", checkpoint_path)
    if hasattr(trainloader, "shutdown"):
        trainloader.shutdown()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
