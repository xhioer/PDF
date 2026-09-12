"""One runner for V0–V4. Default is a bounded smoke test, never full training.

Use --phase train only after the user's separate long-training confirmation.
"""
import argparse
import csv
import json
import logging
import os
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import distributed as dist
from data import build_dataloader
from data.semantic_reliability import sha256, attribute_phrase, aggregate_identity, MODES
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from losses import build_losses
from tools.utils import set_seed
from tools.lr_scheduler import WarmupMultiStepLR
from tools.reliability_sanity import stats
from train import tokenize, train_clip_combiner
from test import test_prcc_clip_combiner

VARIANTS = dict(zip(MODES, ('V1_p2_none', 'V2_p2_attribute', 'V3_p2_confidence', 'V4_p2_joint')))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class Tee:
    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, value):
        self.stream.write(value)
        self.handle.write(value)
        self.handle.flush()

    def flush(self):
        self.stream.flush()
        self.handle.flush()


@torch.no_grad()
def build_bank(model, dataset):
    # Shared frozen encoder; the bank is input data, not model state or a branch.
    tokenizer = SimpleTokenizer()
    texts = [attribute_phrase(*item) for item in dataset.attribute_vocabulary]
    encoded = []
    for start in range(0, len(texts), 64):
        tokens = tokenize(texts[start:start + 64], tokenizer, truncate=False).cuda()
        encoded.append(model.encode_text(tokens).float())
    return torch.cat([torch.zeros(1, 512, device='cuda')] + encoded).detach()


def swap_counters():
    result = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in ('pswpin', 'pswpout'):
            result[key] = int(value)
    return result


def main(args):
    config = runpy.run_path(str(Path(args.cfg).resolve()))['CONFIG'].clone()
    if args.v0:
        config.defrost()
        config.DATA.SEMANTIC_CACHE = ''
        config.freeze()
    variant = 'V0_original' if args.v0 else VARIANTS[config.SEMANTIC_RELIABILITY_MODE]
    output = Path(args.output) if args.output else Path(config.OUTPUT) / variant
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite existing run: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    console = (output / 'console.log').open('w')
    sys.stdout = Tee(sys.stdout, console)
    sys.stderr = Tee(sys.stderr, console)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output / 'train.log'), logging.StreamHandler(sys.stdout)])
    (output / 'config_snapshot.yaml').write_text(config.dump())
    (output / 'config_snapshot.py').write_text(Path(args.cfg).read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    (output / 'git_commit.txt').write_text(commit + '\n')
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=str(ROOT), text=True)
    (output / 'git_diff.patch').write_text(diff)
    manifest = dict(variant=variant, phase=args.phase, git_commit=commit, tracked_diff_empty=not diff,
        config=config.dump(), world_size=1, effective_batch_size=64,
        checkpoint_selection='primary=final epoch50; auxiliary=earliest best test Diff Rank1',
        backbone='CLIP ViT-B/16', pretrained_sha256=sha256('/root/.cache/clip/ViT-B-16.pt'),
        caption_sha256=sha256(config.DATA.CAPTION_PATH),
        torch_version=torch.__version__, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        seconds=args.seconds, original_losses_unchanged=True,
        triplet_loss_active=False, new_semantic_loss=False)
    write_json(output / 'run_manifest.json', manifest)
    status = dict(status='initializing', variant=variant, phase=args.phase, git_commit=commit)
    write_json(output / 'status.json', status)
    loader = None
    times = []
    swaps_before = swap_counters()
    try:
        sanity = json.loads((ROOT / 'reports/reliability_sanity_check.json').read_text())
        validation = json.loads((ROOT / 'reports/prcc_semantic_cache_validation.json').read_text())
        if not sanity['passed'] or not validation['passed']:
            raise RuntimeError('Failed preflight reports')
        if not args.v0 and sha256(config.DATA.SEMANTIC_CACHE) != validation['cache_sha256']:
            raise RuntimeError('Cache changed after validation')
        with tempfile.TemporaryDirectory(prefix='pdf-run-') as directory:
            torch.cuda.set_device(0)
            dist.init_process_group('nccl', init_method='file://' + directory + '/rendezvous', rank=0, world_size=1)
            set_seed(config.SEED)
            loader, same, different, gallery, dataset, sampler = build_dataloader(config)
            if config.DATA.TRAIN_BATCH != 64:
                raise ValueError('Frozen batch size must be 64')
            model, _ = build_CLIP_from_openai_pretrained('ViT-B/16', (config.DATA.HEIGHT, config.DATA.WIDTH), 16, 150)
            model.eval().float()
            model.freeze_text_encoder()
            model.cuda()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            original_keys = tuple(model.state_dict())
            bank = None if args.v0 else build_bank(model, loader.dataset)
            assert tuple(model.state_dict()) == original_keys
            assert trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)
            manifest.update(trainable_parameters=trainable, added_trainable_parameters=0,
                            inference_dimension=512, device=torch.cuda.get_device_name(0))
            if bank is not None:
                manifest['semantic_bank_shape'] = list(bank.shape)
                write_json(output / 'reliability_stats.json', stats(loader.dataset.reliability_by_mode))
            else:
                write_json(output / 'reliability_stats.json', {'mode': 'original', 'applied': False})
            write_json(output / 'run_manifest.json', manifest)
            criterion_cla, criterion_pair, _, _ = build_losses(config, dataset.num_train_clothes)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            scheduler = WarmupMultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
                gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE, warmup_factor=0.1, warmup_iters=10)
            metrics_handle = (output / 'metrics.csv').open('w', newline='')
            columns = ('epoch', 'iteration', 'train_loss', 'id_loss', 'triplet_loss', 'pair_loss',
                       'pdf_opl_loss', 'semantic_loss', 'learning_rate', 'iteration_seconds', 'batch_size')
            writer = csv.DictWriter(metrics_handle, fieldnames=columns)
            writer.writeheader()

            def observe(row):
                writer.writerow(row)
                metrics_handle.flush()
                times.append(row['iteration_seconds'])
                if len(times) == 1 or len(times) % 20 == 0:
                    logging.info('METRICS %s', json.dumps(row))

            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            stop_at = started + args.seconds if args.phase == 'smoke' else None
            status['status'] = 'running'
            write_json(output / 'status.json', status)
            best_rank, best_epoch, evaluation = -1.0, None, []
            for epoch in range(config.TRAIN.MAX_EPOCH):
                sampler.set_epoch(epoch)
                train_clip_combiner(config, epoch, model, criterion_cla, criterion_pair, optimizer,
                    loader, torch.from_numpy(dataset.pid2clothes), semantic_bank=bank,
                    observer=observe, stop_at=stop_at)
                if args.phase == 'train' and ((epoch + 1) % config.TEST.EVAL_STEP == 0
                        or epoch + 1 == config.TRAIN.MAX_EPOCH):
                    result = test_prcc_clip_combiner(model, same, different, gallery, dataset, return_metrics=True)
                    result['epoch'] = epoch + 1
                    evaluation.append(result)
                    write_json(output / 'evaluation.json', evaluation)
                    score = result['different']['Rank-1']
                    if score > best_rank:
                        best_rank, best_epoch = score, epoch + 1
                        torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch + 1,
                            git_commit=commit, config=config.dump()), output / 'best.pth')
                scheduler.step()
                if stop_at is not None and time.monotonic() >= stop_at:
                    break
            elapsed = time.monotonic() - started
            if args.phase == 'train':
                torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), epoch=epoch + 1, git_commit=commit,
                    config=config.dump()), output / 'final.pth')
            measured = times[5:] if len(times) > 5 else times
            status.update(status='passed' if args.phase == 'smoke' else 'complete',
                completed_iterations=len(times), training_seconds=elapsed, measured_iterations=len(measured),
                mean_iteration_seconds=statistics.mean(measured), median_iteration_seconds=statistics.median(measured),
                images_per_second=config.DATA.TRAIN_BATCH / statistics.mean(measured),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                swap_delta={key: swap_counters()[key] - value for key, value in swaps_before.items()},
                no_nan_inf=True, no_oom=True, best_epoch=best_epoch,
                final_epoch=epoch + 1, full_epochs_completed=args.phase == 'train')
            metrics_handle.close()
            logging.info('RESULT %s', json.dumps(status))
    except Exception as exc:
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      completed_iterations=len(times))
        logging.exception('Experiment stopped; no hyperparameters will be changed')
        raise
    finally:
        write_json(output / 'status.json', status)
        # Do not drain the prefetch queue on failure; the isolated process exits.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--phase', choices=['smoke', 'train'], default='smoke')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--output')
    parser.add_argument('--v0', action='store_true')
    args = parser.parse_args()
    if args.phase == 'smoke' and not 300 <= args.seconds <= 600:
        parser.error('Smoke duration must be 300–600 seconds')
    main(args)
