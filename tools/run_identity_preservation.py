"""Run one pre-registered PDF identity-preservation experiment.

Each invocation owns exactly one visible device and one world-size-1 process.
The runner never passes semantic guidance into the original caption path.
"""

import argparse
import csv
import json
import logging
import os
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

from configs.identity_preservation_common import build_config
from configs.identity_preservation_registry import output_name, spec_for
from data import build_dataloader
from data.semantic_reliability import sha256
from models.clip_model import build_CLIP_from_openai_pretrained
from models.utils.simple_tokenizer import SimpleTokenizer
from losses import build_losses
from tools.lr_scheduler import WarmupMultiStepLR
from tools.reliability_sanity import stats
from tools.utils import set_seed
from train import tokenize, train_clip_combiner
from test import test_prcc_clip_combiner


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class Tee:
    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, value):
        self.stream.write(value)
        self.handle.write(value)
        self.handle.flush()

    def flush(self):
        self.stream.flush()
        self.handle.flush()


@torch.no_grad()
def build_bank(model, dataset):
    """Encode the already-frozen four-attribute vocabulary exactly once."""
    tokenizer = SimpleTokenizer()
    from data.semantic_reliability import attribute_phrase

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


def checkpoint_is_finite(path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' not in checkpoint:
        raise ValueError('checkpoint missing model_state_dict: ' + str(path))
    for key, value in checkpoint['model_state_dict'].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all().item():
            raise FloatingPointError('non-finite checkpoint tensor: ' + key)
    return checkpoint


def main(args):
    spec = spec_for(args.variant)
    if args.seed not in (0, 1, 2):
        raise ValueError('Formal V2 seeds are restricted to 0, 1, 2')
    expected_name = output_name(args.variant, args.seed)
    output = Path(args.output)
    if output.name != expected_name:
        raise ValueError('Output directory does not match pre-registered name: {} != {}'.format(
            output.name, expected_name))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite existing run: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)

    config = build_config(spec, args.seed, output)
    console = (output / 'console.log').open('w')
    sys.stdout = Tee(sys.stdout, console)
    sys.stderr = Tee(sys.stderr, console)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output / 'train.log'),
                                  logging.StreamHandler(sys.stdout)])

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=str(ROOT), text=True)
    config_snapshot = config.dump()
    (output / 'config_snapshot.yaml').write_text(config_snapshot)
    (output / 'git_commit.txt').write_text(commit + '\n')
    (output / 'git_diff.patch').write_text(diff)
    manifest = dict(
        variant=args.variant,
        seed=args.seed,
        mechanism=spec['mechanism'],
        reliability=spec['reliability'],
        lambda_raw=spec['lambda_raw'],
        lambda_pres=spec['lambda_pres'],
        lambda_excl=spec['lambda_excl'],
        lambda_rank=spec['lambda_rank'],
        margin=spec['margin'],
        phase=args.phase,
        git_commit=commit,
        tracked_diff_empty=not diff,
        config=config_snapshot,
        world_size=1,
        effective_batch_size=64,
        batch_size=64,
        checkpoint_selection='primary=final epoch50; auxiliary=earliest best test Different Clothes Rank-1',
        backbone='CLIP ViT-B/16',
        pretrained_sha256=sha256('/root/.cache/clip/ViT-B-16.pt'),
        caption_sha256=sha256(config.DATA.CAPTION_PATH),
        semantic_cache_sha256=sha256(config.DATA.SEMANTIC_CACHE),
        semantic_cache_expected_records=17896,
        torch_version=torch.__version__,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seconds=args.seconds,
        eot_additive_guidance=False,
        original_caption_branch_unchanged=True,
        original_pdf_loss_unchanged=True,
        observation_gates=False,
        added_trainable_parameters=0,
        triplet_loss_active=False,
    )
    write_json(output / 'run_manifest.json', manifest)
    status = dict(status='initializing', variant=args.variant, seed=args.seed,
                  phase=args.phase, git_commit=commit)
    write_json(output / 'status.json', status)

    times = []
    iteration_records = []
    epoch_records = []
    swaps_before = swap_counters()
    loader = None
    metrics_handle = None
    epoch_handle = None
    try:
        validation_path = ROOT / 'reports/prcc_semantic_cache_validation.json'
        validation = json.loads(validation_path.read_text())
        if not validation['passed']:
            raise RuntimeError('Semantic cache validation report is not passed')
        if validation['cache_sha256'] != sha256(config.DATA.SEMANTIC_CACHE):
            raise RuntimeError('Semantic cache SHA256 mismatch after validation')
        if validation['train_images'] != 17896 or validation['semantic_records'] != 17896:
            raise RuntimeError('Semantic cache record count mismatch')

        with tempfile.TemporaryDirectory(prefix='pdf-identity-v2-') as directory:
            torch.cuda.set_device(0)
            dist.init_process_group('nccl', init_method='file://' + directory + '/rendezvous',
                                    rank=0, world_size=1)
            set_seed(config.SEED)
            loader, same, different, gallery, dataset, sampler = build_dataloader(config)
            if config.DATA.TRAIN_BATCH != 64:
                raise ValueError('Frozen batch size must be 64')
            if len(dataset.train) != 17896:
                raise ValueError('Unexpected PRCC train size: {}'.format(len(dataset.train)))

            model, _ = build_CLIP_from_openai_pretrained(
                'ViT-B/16', (config.DATA.HEIGHT, config.DATA.WIDTH), 16, 150)
            model.eval().float()
            model.freeze_text_encoder()
            model.cuda()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            original_keys = tuple(model.state_dict())
            bank = build_bank(model, loader.dataset)
            if tuple(model.state_dict()) != original_keys:
                raise AssertionError('Frozen semantic bank construction changed model state')
            if trainable != sum(p.numel() for p in model.parameters() if p.requires_grad):
                raise AssertionError('Semantic bank construction changed trainable parameter count')
            if bank.ndim != 2 or bank.shape[1] != 512:
                raise ValueError('Unexpected semantic bank shape: {}'.format(tuple(bank.shape)))
            manifest.update(trainable_parameters=trainable,
                            semantic_bank_shape=list(bank.shape),
                            semantic_statistics=stats(loader.dataset.reliability_by_mode),
                            inference_dimension=512)
            write_json(output / 'run_manifest.json', manifest)

            criterion_cla, criterion_pair, _, _ = build_losses(config, dataset.num_train_clothes)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.TRAIN.OPTIMIZER.LR,
                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
            scheduler = WarmupMultiStepLR(
                optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE,
                gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                warmup_factor=0.1, warmup_iters=10)

            metric_columns = (
                'epoch', 'iteration', 'train_loss', 'id_loss', 'triplet_loss', 'pair_loss',
                'pdf_opl_loss', 'semantic_loss', 'L_raw', 'L_pres', 'L_excl', 'L_rank',
                'cos_raw_sem', 'cos_res1_sem', 'cos_res2_sem', 'cos_com_sem',
                'semantic_margin_res1_minus_com', 'semantic_margin_res2_minus_com',
                'semantic_valid_rate', 'semantic_valid_count', 'learning_rate',
                'iteration_seconds', 'batch_size', 'scaler_scale_before',
                'scaler_scale_after', 'optimizer_step_skipped',
                'unscaled_gradient_finite', 'memory_allocated', 'memory_reserved',
                'global_iteration')
            metrics_handle = (output / 'metrics.csv').open('w', newline='')
            writer = csv.DictWriter(metrics_handle, fieldnames=metric_columns)
            writer.writeheader()
            epoch_columns = ('epoch', 'L_raw', 'L_pres', 'L_excl', 'L_rank',
                             'cos_raw_sem', 'cos_res1_sem', 'cos_res2_sem', 'cos_com_sem',
                             'semantic_margin_res1_minus_com', 'semantic_margin_res2_minus_com',
                             'semantic_valid_count', 'semantic_valid_rate')
            epoch_handle = (output / 'epoch_diagnostics.csv').open('w', newline='')
            epoch_writer = csv.DictWriter(epoch_handle, fieldnames=epoch_columns)
            epoch_writer.writeheader()

            def observe(row):
                row['global_iteration'] = len(iteration_records) + 1
                writer.writerow(row)
                metrics_handle.flush()
                times.append(float(row['iteration_seconds']))
                iteration_records.append(row)
                if len(times) == 1 or len(times) % 20 == 0 or row['optimizer_step_skipped']:
                    logging.info('METRICS %s', json.dumps(row))

            def observe_epoch(row):
                epoch_writer.writerow(row)
                epoch_handle.flush()
                epoch_records.append(row)

            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            stop_at = started + args.seconds if args.phase == 'smoke' else None
            status['status'] = 'running'
            write_json(output / 'status.json', status)
            best_rank = -1.0
            best_epoch = None
            evaluation = []
            for epoch in range(config.TRAIN.MAX_EPOCH):
                sampler.set_epoch(epoch)
                train_clip_combiner(
                    config, epoch, model, criterion_cla, criterion_pair, optimizer, loader,
                    torch.from_numpy(dataset.pid2clothes), semantic_bank=bank,
                    observer=observe, stop_at=stop_at, allow_amp_overflow=True,
                    identity_config=config.IDENTITY_PRESERVATION,
                    epoch_observer=observe_epoch)
                if args.phase == 'train' and ((epoch + 1) % config.TEST.EVAL_STEP == 0
                                               or epoch + 1 == config.TRAIN.MAX_EPOCH):
                    result = test_prcc_clip_combiner(
                        model, same, different, gallery, dataset, return_metrics=True)
                    result['epoch'] = epoch + 1
                    evaluation.append(result)
                    write_json(output / 'evaluation.json', evaluation)
                    score = result['different']['Rank-1']
                    if score > best_rank:
                        best_rank, best_epoch = score, epoch + 1
                        torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch + 1,
                                        git_commit=commit, config=config_snapshot), output / 'best.pth')
                scheduler.step()
                if stop_at is not None and time.monotonic() >= stop_at:
                    break

            elapsed = time.monotonic() - started
            completed_epochs = epoch + 1
            if args.phase == 'train':
                torch.save(dict(model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(),
                                epoch=completed_epochs, git_commit=commit,
                                config=config_snapshot), output / 'final.pth')
                checkpoint_is_finite(output / 'best.pth')
                checkpoint_is_finite(output / 'final.pth')

            skipped = [row['global_iteration'] for row in iteration_records
                       if row['optimizer_step_skipped']]
            first_update = next((row['global_iteration'] for row in iteration_records
                                 if not row['optimizer_step_skipped']), None)
            model_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
            stable_window = min(20, len(iteration_records))
            terminal_records = iteration_records[-stable_window:] if stable_window else []
            stable = (model_finite and bool(terminal_records)
                      and all(row['unscaled_gradient_finite']
                              and not row['optimizer_step_skipped']
                              and row['scaler_scale_before'] == row['scaler_scale_after']
                              for row in terminal_records))
            consecutive = 0
            max_consecutive = 0
            for row in iteration_records:
                if row['optimizer_step_skipped'] or not row['unscaled_gradient_finite']:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            actual_updates = max(int(state['step']) for state in optimizer.state.values()) if optimizer.state else 0
            if actual_updates != len(iteration_records) - len(skipped):
                raise AssertionError('GradScaler skip telemetry disagrees with Adam step counters')
            if max_consecutive >= 20:
                raise FloatingPointError('Persistent non-finite gradients: {} consecutive iterations'.format(
                    max_consecutive))
            if args.phase == 'smoke' and not stable:
                raise RuntimeError('Smoke did not reach a terminal stable GradScaler window')
            if args.phase == 'train' and completed_epochs != config.TRAIN.MAX_EPOCH:
                raise RuntimeError('Formal run did not complete epoch50')

            measured = times[5:] if len(times) > 5 else times
            status.update(
                status='passed' if args.phase == 'smoke' else 'complete',
                completed_iterations=len(times),
                training_seconds=elapsed,
                measured_iterations=len(measured),
                mean_iteration_seconds=statistics.mean(measured) if measured else None,
                median_iteration_seconds=statistics.median(measured) if measured else None,
                images_per_second=(config.DATA.TRAIN_BATCH / statistics.mean(measured)
                                   if measured and statistics.mean(measured) else None),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                swap_delta={key: swap_counters()[key] - value for key, value in swaps_before.items()},
                no_nonfinite_loss=True,
                model_parameters_finite=model_finite,
                gradient_overflow_iterations=skipped,
                initial_overflow_count=(first_update - 1) if first_update else len(times),
                overflow_count=len(skipped),
                successful_optimizer_updates=len(times) - len(skipped),
                adam_step_counter=actual_updates,
                first_successful_update_iteration=first_update,
                max_consecutive_nonfinite_gradient_iterations=max_consecutive,
                scaler_stable=stable,
                stable_window_iterations=stable_window,
                final_scale=iteration_records[-1]['scaler_scale_after'] if iteration_records else None,
                scaler_protocol='default GradScaler(), overflow/skipped steps tolerated unless persistent',
                all_iteration_mean_seconds=statistics.mean(times) if times else None,
                no_oom=True,
                best_epoch=best_epoch,
                final_epoch=completed_epochs,
                primary_result_epoch=50 if args.phase == 'train' else None,
                full_epochs_completed=args.phase == 'train',
            )
            logging.info('RESULT %s', json.dumps(status))
    except Exception as exc:
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      completed_iterations=len(iteration_records))
        logging.exception('Experiment stopped; pre-registered hyperparameters were not changed')
        raise
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        if epoch_handle is not None:
            epoch_handle.close()
        if torch.cuda.is_initialized():
            status['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            status['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        write_json(output / 'status.json', status)
        if loader is not None:
            try:
                loader.shutdown()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
        console.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--phase', choices=['smoke', 'train'], default='smoke')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.phase == 'smoke' and not 300 <= args.seconds <= 600:
        parser.error('Smoke duration must be 300-600 seconds')
    main(args)
