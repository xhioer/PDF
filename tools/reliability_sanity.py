"""100 actual image batches; all modes use the exact training lookup tensors."""
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from data import build_dataloader
from data.semantic_reliability import ATTRIBUTES, MODES, ATTRIBUTE_RELIABILITY, CONFIDENCE_WEIGHT
from tools.utils import set_seed


def stats(weights):
    result = []
    for mode in MODES:
        for index, attribute in enumerate(ATTRIBUTES):
            values = weights[mode][:, index].double()
            result.append(dict(mode=mode, attribute=attribute, mean=values.mean().item(),
                std=values.std(unbiased=False).item(), min=values.min().item(), max=values.max().item(),
                zero_rate=(values == 0).double().mean().item(),
                unique_values=[round(v, 8) for v in values.unique().tolist()]))
    return result


def main():
    config = runpy.run_path(str(ROOT / 'configs/v1_p2_none.py'))['CONFIG']
    with tempfile.TemporaryDirectory(prefix='pdf-sanity-') as directory:
        dist.init_process_group('nccl', init_method='file://' + directory + '/rendezvous', rank=0, world_size=1)
        torch.cuda.set_device(0)
        set_seed(config.SEED)
        loader, *rest = build_dataloader(config)
        dataset = loader.dataset
        collected = {mode: [] for mode in MODES}
        index_batches = []
        for batch_idx, batch in enumerate(loader):
            payload = batch[-1]
            indices = payload['dataset_index'].cpu()
            index_batches.append(indices)
            for mode in MODES:
                actual = dataset.reliability_by_mode[mode][indices]
                expected = []
                for index in indices.tolist():
                    desc = dataset.semantic_records[index]['description']
                    row = []
                    for attribute in ATTRIBUTES:
                        value = 0.0 if desc[attribute] == 'unknown' else 1.0
                        if mode in ('attribute', 'joint'):
                            value *= ATTRIBUTE_RELIABILITY[attribute]
                        if mode in ('confidence', 'joint'):
                            value *= CONFIDENCE_WEIGHT[desc[attribute + '_confidence']]
                        row.append(value)
                    expected.append(row)
                torch.testing.assert_close(actual, torch.tensor(expected))
                collected[mode].append(actual)
            torch.testing.assert_close(payload['reliability'].cpu(), collected['none'][-1])
            if (batch_idx + 1) % 20 == 0:
                print('Checked {} actual image batches'.format(batch_idx + 1), flush=True)
            if batch_idx + 1 == 100:
                break
        if len(index_batches) < 100:
            raise AssertionError('Fewer than 100 batches')
        weights = {mode: torch.cat(values) for mode, values in collected.items()}
        if any(torch.equal(weights[a], weights[b]) for i, a in enumerate(MODES) for b in MODES[i + 1:]):
            raise AssertionError('Identical reliability tensors across two modes')
        configs = [runpy.run_path(str(ROOT / path))['CONFIG'] for path in (
            'configs/v1_p2_none.py', 'configs/v2_p2_attribute.py',
            'configs/v3_p2_confidence.py', 'configs/v4_p2_joint.py')]
        canonical = []
        for cfg in configs:
            cfg = cfg.clone()
            cfg.defrost()
            cfg.SEMANTIC_RELIABILITY_MODE = 'none'
            canonical.append(cfg.dump())
        assert len(set(canonical)) == 1
        report = dict(passed=True, actual_image_dataloader_batches=100,
            sampled_images=sum(len(i) for i in index_batches),
            unique_images=torch.cat(index_batches).unique().numel(),
            batch_size=config.DATA.TRAIN_BATCH, seed=config.SEED,
            config_only_difference='SEMANTIC_RELIABILITY_MODE',
            cache_sha256=dataset.semantic_validation['cache_sha256'],
            statistics=stats(weights), full_cache_statistics=stats(dataset.reliability_by_mode))
        (ROOT / 'reports/reliability_sanity_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'full_cache_statistics'}, indent=2))
        loader.shutdown()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
