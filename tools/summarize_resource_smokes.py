"""Summarize bounded smoke telemetry without deriving ReID performance claims."""
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/pdf_reliability_ablation/preflight/adaptive_amp_v2'


def summarize_run(path):
    status = json.loads((path / 'status.json').read_text())
    rows = list(csv.DictReader((path / 'metrics.csv').open()))
    epochs = {}
    for row in rows:
        epochs.setdefault(row['epoch'], []).append(row)
    scaler_epochs = []
    for epoch, values in epochs.items():
        skips = [int(row['iteration']) for row in values if row['optimizer_step_skipped'] == 'True']
        first = next((int(row['iteration']) for row in values if row['optimizer_step_skipped'] == 'False'), None)
        scaler_epochs.append(dict(epoch=int(epoch), iterations=len(values), initial_overflows=first - 1 if first else len(values),
            overflow_iterations=skips, stable_after_last_overflow_iteration=skips[-1] + 1 if skips else 1,
            final_scale=float(values[-1]['scaler_scale_after'])))
    stamps = []
    for line in (path / 'train.log').read_text().splitlines():
        if 'METRICS ' in line:
            stamps.append(datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f').timestamp())
    return dict(status=status, per_epoch_scaler=scaler_epochs,
                measured_log_window=[min(stamps), max(stamps)] if stamps else None)


def resource_summary(stage, runs):
    samples = [json.loads(line) for line in (OUT / (stage + '_resources.jsonl')).read_text().splitlines()]
    errors = [sample for sample in samples if 'telemetry_error' in sample]
    samples = [sample for sample in samples if 'telemetry_error' not in sample]
    windows = [value['measured_log_window'] for value in runs.values() if value['measured_log_window']]
    start, end = max(value[0] for value in windows), min(value[1] for value in windows)
    overlap = [sample for sample in samples if start <= sample['timestamp'] <= end]
    devices = []
    for device in (0, 1):
        all_gpu = [sample['gpus'][device] for sample in samples]
        gpu = [sample['gpus'][device] for sample in overlap]
        devices.append(dict(device=device, peak_used_mib=max(row['used_mib'] for row in all_gpu),
            total_mib=all_gpu[0]['total_mib'], peak_fraction=max(row['used_mib'] for row in all_gpu) / all_gpu[0]['total_mib'],
            steady_utilization_mean_percent=statistics.mean(row['utilization_percent'] for row in gpu) if gpu else None,
            steady_utilization_max_percent=max(row['utilization_percent'] for row in gpu) if gpu else None))
    mem = [sample['host_memory'] for sample in samples]
    return dict(sample_interval_seconds=2, valid_samples=len(samples), telemetry_errors=errors,
        concurrent_training_overlap_seconds=max(0, end - start), steady_overlap_samples=len(overlap),
        devices=devices, host_total_gib=mem[0]['MemTotal'] / 2**30,
        host_peak_used_gib=max(row['used_bytes'] for row in mem) / 2**30,
        host_min_available_gib=min(row['MemAvailable'] for row in mem) / 2**30,
        host_min_available_fraction=min(row['MemAvailable'] / row['MemTotal'] for row in mem),
        peak_sum_process_tree_rss_gib=max(sum(sample['process_tree_rss_bytes'].values()) for sample in samples) / 2**30,
        cgroup_peak_bytes=max(int(row.get('cgroup_memory.current', 0)) for row in mem),
        cgroup_memory_limit=mem[0].get('cgroup_memory.max'))


def main():
    singles = {name: summarize_run(OUT / name) for name in ('single_V0', 'single_V4')}
    duals = {name: summarize_run(OUT / name) for name in ('dual_V1', 'dual_V2', 'dual_V3', 'dual_V4') if (OUT / name / 'status.json').exists()}
    resource = {'single': resource_summary('single', singles)}
    if duals:
        resource['dual'] = resource_summary('dual', duals)
    reference = singles['single_V4']['status']['mean_iteration_seconds']
    comparisons = {}
    for name, value in duals.items():
        duration = value['status'].get('mean_iteration_seconds')
        if duration:
            comparisons[name] = dict(iteration_time_ratio=duration / reference,
                iteration_time_increase_percent=(duration / reference - 1) * 100,
                per_process_throughput_drop_percent=(1 - reference / duration) * 100,
                reference='single_V4; same P2 computation, representative reference for V1–V3; not mode-matched singles')
    all_passed = len(duals) == 4 and all(value['status']['status'] == 'passed' for value in list(singles.values()) + list(duals.values()))
    resource_pressure = bool(duals) and (any(row['peak_fraction'] >= .9 for row in resource['dual']['devices'])
        or resource['dual']['host_min_available_fraction'] <= .1)
    slow = any(row['iteration_time_ratio'] > 1.3 for row in comparisons.values())
    recommendation = ('one_process_per_device_two_waves' if slow or resource_pressure
                      else 'two_processes_per_device_feasible') if all_passed else 'stop_unstable'
    report = dict(singles=singles, duals=duals, resources=resource, throughput_comparisons=comparisons,
        all_smokes_passed=all_passed, recommendation=recommendation,
        predetermined_resource_thresholds=dict(iteration_time_ratio=1.3, device_memory_fraction=.9, host_min_available_fraction=.1),
        default_gradscaler_preserved=True, initial_scale_changed=False, opl_unchanged=True,
        formal_training_started=False)
    target = ROOT / 'reports/reliability_adaptive_amp_resource_results.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
