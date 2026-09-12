"""Bounded default-AMP smoke matrix; NEVER launches phase=train.

Singles are one process per physical card; only after both pass, launch pairs.
Each child gets independent model/optimizer/rendezvous/output/logs.
"""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/pdf_reliability_ablation/preflight/adaptive_amp_v2'
PYTHON = '/data/envs/PDF/bin/python'
SMI = '/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi'


def host_memory():
    info = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    result = {key: info[key] for key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    result['used_bytes'] = info['MemTotal'] - info['MemAvailable']
    for name in ('memory.current', 'memory.max', 'memory.peak'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result['cgroup_' + name] = path.read_text().strip()
    return result


def process_rss(pids):
    lines = subprocess.check_output(['ps', '-e', '-o', 'pid=,ppid=,rss='], text=True).splitlines()
    rows = [tuple(map(int, line.split())) for line in lines]
    results = {}
    for root_pid in pids:
        descendants = {root_pid}
        while True:
            expanded = descendants | {pid for pid, parent, rss in rows if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        results[str(root_pid)] = sum(rss * 1024 for pid, parent, rss in rows if pid in descendants)
    return results


def sample(processes):
    query = subprocess.check_output([SMI, '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    gpus = []
    for fields in csv.reader(io.StringIO(query)):
        values = [int(value.strip()) for value in fields]
        gpus.append(dict(device=values[0], used_mib=values[1], total_mib=values[2], utilization_percent=values[3]))
    return dict(timestamp=time.time(), gpus=gpus, host_memory=host_memory(),
                process_tree_rss_bytes=process_rss([process.pid for process in processes.values()]),
                live=[name for name, process in processes.items() if process.poll() is None])


def stage(name, jobs):
    processes, handles, outputs = {}, [], {}
    for label, device, cfg, v0 in jobs:
        target = OUT / label
        outputs[label] = target
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONDONTWRITEBYTECODE='1',
            LD_LIBRARY_PATH='/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [PYTHON, '-u', 'tools/run_reliability.py', '--cfg', 'configs/' + cfg,
                   '--phase', 'smoke', '--seconds', '300', '--output', str(target)]
        if v0:
            command.append('--v0')
        handle = (OUT / (label + '.launcher.log')).open('w')
        handles.append(handle)
        processes[label] = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        print('START', label, 'device', device, 'pid', processes[label].pid, flush=True)
    telemetry = OUT / (name + '_resources.jsonl')
    with telemetry.open('w') as handle:
        while any(process.poll() is None for process in processes.values()):
            try:
                record = sample(processes)
            except Exception as exc:
                record = dict(timestamp=time.time(), telemetry_error=repr(exc))
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            time.sleep(2)
    results = {}
    for label, process in processes.items():
        path = outputs[label] / 'status.json'
        results[label] = json.loads(path.read_text()) if path.exists() else {'status': 'failed', 'reason': 'no status file'}
        results[label]['exit_code'] = process.returncode
        print('FINISH', label, json.dumps(results[label]), flush=True)
    for handle in handles:
        handle.close()
    (OUT / (name + '_summary.json')).write_text(json.dumps(results, indent=2) + '\n')
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    singles = stage('single', [('single_V0', 0, 'v1_p2_none.py', True),
                               ('single_V4', 1, 'v4_p2_joint.py', False)])
    if not all(row['status'] == 'passed' and row['exit_code'] == 0 for row in singles.values()):
        print('STOP: single-process stability gate failed; dual-process tests not launched', flush=True)
        return
    stage('dual', [('dual_V1', 0, 'v1_p2_none.py', False), ('dual_V2', 0, 'v2_p2_attribute.py', False),
                   ('dual_V3', 1, 'v3_p2_confidence.py', False), ('dual_V4', 1, 'v4_p2_joint.py', False)])
    print('All bounded tests finished. No formal training was launched.', flush=True)


if __name__ == '__main__':
    main()
