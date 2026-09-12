"""Dynamic two-device queue for the pre-registered 27 V2 runs."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.identity_preservation_registry import output_name, queue_tasks


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def task_key(variant, seed):
    return '{}_seed{}'.format(variant, seed)


def main(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / 'queue_logs'
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = deque(queue_tasks())
    manifest = dict(
        git_commit=os.popen('git rev-parse HEAD').read().strip(),
        max_concurrent_jobs=2,
        max_jobs_per_device=1,
        devices=[0, 1],
        time_budget_hours=args.time_budget_hours,
        priority=[
            'Priority 1: all seed0 screening N00-N16',
            'Priority 2: N00/N09/N14/N15/N16 seed1',
            'Priority 3: N00/N09/N14/N15/N16 seed2',
        ],
        planned_tasks=[dict(variant=v, seed=s,
                            output=str(output_root / 'seed{}'.format(s) / output_name(v, s)))
                       for v, s in tasks],
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    write_json(output_root / 'queue_manifest.json', manifest)
    status = dict(manifest, status='running', pending=[task_key(v, s) for v, s in tasks],
                  running={}, completed=[], failed=[], not_started=[])
    write_json(output_root / 'queue_status.json', status)

    deadline = time.monotonic() + args.time_budget_hours * 3600.0
    running = {}
    stopped_launching = False
    failure_reason = None
    python = sys.executable

    def refresh_status():
        status['pending'] = [task_key(v, s) for v, s in tasks]
        status['running'] = {str(device): task_key(v, s)
                             for device, (v, s, process, handle) in running.items()}
        status['elapsed_hours'] = (time.monotonic() - start) / 3600.0
        status['budget_exhausted'] = time.monotonic() >= deadline
        status['stopped_launching'] = stopped_launching
        status['failure_reason'] = failure_reason
        write_json(output_root / 'queue_status.json', status)

    start = time.monotonic()
    while tasks or running:
        if time.monotonic() >= deadline:
            stopped_launching = True
        free_devices = [device for device in (0, 1) if device not in running]
        while free_devices and tasks and not stopped_launching:
            device = free_devices.pop(0)
            variant, seed = tasks.popleft()
            run_output = output_root / 'seed{}'.format(seed) / output_name(variant, seed)
            run_output.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_root / (task_key(variant, seed) + '.log')
            handle = log_path.open('w')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(device)
            env['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64' \
                + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            env.setdefault('MKL_NUM_THREADS', '1')
            env.setdefault('OMP_NUM_THREADS', '1')
            command = [python, str(ROOT / 'tools/run_identity_preservation.py'),
                       '--variant', variant, '--seed', str(seed), '--phase', 'train',
                       '--output', str(run_output)]
            process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            running[device] = (variant, seed, process, handle)
            print('START device={} {} pid={} output={}'.format(
                device, task_key(variant, seed), process.pid, run_output), flush=True)
            refresh_status()

        finished = []
        for device, (variant, seed, process, handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            run_output = output_root / 'seed{}'.format(seed) / output_name(variant, seed)
            run_status = {}
            status_path = run_output / 'status.json'
            if status_path.exists():
                try:
                    run_status = json.loads(status_path.read_text())
                except Exception as exc:
                    run_status = {'status': 'unreadable', 'error': repr(exc)}
            row = dict(device=device, variant=variant, seed=seed, return_code=return_code,
                       status=run_status.get('status', 'missing'),
                       output=str(run_output), finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            if return_code == 0 and row['status'] in ('complete', 'passed'):
                status['completed'].append(row)
                print('DONE  device={} {} status={}'.format(device, task_key(variant, seed),
                                                             row['status']), flush=True)
            else:
                status['failed'].append(row)
                print('FAIL  device={} {} return_code={} status={}'.format(
                    device, task_key(variant, seed), return_code, row['status']), flush=True)
                if args.stop_on_failure:
                    stopped_launching = True
                    failure_reason = 'run failure: ' + task_key(variant, seed)
            finished.append(device)
        for device in finished:
            del running[device]
        if not tasks and not running:
            break
        refresh_status()
        time.sleep(5)

    if tasks:
        stopped_launching = True
        failure_reason = failure_reason or 'time budget exhausted before pending tasks started'
        while tasks:
            variant, seed = tasks.popleft()
            status['not_started'].append(dict(variant=variant, seed=seed,
                                              reason=failure_reason))
    status['status'] = 'complete' if not status['failed'] and not status['not_started'] else 'incomplete'
    status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    refresh_status()
    print('QUEUE {} completed={} failed={} not_started={}'.format(
        status['status'], len(status['completed']), len(status['failed']),
        len(status['not_started'])), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default=str(ROOT / 'outputs/pdf_identity_preservation_v2'))
    parser.add_argument('--time-budget-hours', type=float, default=15.0)
    parser.add_argument('--stop-on-failure', action='store_true', default=True)
    args = parser.parse_args()
    raise SystemExit(main(args))
