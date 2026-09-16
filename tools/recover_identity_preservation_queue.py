"""Resume the V2 queue after a scheduler-only false nonzero exit.

The training runner writes an authoritative status.json before Python/PPU
shutdown.  In this environment a healthy runner can occasionally exit 120;
this recovery scheduler accepts that code only when the run status is already
complete, and otherwise follows the original stop-on-failure policy.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.identity_preservation_registry import output_name, queue_tasks


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def task_key(variant, seed):
    return '{}_seed{}'.format(variant, seed)


def read_run_status(output_root, variant, seed):
    output = output_root / 'seed{}'.format(seed) / output_name(variant, seed)
    path = output / 'status.json'
    if not path.exists():
        return output, {}
    try:
        return output, json.loads(path.read_text())
    except Exception as exc:
        return output, {'status': 'unreadable', 'error': repr(exc)}


def main(args):
    output_root = Path(args.output_root)
    queue_status_path = output_root / 'queue_status.json'
    status = json.loads(queue_status_path.read_text())
    log_root = output_root / 'queue_logs'
    log_root.mkdir(parents=True, exist_ok=True)

    # Promote runs that completed training but were mislabeled by the first
    # scheduler because the process returned 120 during interpreter teardown.
    normalized_failed = []
    for row in status.get('failed', []):
        _, run_status = read_run_status(output_root, row['variant'], row['seed'])
        if (row.get('return_code') == 120 and
                run_status.get('status') in ('complete', 'passed')):
            row = dict(row)
            row['scheduler_recovered'] = True
            row['accepted_return_code'] = 120
            status.setdefault('completed', []).append(row)
        else:
            normalized_failed.append(row)
    status['failed'] = normalized_failed

    completed_keys = {
        task_key(row['variant'], row['seed'])
        for row in status.get('completed', [])
    }
    tasks = deque(
        (variant, seed) for variant, seed in queue_tasks()
        if task_key(variant, seed) not in completed_keys
    )

    started = datetime.strptime(
        status['started_utc'], '%Y-%m-%dT%H:%M:%SZ'
    ).replace(tzinfo=timezone.utc)
    deadline = started + timedelta(hours=float(status['time_budget_hours']))
    status['status'] = 'running'
    status['recovery_started_utc'] = datetime.now(timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )
    status['failure_reason'] = None
    status['stopped_launching'] = False
    status['not_started'] = []
    running = {}
    python = sys.executable

    def refresh_status():
        now = datetime.now(timezone.utc)
        status['pending'] = [task_key(v, s) for v, s in tasks]
        status['running'] = {
            str(device): task_key(v, s)
            for device, (v, s, process, handle) in running.items()
        }
        status['elapsed_hours'] = max(
            0.0, (now - started).total_seconds() / 3600.0
        )
        status['budget_exhausted'] = now >= deadline
        write_json(queue_status_path, status)

    refresh_status()
    while tasks or running:
        now = datetime.now(timezone.utc)
        if now >= deadline:
            status['stopped_launching'] = True
            status['failure_reason'] = (
                status['failure_reason'] or
                'time budget exhausted before pending tasks started'
            )

        free_devices = [device for device in (0, 1) if device not in running]
        while free_devices and tasks and not status['stopped_launching']:
            device = free_devices.pop(0)
            variant, seed = tasks.popleft()
            run_output = output_root / 'seed{}'.format(seed) / output_name(variant, seed)
            run_output.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_root / (task_key(variant, seed) + '.log')
            handle = log_path.open('w')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(device)
            env['LD_LIBRARY_PATH'] = (
                '/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64' +
                (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            )
            env.setdefault('MKL_NUM_THREADS', '1')
            env.setdefault('OMP_NUM_THREADS', '1')
            command = [
                python, str(ROOT / 'tools/run_identity_preservation.py'),
                '--variant', variant, '--seed', str(seed), '--phase', 'train',
                '--output', str(run_output),
            ]
            process = subprocess.Popen(
                command, cwd=str(ROOT), env=env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            running[device] = (variant, seed, process, handle)
            print('START device={} {} pid={} output={}'.format(
                device, task_key(variant, seed), process.pid, run_output
            ), flush=True)
            refresh_status()

        finished = []
        for device, (variant, seed, process, handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            run_output, run_status = read_run_status(output_root, variant, seed)
            row = dict(
                device=device, variant=variant, seed=seed,
                return_code=return_code,
                status=run_status.get('status', 'missing'),
                output=str(run_output),
                finished_utc=datetime.now(timezone.utc).strftime(
                    '%Y-%m-%dT%H:%M:%SZ'
                ),
            )
            accepted = (
                return_code in (0, 120) and
                row['status'] in ('complete', 'passed')
            )
            if accepted:
                if return_code == 120:
                    row['scheduler_recovered'] = True
                    row['accepted_return_code'] = 120
                status.setdefault('completed', []).append(row)
                print('DONE  device={} {} status={} return_code={}'.format(
                    device, task_key(variant, seed), row['status'], return_code
                ), flush=True)
            else:
                status.setdefault('failed', []).append(row)
                status['stopped_launching'] = True
                status['failure_reason'] = 'run failure: ' + task_key(variant, seed)
                print('FAIL  device={} {} return_code={} status={}'.format(
                    device, task_key(variant, seed), return_code,
                    row['status']
                ), flush=True)
            finished.append(device)
        for device in finished:
            del running[device]

        if status['stopped_launching'] and not running:
            while tasks:
                variant, seed = tasks.popleft()
                status['not_started'].append(dict(
                    variant=variant, seed=seed,
                    reason=status['failure_reason'],
                ))
        refresh_status()
        if tasks or running:
            time.sleep(5)

    status['status'] = (
        'complete' if not status.get('failed') and not status.get('not_started')
        else 'incomplete'
    )
    status['finished_utc'] = datetime.now(timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )
    refresh_status()
    print('QUEUE {} completed={} failed={} not_started={}'.format(
        status['status'], len(status.get('completed', [])),
        len(status.get('failed', [])), len(status.get('not_started', []))
    ), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-root',
        default=str(ROOT / 'outputs/pdf_identity_preservation_v2'),
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
