"""Continue the explicitly requested seed-0 priority tail (R13-R15).

This is intentionally separate from the 18-hour overnight queue.  It is used
only after the gate has stopped the original queue, and never launches
seed-1/seed-2 jobs or re-mines the frozen relation graph.
"""
from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time


TASKS = [
    ("R13", "R13_full_l020"),
    ("R14", "R14_positive_only"),
    ("R15", "R15_negative_only"),
]


def write_status(path, status):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def is_complete(output):
    return os.path.exists(os.path.join(output, "epoch50_final.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--run-ids", nargs="+", choices=[run_id for run_id, _ in TASKS],
                        default=[run_id for run_id, _ in TASKS])
    parser.add_argument("--master-port", type=int, default=29640)
    parser.add_argument("--max-hours", type=float, default=18.0)
    args = parser.parse_args()

    requested_tasks = [(run_id, name) for run_id, name in TASKS
                       if run_id in args.run_ids]

    status_path = os.path.join(args.root, "seed0_priority_continuation_status.json")
    status = {
        "experiment": "RCHRL-V1",
        "purpose": "explicit seed0 R13-R15 completion after runtime gate",
        "priority": "seed0 Priority 1",
        "max_concurrent_jobs": 2,
        "max_jobs_per_device": 1,
        "started": time.time(),
        "tasks": [],
    }
    write_status(status_path, status)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = args.worktree + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base["LD_LIBRARY_PATH"] = "/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:" + env_base.get("LD_LIBRARY_PATH", "")
    env_base["RCHRL_LOCAL_PRCC_ROOT"] = "/tmp/rchrl_prcc"
    python = sys.executable
    pending = []
    for run_id, name in requested_tasks:
        output = os.path.join(args.root, "seed0", name)
        if is_complete(output):
            status["tasks"].append({"run_id": run_id, "seed": 0, "output": output,
                                     "status": "already_complete"})
        else:
            pending.append((run_id, output))
    write_status(status_path, status)

    active = {}
    next_index = 0
    port = args.master_port
    failure = None
    start = time.time()
    while next_index < len(pending) or active:
        # Unlike the overnight queue, this continuation is explicitly scoped
        # to the remaining seed-0 priority-1 tasks.  The wall-clock value is
        # recorded, but it must not silently drop R13-R15 after user direction.
        if time.time() - start >= args.max_hours * 3600:
            status["wall_clock_limit_reached"] = True

        for gpu in args.gpus:
            if gpu in active or next_index >= len(pending) or failure:
                continue
            run_id, output = pending[next_index]
            next_index += 1
            os.makedirs(output, exist_ok=True)
            log_path = os.path.join(output, "continuation_queue_process.log")
            log_handle = open(log_path, "w")
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                python, "-m", "torch.distributed.run", "--nproc_per_node=1",
                "--master_port", str(port), os.path.join(args.worktree, "tools", "run_rchrl.py"),
                "--run-id", run_id, "--seed", "0", "--graph", args.graph,
                "--output", output, "--phase", "train",
            ]
            port += 1
            process = subprocess.Popen(command, cwd=args.worktree, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            task = {"run_id": run_id, "seed": 0, "gpu": str(gpu),
                    "output": output, "pid": process.pid,
                    "started": time.time(), "status": "running",
                    "command": command}
            active[gpu] = (process, task, log_handle)
            status["tasks"].append(task)
            write_status(status_path, status)

        for gpu, (process, task, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            task["finished"] = time.time()
            task["elapsed_seconds"] = task["finished"] - task["started"]
            task["return_code"] = code
            task["status"] = "complete" if code == 0 else "failed"
            del active[gpu]
            if code != 0:
                failure = task
                break
            write_status(status_path, status)
        if failure:
            break
        time.sleep(2.0)

    if failure:
        status["failure"] = failure
        for process, task, log_handle in active.values():
            process.terminate()
            task["status"] = "terminated_after_failure"
            task["finished"] = time.time()
            log_handle.close()
    status["finished"] = time.time()
    status["elapsed_seconds"] = status["finished"] - status["started"]
    status["completed_count"] = sum(
        task.get("status") in ("complete", "already_complete")
        for task in status["tasks"]
    )
    status["requested_run_ids"] = [run_id for run_id, _ in requested_tasks]
    status["requested_count"] = len(requested_tasks)
    status["status"] = "failed" if failure else (
        "complete" if status["completed_count"] == len(requested_tasks) else "incomplete"
    )
    write_status(status_path, status)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
