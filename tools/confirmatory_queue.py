"""Two-device queue for the six Confirmatory Round runs."""
from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time


TASKS = [
    {"seed": 1, "run_id": "R00"}, {"seed": 1, "run_id": "R02"},
    {"seed": 1, "run_id": "R08"}, {"seed": 2, "run_id": "R00"},
    {"seed": 2, "run_id": "R02"}, {"seed": 2, "run_id": "R08"},
]


def write_status(path, status):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def output_path(root, seed, run_id):
    return os.path.join(root, "seed{}".format(seed), run_id)


def is_complete(output):
    return os.path.exists(os.path.join(output, "epoch50_final.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--master-port", type=int, default=29720)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--config-diff", required=True)
    args = parser.parse_args()

    with open(args.config_diff) as handle:
        config_gate = json.load(handle)
    if config_gate.get("status") != "pass" or config_gate.get("diffs"):
        raise SystemExit("config consistency gate is not pass; no training launched")

    status_path = os.path.join(args.root, "confirmatory_queue_status.json")
    status = {
        "experiment": "RCHRL-V1 Confirmatory Round",
        "max_concurrent_jobs": 2,
        "max_jobs_per_device": 1,
        "max_hours": args.max_hours,
        "started": time.time(),
        "tasks": [],
        "task_order": TASKS,
        "no_dynamic_remining": True,
        "test_data_used_for_mining_or_training": False,
    }
    write_status(status_path, status)
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = args.worktree + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base["LD_LIBRARY_PATH"] = "/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:" + env_base.get("LD_LIBRARY_PATH", "")
    env_base["RCHRL_LOCAL_PRCC_ROOT"] = "/tmp/rchrl_prcc"
    env_base["RCHRL_ORIGINAL_CAPTION"] = "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/data/captions/prcc.json"
    python = sys.executable
    pending = []
    for task in TASKS:
        output = output_path(args.root, task["seed"], task["run_id"])
        entry = dict(task, output=output)
        if is_complete(output):
            entry["status"] = "already_complete"
            status["tasks"].append(entry)
        else:
            pending.append(entry)
    write_status(status_path, status)

    active = {}
    next_index = 0
    port = args.master_port
    failure = None
    start = time.time()
    while next_index < len(pending) or active:
        gate = time.time() - start >= args.max_hours * 3600
        status["runtime_gate_reached"] = gate
        for gpu in args.gpus:
            if gpu in active or next_index >= len(pending) or failure or gate:
                continue
            task = pending[next_index]
            next_index += 1
            os.makedirs(task["output"], exist_ok=True)
            log_path = os.path.join(task["output"], "queue_process.log")
            log_handle = open(log_path, "w")
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                python, "-m", "torch.distributed.run", "--nproc_per_node=1",
                "--master_port", str(port), os.path.join(args.worktree, "tools", "run_rchrl.py"),
                "--run-id", task["run_id"], "--seed", str(task["seed"]),
                "--graph", args.graph, "--output", task["output"], "--phase", "train",
            ]
            port += 1
            process = subprocess.Popen(command, cwd=args.worktree, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            task.update({"gpu": str(gpu), "pid": process.pid, "started": time.time(),
                         "status": "running", "command": command})
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
            write_status(status_path, status)
            if code != 0:
                failure = dict(task)
                break
        if failure:
            break
        if gate and not active:
            break
        time.sleep(2.0)

    if failure:
        for process, task, log_handle in active.values():
            process.terminate()
            task["status"] = "terminated_after_failure"
            task["finished"] = time.time()
            log_handle.close()
        status["failure"] = failure
    status["finished"] = time.time()
    status["elapsed_seconds"] = status["finished"] - status["started"]
    status["next_queue_index"] = next_index
    status["pending_after_gate"] = pending[next_index:]
    status["completed_count"] = sum(task.get("status") in ("complete", "already_complete")
                                     for task in status["tasks"])
    status["requested_count"] = len(TASKS)
    if failure:
        status["status"] = "failed"
    elif status["completed_count"] == len(TASKS):
        status["status"] = "complete"
    elif status.get("runtime_gate_reached"):
        status["status"] = "runtime_gate_stop"
    else:
        status["status"] = "incomplete"
    write_status(status_path, status)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
