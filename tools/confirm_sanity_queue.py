"""Run real AMP sanity updates for the Confirmatory Round variants."""
from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time


TASKS = [("R00", 1), ("R02", 1), ("R08", 1)]


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--master-port", type=int, default=29710)
    args = parser.parse_args()
    sanity_root = os.path.join(args.root, "diagnostics", "sanity")
    os.makedirs(sanity_root, exist_ok=True)
    status_path = os.path.join(sanity_root, "sanity_queue_status.json")
    status = {"experiment": "RCHRL-V1 Confirmatory Round", "seed": 1,
              "tasks": [], "max_concurrent_jobs": 2, "max_jobs_per_device": 1,
              "started": time.time()}
    write_json(status_path, status)
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = args.worktree + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base["LD_LIBRARY_PATH"] = "/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:" + env_base.get("LD_LIBRARY_PATH", "")
    env_base["RCHRL_LOCAL_PRCC_ROOT"] = "/tmp/rchrl_prcc"
    env_base["RCHRL_ORIGINAL_CAPTION"] = "/data/projects/PDF-worktrees/pdf-crossclothes-relation-v1/data/captions/prcc.json"
    pending = []
    for run_id, seed in TASKS:
        output = os.path.join(sanity_root, run_id)
        os.makedirs(output, exist_ok=True)
        if os.path.exists(os.path.join(output, "sanity.json")):
            status["tasks"].append({"run_id": run_id, "seed": seed,
                                     "output": output, "status": "already_complete"})
        else:
            pending.append((run_id, seed, output))
    write_json(status_path, status)
    active = {}
    next_index = 0
    port = args.master_port
    failure = None
    while next_index < len(pending) or active:
        for gpu in args.gpus:
            if gpu in active or next_index >= len(pending) or failure:
                continue
            run_id, seed, output = pending[next_index]
            next_index += 1
            log_path = os.path.join(output, "sanity_process.log")
            log_handle = open(log_path, "w")
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
                       "--master_port", str(port), os.path.join(args.worktree, "tools", "run_rchrl.py"),
                       "--run-id", run_id, "--seed", str(seed), "--graph", args.graph,
                       "--output", output, "--phase", "sanity"]
            port += 1
            process = subprocess.Popen(command, cwd=args.worktree, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            task = {"run_id": run_id, "seed": seed, "gpu": str(gpu), "output": output,
                    "pid": process.pid, "started": time.time(), "status": "running",
                    "command": command}
            active[gpu] = (process, task, log_handle)
            status["tasks"].append(task)
            write_json(status_path, status)
        for gpu, (process, task, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            task["finished"] = time.time()
            task["return_code"] = code
            task["elapsed_seconds"] = task["finished"] - task["started"]
            task["status"] = "complete" if code == 0 else "failed"
            del active[gpu]
            write_json(status_path, status)
            if code != 0:
                failure = task
                break
        if failure:
            break
        time.sleep(2.0)
    if failure:
        for process, task, log_handle in active.values():
            process.terminate()
            task["status"] = "terminated_after_failure"
            log_handle.close()
        status["failure"] = failure
    status["finished"] = time.time()
    status["elapsed_seconds"] = status["finished"] - status["started"]
    status["completed_count"] = sum(x.get("status") in ("complete", "already_complete")
                                     for x in status["tasks"])
    status["status"] = "failed" if failure else (
        "pass" if status["completed_count"] == len(TASKS) else "incomplete")
    write_json(status_path, status)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
