"""Two-device, one-job-per-device RCHRL-V1 overnight queue."""
from __future__ import absolute_import

import argparse
import json
import os
import subprocess
import sys
import time


def tasks(root):
    result = []
    seed0_names = {
        "R00": "R00_control", "R01": "R01_matched_random", "R02": "R02_visualhard",
        "R03": "R03_semhard", "R04": "R04_hybrid", "R05": "R05_hybrid_conf",
        "R06": "R06_hybrid_agreement", "R07": "R07_hybrid_joint", "R08": "R08_visual_joint",
        "R09": "R09_sem_joint", "R10": "R10_hybrid_hardness", "R11": "R11_full",
        "R12": "R12_full_l005", "R13": "R13_full_l020", "R14": "R14_positive_only",
        "R15": "R15_negative_only",
    }
    for run_id, name in seed0_names.items():
        result.append({"seed": 0, "run_id": run_id,
                       "output": os.path.join(root, "seed0", name)})
    for seed in (1, 2):
        for run_id in ("R00", "R04", "R07", "R11"):
            result.append({"seed": seed, "run_id": run_id,
                           "output": os.path.join(root, "seed{}".format(seed), run_id)})
    return result


def write_status(path, status):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--max-hours", type=float, default=18.0)
    args = parser.parse_args()
    queue = tasks(args.root)
    status_path = os.path.join(args.root, "queue_status.json")
    status = {"experiment": "RCHRL-V1", "max_concurrent_jobs": 2,
              "max_jobs_per_device": 1, "started": time.time(), "tasks": []}
    write_status(status_path, status)
    active = {}
    completed = []
    failed = None
    next_index = 0
    port = 29620
    python = sys.executable
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = args.worktree + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base["LD_LIBRARY_PATH"] = "/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:" + env_base.get("LD_LIBRARY_PATH", "")
    env_base["RCHRL_LOCAL_PRCC_ROOT"] = "/tmp/rchrl_prcc"
    start = time.time()
    while next_index < len(queue) or active:
        if time.time() - start > args.max_hours * 3600:
            failed = {"reason": "runtime_limit_exceeded", "hours": args.max_hours}
            break
        for gpu in args.gpus:
            if gpu in active or next_index >= len(queue) or failed:
                continue
            task = queue[next_index]
            next_index += 1
            os.makedirs(task["output"], exist_ok=True)
            log_path = os.path.join(task["output"], "queue_process.log")
            log_handle = open(log_path, "w")
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [python, "-m", "torch.distributed.run", "--nproc_per_node=1",
                       "--master_port", str(port), os.path.join(args.worktree, "tools", "run_rchrl.py"),
                       "--run-id", task["run_id"], "--seed", str(task["seed"]),
                       "--graph", args.graph, "--output", task["output"], "--phase", "train"]
            port += 1
            process = subprocess.Popen(command, cwd=args.worktree, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            task = dict(task, gpu=str(gpu), pid=process.pid, started=time.time(),
                        status="running", command=command)
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
            completed.append(task)
            del active[gpu]
            if code != 0:
                failed = {"reason": "formal_run_failed", "task": task}
                break
            write_status(status_path, status)
        if failed:
            break
        time.sleep(2.0)
    if failed:
        for process, task, log_handle in active.values():
            process.terminate()
            task["status"] = "terminated_after_failure"
            task["finished"] = time.time()
            log_handle.close()
        status["failure"] = failed
    status["finished"] = time.time()
    status["elapsed_seconds"] = status["finished"] - status["started"]
    status["completed_count"] = len([x for x in status["tasks"] if x.get("status") == "complete"])
    status["queue_length"] = len(queue)
    status["status"] = "complete" if not failed and status["completed_count"] == len(queue) else "failed"
    write_status(status_path, status)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
