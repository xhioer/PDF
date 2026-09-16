"""Dispatch seed-0 R15 to GPU1 as soon as the orphaned R13 frees it.

R13 was already running independently when the continuation scheduler was
started for R14.  This small orchestration helper preserves the one-job per
device rule, prevents that scheduler from later claiming R15 on GPU0, and
records the dispatch separately from the R14 scheduler status file.
"""
from __future__ import absolute_import

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def write_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def process_is_scheduler(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as handle:
            command = handle.read().decode("utf-8", "replace")
    except (OSError, IOError):
        return False
    return "continue_seed0_priority.py" in command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--scheduler-pid", type=int, required=True)
    parser.add_argument("--master-port", type=int, default=29660)
    args = parser.parse_args()

    output = os.path.join(args.root, "seed0", "R15_negative_only")
    marker = os.path.join(args.root, "seed0", "R13_full_l020", "epoch50_final.pth")
    status_path = os.path.join(args.root, "seed0_r15_gpu1_dispatch_status.json")
    log_path = os.path.join(output, "gpu1_dispatch.log")
    os.makedirs(output, exist_ok=True)
    status = {
        "experiment": "RCHRL-V1",
        "run_id": "R15",
        "seed": 0,
        "gpu": "1",
        "wait_for": marker,
        "scheduler_pid": args.scheduler_pid,
        "status": "waiting_for_R13",
        "started": time.time(),
    }
    write_json(status_path, status)

    while not os.path.exists(marker):
        time.sleep(15)

    # The scheduler currently owns R14 and would claim R15 on GPU0 after R14.
    # Stop only that known scheduler process; its already-running R14 child is
    # allowed to continue as an orphan, as happened with the earlier R13 run.
    if process_is_scheduler(args.scheduler_pid):
        try:
            os.kill(args.scheduler_pid, signal.SIGTERM)
            status["scheduler_terminated"] = True
        except OSError:
            status["scheduler_terminated"] = False
    else:
        status["scheduler_terminated"] = False
    status["status"] = "launching"
    status["dispatch_time"] = time.time()
    write_json(status_path, status)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONPATH"] = args.worktree + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env["RCHRL_LOCAL_PRCC_ROOT"] = "/tmp/rchrl_prcc"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
        "--master_port", str(args.master_port),
        os.path.join(args.worktree, "tools", "run_rchrl.py"),
        "--run-id", "R15", "--seed", "0", "--graph", args.graph,
        "--output", output, "--phase", "train",
    ]
    status["command"] = command
    with open(log_path, "w") as log_handle:
        process = subprocess.Popen(command, cwd=args.worktree, env=env,
                                   stdout=log_handle, stderr=subprocess.STDOUT)
        status["pid"] = process.pid
        status["status"] = "running"
        write_json(status_path, status)
        code = process.wait()
    status["finished"] = time.time()
    status["elapsed_seconds"] = status["finished"] - status["started"]
    status["return_code"] = code
    status["status"] = "complete" if code == 0 else "failed"
    write_json(status_path, status)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
