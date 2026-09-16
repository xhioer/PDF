"""Low-impact live monitor for the RCHRL-V1 training queue.

The monitor is read-only with respect to training processes.  It records a
heartbeat and flags missing progress, failed jobs, and unexpected queue
states; it never changes checkpoints, relation graphs, or model parameters.
"""
from __future__ import absolute_import

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


RUNS = [
    ("seed0", "R{:02d}".format(i), name)
    for i, name in enumerate([
        "R00_control", "R01_matched_random", "R02_visualhard",
        "R03_semhard", "R04_hybrid", "R05_hybrid_conf",
        "R06_hybrid_agreement", "R07_hybrid_joint", "R08_visual_joint",
        "R09_sem_joint", "R10_hybrid_hardness", "R11_full",
        "R12_full_l005", "R13_full_l020", "R14_positive_only",
        "R15_negative_only",
    ])
] + [
    (seed, run_id, name)
    for seed in ("seed1", "seed2")
    for run_id, name in (
        ("R00", "R00"), ("R04", "R04"), ("R07", "R07"), ("R11", "R11"),
    )
]

EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+\[(\d+)/(\d+)\]")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(str(temporary), str(path))


def active_run_processes(run_id, output):
    matches = []
    needle = str(output)
    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches
    for entry in entries:
        if not entry.isdigit():
            continue
        cmd_path = "/proc/{}/cmdline".format(entry)
        try:
            raw = Path(cmd_path).read_bytes()
            command = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, IOError):
            continue
        if "run_rchrl.py" in command and "--run-id {} ".format(run_id) in command and needle in command:
            matches.append(int(entry))
    return matches


def last_progress(log_path):
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except (OSError, IOError):
        return None
    for line in reversed(lines):
        match = EPOCH_RE.search(line)
        if match:
            return {
                "line": line[-240:],
                "epoch": int(match.group(1)),
                "step": int(match.group(2)),
                "steps_per_epoch": int(match.group(3)),
                "timestamp": line[:23],
            }
    return None


def inspect_run(root, seed, run_id, name, now):
    output = root / seed / name
    final = output / "epoch50_final.pth"
    log_candidates = [output / "train.log", output / "gpu1_dispatch.log"]
    log = next((path for path in log_candidates if path.exists()), output / "train.log")
    progress = last_progress(log)
    mtime = log.stat().st_mtime if log.exists() else None
    pids = active_run_processes(run_id, output)
    if final.exists():
        state = "complete"
    elif pids:
        state = "running"
    elif progress:
        state = "not_running_without_final"
    else:
        state = "pending"
    item = {
        "seed": seed,
        "run_id": run_id,
        "name": name,
        "state": state,
        "final_checkpoint": final.exists(),
        "active_pids": pids,
        "log": str(log),
        "log_mtime": mtime,
        "seconds_since_log_update": None if mtime is None else max(0.0, now - mtime),
        "last_progress": progress,
    }
    if state == "running" and item["seconds_since_log_update"] is not None:
        item["stalled_warning"] = item["seconds_since_log_update"] > 900.0
    else:
        item["stalled_warning"] = False
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    root = Path(args.root)
    reports = root.parent.parent / "reports"
    jsonl_path = reports / "rchrl_v1_live_monitor.jsonl"
    latest_path = reports / "rchrl_v1_live_monitor_status.json"
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop["requested"]:
        now = time.time()
        runs = [inspect_run(root, seed, run_id, name, now) for seed, run_id, name in RUNS]
        active = [item for item in runs if item["state"] == "running"]
        warnings = [item for item in runs if item["stalled_warning"] or item["state"] == "not_running_without_final"]
        payload = {
            "experiment": "RCHRL-V1",
            "timestamp": now,
            "active_run_count": len(active),
            "completed_count": sum(item["state"] == "complete" for item in runs),
            "warnings": warnings,
            "runs": runs,
        }
        reports.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        atomic_json(latest_path, payload)
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
