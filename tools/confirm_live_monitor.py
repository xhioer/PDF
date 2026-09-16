"""Read-only heartbeat monitor for the six Confirmatory Round runs."""
from __future__ import absolute_import

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path


TASKS = [(1, "R00"), (1, "R02"), (1, "R08"),
         (2, "R00"), (2, "R02"), (2, "R08")]
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+\[(\d+)/(\d+)\]")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(str(temporary), str(path))


def pids_for(seed, run_id, output):
    result = []
    needle = str(output)
    try:
        entries = os.listdir("/proc")
    except OSError:
        return result
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc/{}/cmdline".format(entry)).read_bytes().replace(
                b"\0", b" ").decode("utf-8", "replace")
        except (OSError, IOError):
            continue
        if ("run_rchrl.py" in command and "--run-id {} ".format(run_id) in command
                and "--seed {} ".format(seed) in command and needle in command):
            result.append(int(entry))
    return result


def inspect(root, seed, run_id, now):
    output = root / "seed{}".format(seed) / run_id
    log = output / "train.log"
    final = output / "epoch50_final.pth"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    progress = None
    for line in reversed(lines):
        match = EPOCH_RE.search(line)
        if match:
            progress = {"line": line[-240:], "epoch": int(match.group(1)),
                        "step": int(match.group(2)), "steps_per_epoch": int(match.group(3))}
            break
    mtime = log.stat().st_mtime if log.exists() else None
    pids = pids_for(seed, run_id, output)
    if final.exists():
        state = "complete"
    elif pids:
        state = "running"
    elif progress:
        state = "failed_or_stopped"
    else:
        state = "pending"
    return {
        "seed": seed, "run_id": run_id, "output": str(output), "state": state,
        "final_checkpoint": final.exists(), "active_pids": pids,
        "last_progress": progress, "log_mtime": mtime,
        "seconds_since_log_update": None if mtime is None else max(0.0, now - mtime),
        "stalled_warning": bool(pids and mtime and now - mtime > 900.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    root = Path(args.root)
    reports = root.parent.parent / "reports"
    latest = reports / "confirm_live_monitor_status.json"
    history = reports / "confirm_live_monitor.jsonl"
    stopping = {"value": False}

    def stop(signum, frame):
        stopping["value"] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping["value"]:
        now = time.time()
        runs = [inspect(root, seed, run_id, now) for seed, run_id in TASKS]
        payload = {
            "experiment": "RCHRL-V1 Confirmatory Round",
            "timestamp": now,
            "active_run_count": sum(x["state"] == "running" for x in runs),
            "completed_count": sum(x["state"] == "complete" for x in runs),
            "warnings": [x for x in runs if x["stalled_warning"] or x["state"] == "failed_or_stopped"],
            "runs": runs,
        }
        reports.mkdir(parents=True, exist_ok=True)
        with history.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        atomic_json(latest, payload)
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
