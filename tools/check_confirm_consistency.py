"""Check Confirmatory Round protocol consistency against seed0 RCHRL-V1."""
from __future__ import absolute_import

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from tools.run_rchrl import RUNS, make_config
from tools.rchrl_common import config_snapshot, sha256_file, source_hashes


RUN_NAMES = {"R00": "R00_control", "R02": "R02_visualhard", "R08": "R08_visual_joint"}


def normalized_snapshot(value):
    # Config differences in seed and output path are explicitly allowed.  All
    # other fields remain byte-for-byte comparable after line normalization.
    kept = []
    for line in value.splitlines():
        if re.match(r"^SEED:\s", line) or re.match(r"^OUTPUT:\s", line):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip() + "\n"


def file_sha_from_git(worktree, relative):
    data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=worktree)
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--previous-worktree", required=True)
    parser.add_argument("--confirm-root", required=True)
    parser.add_argument("--confirm-worktree", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    diffs = {}
    snapshots = {}
    specs = {}
    previous_core = {}
    current_core = source_hashes(args.confirm_worktree)
    previous_run_provenance = {}
    for run_id, name in RUN_NAMES.items():
        previous_path = Path(args.previous_root) / "seed0" / name / "run_provenance.json"
        if not previous_path.exists():
            diffs[run_id + ".previous_provenance"] = "missing: {}".format(previous_path)
            continue
        previous = json.load(open(str(previous_path)))
        previous_run_provenance[run_id] = previous
        previous_snapshot = previous.get("config_snapshot", "")
        current_output = os.path.join(args.confirm_root, "seed1", run_id)
        current_snapshot = config_snapshot(make_config(1, current_output))
        snapshots[run_id] = {
            "previous_normalized": normalized_snapshot(previous_snapshot),
            "current_normalized": normalized_snapshot(current_snapshot),
            "match_except_seed_output": normalized_snapshot(previous_snapshot) == normalized_snapshot(current_snapshot),
        }
        if not snapshots[run_id]["match_except_seed_output"]:
            diffs[run_id + ".config_snapshot"] = {
                "previous": snapshots[run_id]["previous_normalized"],
                "current": snapshots[run_id]["current_normalized"],
            }
        previous_spec = dict(previous.get("spec", {}))
        current_spec = dict(RUNS[run_id])
        current_spec["run_id"] = run_id
        specs[run_id] = {"previous": previous_spec, "current": current_spec,
                         "match": previous_spec == current_spec}
        if previous_spec != current_spec:
            diffs[run_id + ".run_spec"] = {"previous": previous_spec, "current": current_spec}
        previous_core[run_id] = previous.get("source_core_file_sha256", {})
        if previous_core[run_id] != current_core:
            diffs[run_id + ".core_file_sha256"] = {"previous": previous_core[run_id], "current": current_core}

    previous_runner_sha = file_sha_from_git(args.previous_worktree, "tools/run_rchrl.py")
    current_runner_sha = sha256_file(os.path.join(args.confirm_worktree, "tools/run_rchrl.py"))
    if previous_runner_sha != current_runner_sha:
        diffs["tools/run_rchrl.py"] = {"previous": previous_runner_sha, "current": current_runner_sha}

    graph_ledger_path = Path(args.graph) / "relation_graph_hashes.json"
    current_ledger = json.load(open(str(graph_ledger_path)))
    graph_comparisons = {}
    for run_id, previous in previous_run_provenance.items():
        previous_ledger = (previous.get("graph_ledger") or {}).get("files", {})
        current_files = current_ledger.get("files", {})
        graph_comparisons[run_id] = {
            "previous": previous_ledger,
            "current": current_files,
            "match": previous_ledger == current_files,
        }
        if previous_ledger != current_files:
            diffs[run_id + ".graph_ledger"] = {"previous": previous_ledger, "current": current_files}

    payload = {
        "experiment": "RCHRL-V1 Confirmatory Round",
        "previous_worktree": args.previous_worktree,
        "confirm_worktree": args.confirm_worktree,
        "previous_root": args.previous_root,
        "confirm_root": args.confirm_root,
        "graph": args.graph,
        "allowed_differences": ["seed", "output path", "master port (launcher metadata only)"],
        "config_snapshots": snapshots,
        "run_specs": specs,
        "core_file_sha256_current": current_core,
        "runner_sha256_previous": previous_runner_sha,
        "runner_sha256_current": current_runner_sha,
        "runner_match": previous_runner_sha == current_runner_sha,
        "graph_ledger_comparisons": graph_comparisons,
        "diffs": diffs,
        "status": "pass" if not diffs else "STOP",
        "training_authorized": not diffs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "status": payload["status"],
                      "diff_count": len(diffs)}, indent=2))
    if diffs:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
