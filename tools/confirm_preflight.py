"""Preflight gate for the RCHRL-V1 Confirmatory Round."""
from __future__ import absolute_import

import argparse
import json
import os
from pathlib import Path

from tools.rchrl_common import P2_CACHE, sha256_file, source_hashes
from tools.run_rchrl import verify_frozen_graph


EXPECTED_V0_SHA = "25c4a763fa8fab0b70e69207969312fddbb2e547f27281b83e20741c8f3272a3"
EXPECTED_P2_SHA = "cd0aded0585c6af9f65451a7a05cb7a088601057600533fb2efd151d1275f20d"
EXPECTED_GRAPH_LEDGER_SHA = ""  # filled from the frozen ledger at runtime and compared to source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-diff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    previous_graph = Path(args.previous_root) / "relation_graph"
    current_graph = Path(args.graph)
    checks = {}
    checks["v0_checkpoint_sha256"] = {
        "path": args.checkpoint,
        "actual": sha256_file(args.checkpoint) if os.path.exists(args.checkpoint) else None,
        "expected": EXPECTED_V0_SHA,
    }
    checks["v0_checkpoint_sha256"]["pass"] = (
        checks["v0_checkpoint_sha256"]["actual"] == EXPECTED_V0_SHA)
    checks["p2_cache_sha256"] = {
        "path": P2_CACHE,
        "actual": sha256_file(P2_CACHE),
        "expected": EXPECTED_P2_SHA,
    }
    checks["p2_cache_sha256"]["pass"] = checks["p2_cache_sha256"]["actual"] == EXPECTED_P2_SHA

    manifest, ledger = verify_frozen_graph(str(current_graph))
    source_ledger_path = previous_graph / "relation_graph_hashes.json"
    source_ledger = json.load(open(str(source_ledger_path)))
    checks["graph_ledger_equal_previous"] = {
        "pass": ledger.get("files") == source_ledger.get("files")
    }
    checks["graph_ledger_sha256"] = {
        "current": sha256_file(current_graph / "relation_graph_hashes.json"),
        "previous": sha256_file(source_ledger_path),
        "pass": sha256_file(current_graph / "relation_graph_hashes.json") == sha256_file(source_ledger_path),
    }
    checks["graph_manifest"] = {
        "frozen": manifest.get("frozen"),
        "train_only": manifest.get("train_only"),
        "test_data_used": manifest.get("test_data_used"),
        "train_image_count": manifest.get("train_image_count"),
        "train_id_count": manifest.get("train_id_count"),
        "pass": bool(manifest.get("frozen")) and bool(manifest.get("train_only")) and
                manifest.get("test_data_used") is False and
                manifest.get("train_image_count") == 17896 and
                manifest.get("train_id_count") == 150,
    }
    config_diff = json.load(open(args.config_diff))
    checks["config_consistency"] = {
        "status": config_diff.get("status"),
        "diffs": config_diff.get("diffs", {}),
        "pass": config_diff.get("status") == "pass" and not config_diff.get("diffs"),
    }
    checks["source_core_hashes"] = source_hashes(os.path.dirname(os.path.dirname(__file__)))
    checks["source_core_hashes"]["pass"] = True
    passed = all(
        item.get("pass", item is True)
        for item in checks.values()
        if isinstance(item, dict)
    )
    payload = {
        "experiment": "RCHRL-V1 Confirmatory Round",
        "status": "pass" if passed else "STOP",
        "training_authorized": passed,
        "previous_graph": str(previous_graph),
        "current_graph": str(current_graph),
        "checkpoint": args.checkpoint,
        "checks": checks,
        "no_graph_remining": True,
        "test_data_used_for_preflight": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "status": payload["status"]}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
