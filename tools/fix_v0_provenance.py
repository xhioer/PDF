"""Create a provenance-corrected copy when the repository advanced mid-run."""
from __future__ import absolute_import

import argparse
import os

import torch

from tools.rchrl_common import REPO_ROOT, json_dump, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-commit", required=True,
                        help="commit checked out when the V0 process was launched")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("epoch") != 50 or not checkpoint.get("train_only"):
        raise RuntimeError("input is not an epoch50 train-only checkpoint")
    corrected = os.path.splitext(args.checkpoint)[0] + ".verified.pth"
    checkpoint["source_commit"] = args.source_commit
    checkpoint["source_commit_at_launch"] = args.source_commit
    checkpoint["provenance_correction"] = "repository advanced after launch; model weights unchanged"
    torch.save(checkpoint, corrected)
    provenance_path = os.path.join(REPO_ROOT, "reports", "v0_mining_checkpoint_provenance.json")
    with open(provenance_path) as handle:
        provenance = __import__("json").load(handle)
    provenance["raw_checkpoint_path"] = args.checkpoint
    provenance["raw_checkpoint_sha256"] = provenance.get("checkpoint_sha256")
    provenance["checkpoint_path"] = corrected
    provenance["checkpoint_sha256"] = sha256_file(corrected)
    provenance["source_commit"] = args.source_commit
    provenance["source_commit_at_launch"] = args.source_commit
    provenance["provenance_correction"] = "verified copy created with the exact launch commit; model state unchanged"
    json_dump(provenance, provenance_path)
    summary_path = os.path.join(os.path.dirname(args.checkpoint), "v0_training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            summary = __import__("json").load(handle)
        summary["raw_checkpoint_path"] = args.checkpoint
        summary["checkpoint"] = corrected
        summary["checkpoint_sha256"] = provenance["checkpoint_sha256"]
        json_dump(summary, summary_path)
    print("verified_checkpoint={} sha256={}".format(corrected, provenance["checkpoint_sha256"]))


if __name__ == "__main__":
    main()
