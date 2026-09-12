"""Read-only input validation, using the real PDF PRCC loader."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.datasets.prcc import PRCC
from data.semantic_reliability import validate_cache

if __name__ == '__main__':
    dataset = PRCC(root='/data/datasets/PRCC')
    records, report = validate_cache(dataset.train)
    target = ROOT / 'reports/prcc_semantic_cache_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 1)
