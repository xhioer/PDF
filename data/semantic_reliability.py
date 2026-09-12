"""Fixed P2 evidence, strict path mapping, and parameter-free reliability.

Reliabilities are descriptive priors, NOT ground-truth probabilities.
Never normalize/repair cache records here: the generated cache is read-only.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

ATTRIBUTES = ('gender', 'hair_color', 'hair_length', 'body_build')
ATTRIBUTE_RELIABILITY = dict(zip(ATTRIBUTES, (0.9753, 0.8925, 0.8453, 0.7731)))
CONFIDENCE_WEIGHT = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
MODES = ('none', 'attribute', 'confidence', 'joint')
EPSILON = 1e-6
SOURCE_ROOT = Path('/data/projects/ccreid-semantic-consistency-exp0')
CACHE_PATH = str(SOURCE_ROOT / 'outputs/exp1_full_prcc_mllm/qwen3vl32b_p2_17896images.jsonl')
SOURCE_CONFIG = SOURCE_ROOT / 'configs/exp1_full_prcc_mllm_config.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def strict_loads(value):
    def reject(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(value, object_pairs_hook=strict_object, parse_constant=reject)


def reliability(description, mode):
    if mode not in MODES:
        raise ValueError('Invalid SEMANTIC_RELIABILITY_MODE: ' + str(mode))
    result = []
    for attribute in ATTRIBUTES:
        value = description[attribute]
        confidence = description[attribute + '_confidence']
        if confidence not in CONFIDENCE_WEIGHT:
            raise ValueError('Invalid confidence: ' + str(confidence))
        weight = 0.0 if value == 'unknown' else 1.0
        if mode in ('attribute', 'joint'):
            weight *= ATTRIBUTE_RELIABILITY[attribute]
        if mode in ('confidence', 'joint'):
            weight *= CONFIDENCE_WEIGHT[confidence]
        result.append(weight)
    return result


def validate_cache(train, cache_path=CACHE_PATH):
    """Use actual PRCC.train tuples. Return full untouched records + report.

    The caller MUST check report['passed'] before using any returned record.
    """
    source = json.loads(SOURCE_CONFIG.read_text())
    fields = set(source['schema_fields'])
    paths = [str(Path(row[0]).resolve()) for row in train]
    train_counts = Counter(paths)
    counts, records, errors = Counter(), {}, []
    invalid_schema = invalid_json = invalid_metadata = total = 0
    for line_number, line in enumerate(Path(cache_path).read_text().splitlines(), 1):
        total += 1
        try:
            row = strict_loads(line)
            if not isinstance(row, dict):
                raise ValueError('record is not an object')
        except (ValueError, TypeError) as exc:
            invalid_json += 1
            errors.append({'line': line_number, 'error': str(exc)})
            continue
        try:
            path = str(Path(row['image_path']).resolve())
            counts[path] += 1
            records[path] = row  # duplicates are fatal below, never accepted
            metadata_ok = (str(row['person_id']) == Path(path).parent.name
                           and row['model'] == 'Qwen3-VL-32B-Instruct'
                           and row['prompt_version'] == 'P2_best_estimate_confidence'
                           and row['parse_ok'] is True and row['schema_valid'] is True
                           and row['parse_mode'] == 'exact')
            if not metadata_ok:
                invalid_metadata += 1
            description = row['description']
            raw = strict_loads(row['raw_model_output'])
            for payload in (description, raw):
                if not isinstance(payload, dict) or set(payload) != fields:
                    raise ValueError('incorrect schema fields')
                if not all(isinstance(value, str) for value in payload.values()):
                    raise ValueError('schema values must all be strings')
            for attribute in ATTRIBUTES:
                if not description[attribute]:
                    raise ValueError('empty normalized attribute')
                if description[attribute + '_confidence'] not in CONFIDENCE_WEIGHT:
                    raise ValueError('invalid normalized confidence')
        except (KeyError, ValueError, TypeError) as exc:
            invalid_schema += 1
            errors.append({'line': line_number, 'error': str(exc)})
    missing = sorted(set(paths) - set(counts))
    extra = sorted(set(counts) - set(paths))
    duplicate = sum(n - 1 for n in counts.values())
    train_duplicate = sum(n - 1 for n in train_counts.values())
    report = dict(train_images=len(paths), semantic_records=total,
                  matched_unique=len(set(paths) & set(counts)), missing=len(missing),
                  duplicate=duplicate, train_duplicate=train_duplicate,
                  extra=len(extra), invalid_schema=invalid_schema,
                  invalid_json=invalid_json, invalid_metadata=invalid_metadata,
                  missing_paths=missing, extra_paths=extra, errors=errors,
                  cache_path=str(cache_path), cache_sha256=sha256(cache_path),
                  source_config_sha256=sha256(SOURCE_CONFIG),
                  retained_fields=['description', 'raw_model_output', 'image_path', 'person_id'],
                  mapping='resolved full path, exact person directory, PRCC.train tuples',
                  no_missing_value_imputation=True)
    report['passed'] = (len(paths) == total == 17896 and not any(
        (missing, extra, duplicate, train_duplicate, invalid_schema, invalid_json, invalid_metadata)))
    return records, report


def aggregate_identity(embeddings, weights):
    """[B,4,D] and [B,4], float32 accumulation, zero for no evidence."""
    return (embeddings.float() * weights.float().unsqueeze(-1)).sum(1) / (
        weights.float().sum(1, keepdim=True) + EPSILON)


def attribute_phrase(attribute, value):
    if attribute not in ATTRIBUTES or value == 'unknown':
        raise ValueError('Only valid identity attributes can be encoded')
    return 'A person with {}: {}.'.format(attribute.replace('_', ' '), value)
