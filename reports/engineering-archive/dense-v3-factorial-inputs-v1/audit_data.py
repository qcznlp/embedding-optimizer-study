"""Full fixed-branch/calibration-data relation to the actual revised primary view."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from datasets import Dataset


STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
REVISED = Path('/root/embedding-optimizer-v3-experiment/data/denseon-sft-500k-v3')
BRANCH = Path('/root/embedding-optimizer-study/data/short-branch-50k-seed20260826')
PROBE = Path('/root/embedding-optimizer-study/data/probes/training-1024-seed1729')
SPLITS = ('fiqa', 'hotpotqa', 'msmarco', 'nq', 'fever', 'squadv2', 'trivia')
PINNED = {
    'configs/dense_no_packing_state_operator_factorial_protocol.json':
        '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76',
    'configs/dense_no_packing_state_operator_factorial_implementation_protocol.json':
        '7573a28781735782ec1ec527cfd5d70b01a97ba14e081df77a91104a3a066d97',
    'configs/dense_no_packing_state_operator_claim_wording_amendment.json':
        '929fe6445598b84b546670eb5eefd408c5221cacad6cc8e93a984651dbbe5de7',
    'configs/short_branch_protocol.json':
        'd3dc498f5e952d09d1423a53dc36499b638366d23e3876bb81aadb7e88369bec',
    'configs/common_state_probe.json':
        'ac0661b1077c5413b282adc5d491a5b26343a2ba499cc4b16cb452f595160c60',
    'configs/dense_primary_v3_data_amendment.json':
        '5051ec40c7e44ba677980fec422ed4a49ef851ef3e9fff9c29cc0f6fc3d8e4e1',
    'reports/dense-primary-v3/input-bindings.json':
        '40b7d3e63b1c4a17b995507d5d96c042980acca96baa08f6c0c52dda66b1f1f9',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Require ordinary input file: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(row):
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'


def direct_branch_selection(by_source, quotas, count=50000, seed=20260826):
    total = sum(quotas.values())
    exact = {source: Fraction(count * amount, total) for source, amount in quotas.items()}
    target = {source: math.floor(value) for source, value in exact.items()}
    remainder_order = sorted(quotas, key=lambda source: (-(exact[source] - target[source]), source))
    for source in remainder_order[:count - sum(target.values())]:
        target[source] += 1
    selected = []
    for source in SPLITS:
        seed_bytes = hashlib.blake2b(f'{seed}:{source}:short-branch'.encode(), digest_size=8).digest()
        generator = np.random.default_rng(int.from_bytes(seed_bytes, 'little'))
        candidates = by_source[source]
        selected.extend(candidates[index] for index in generator.permutation(len(candidates))[:target[source]])
    return sorted(selected), target


def direct_calibration_selection(probe, seed=2718, count=32):
    buckets = {}
    metadata = probe.select_columns(['sample_id', 'source'])
    for index, row in enumerate(metadata):
        key = hashlib.blake2b(f'{seed}:{row["sample_id"]}'.encode(), digest_size=16).digest()
        buckets.setdefault(row['source'], []).append((key, index, row['sample_id']))
    for values in buckets.values():
        values.sort()
    records = []
    for offset in range(count):
        for source in sorted(buckets):
            if offset < len(buckets[source]) and len(records) < count:
                _, index, sample_id = buckets[source][offset]
                records.append({'order': len(records), 'dataset_index': index,
                                'sample_id': sample_id, 'source': source})
    return records


def compare_all_fields(parent, child, selected_ids):
    require(len(child) == len(selected_ids), 'Child selected cardinality differs')
    require(child.column_names == parent.column_names, 'Child column order/coverage differs')
    selected = parent.select(selected_ids)
    digest = hashlib.sha256()
    rows = 0
    differences = []
    for expected, actual in zip(selected.iter(batch_size=512), child.iter(batch_size=512), strict=True):
        require(len(expected['sample_id']) == len(actual['sample_id']), 'Batch cardinality differs')
        for index, sample_id in enumerate(actual['sample_id']):
            changed = [column for column in child.column_names if expected[column][index] != actual[column][index]]
            if changed:
                differences.append({'child_row': rows, 'sample_id': sample_id, 'fields': changed})
            digest.update(canonical({column: actual[column][index] for column in child.column_names}))
            rows += 1
    return {'rows': rows, 'columns': child.column_names, 'all_field_values_sha256': digest.hexdigest(),
            'differing_rows': differences}


def audit(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This input audit must hide CUDA')
    require(not output.exists(), 'Preserve prior input audit attempts')
    os.nice(10)
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = file_identity(path)
        if expected is not None:
            require(actual == expected if isinstance(expected, dict) else actual['sha256'] == expected,
                    f'Input identity differs: {path}')
        bindings[str(path)] = actual
        return actual

    for relative, digest in PINNED.items():
        bind((PRIMARY if relative.startswith('reports/') else STORY) / relative, digest)
    design = read_json(STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json')
    branch_design = read_json(STORY / 'configs/short_branch_protocol.json')
    common = read_json(STORY / 'configs/common_state_probe.json')
    amendment = read_json(STORY / 'configs/dense_primary_v3_data_amendment.json')
    revised_binding = amendment['datasets']['training']
    for record in revised_binding['files']:
        relative = Path(record['path']).relative_to(revised_binding['root'])
        bind(REVISED / relative, {key: record[key] for key in ('bytes', 'sha256')})
    revised_manifest = read_json(REVISED / 'manifest.json')
    bind(BRANCH / 'manifest.json', design['branch_data']['manifest_sha256'])
    branch_manifest = read_json(BRANCH / 'manifest.json')
    bind(BRANCH / 'rows.jsonl', design['branch_data']['row_ledger_sha256'])
    for row in branch_manifest['dataset_files']:
        bind(BRANCH / row['path'], {key: row[key] for key in ('bytes', 'sha256')})
    bind(PROBE / 'manifest.json', common['probe_manifest_sha256'])
    for path in sorted(PROBE.rglob('*')):
        if path.is_file():
            bind(path)
    bind(__file__)
    parent = Dataset.load_from_disk(str(REVISED / 'dataset'))
    branch = Dataset.load_from_disk(str(BRANCH / 'dataset'))
    probe = Dataset.load_from_disk(str(PROBE / 'dataset'))
    require(len(parent) == 500000 and len(branch) == 50000 and len(probe) == 1024,
            'Actual Dataset coverage differs')
    require(parent._fingerprint == revised_manifest['dataset_fingerprint'] and
            branch._fingerprint == branch_manifest['dataset_fingerprint'], 'Dataset fingerprint differs')
    calibration = direct_calibration_selection(probe)
    selection_digest = hashlib.sha256(b''.join(
        f'{row["order"]}\t{row["dataset_index"]}\t{row["sample_id"]}\t{row["source"]}\n'.encode()
        for row in calibration)).hexdigest()
    calibration_ids = [row['sample_id'] for row in calibration]
    calibration_ids_digest = hashlib.sha256(''.join(f'{value}\n' for value in calibration_ids).encode()).hexdigest()
    require(selection_digest == common['selection']['expected_selection_sha256'] and
            calibration_ids_digest == common['selection']['expected_sample_ids_sha256'] and
            dict(Counter(row['source'] for row in calibration)) == common['selection']['expected_source_counts'],
            'Original fixed calibration selection differs')
    branch_ledger = [json.loads(line) for line in (BRANCH / 'rows.jsonl').open()]
    branch_ids = [row['sample_id'] for row in branch_ledger]
    retained = set(branch_ids) | set(calibration_ids)
    parent_selected_ledger = {}
    by_source = {source: [] for source in SPLITS}
    parent_rows = 0
    with (REVISED / 'rows.jsonl').open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            require(type(row['sample_id']) is int and row['sample_id'] == index and row['source'] in SPLITS,
                    'Parent full ledger ordering/source differs')
            by_source[row['source']].append(index)
            if index in retained:
                parent_selected_ledger[index] = row
            parent_rows += 1
    require(parent_rows == 500000 and {s: len(v) for s, v in by_source.items()} == revised_manifest['quotas'],
            'Parent full ledger coverage/quotas differ')
    reconstructed_ids, quotas = direct_branch_selection(by_source, revised_manifest['quotas'])
    ids_digest = hashlib.sha256(''.join(f'{value}\n' for value in branch_ids).encode()).hexdigest()
    require(branch_ids == reconstructed_ids and len(set(branch_ids)) == 50000 and
            ids_digest == design['branch_data']['selected_sample_ids_sha256'] and
            quotas == branch_manifest['quotas'], 'Original fixed branch selection differs')
    ledger_differences = [row['sample_id'] for row in branch_ledger if row != parent_selected_ledger[row['sample_id']]]
    branch_comparison = compare_all_fields(parent, branch, branch_ids)
    calibration_comparison = compare_all_fields(parent, probe.select([row['dataset_index'] for row in calibration]), calibration_ids)
    result = {
        'scope': 'actual-fixed-factorial-data-relation-to-revised-primary',
        'scientific_admission': False, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': bindings, 'parent_ledger_rows_checked': parent_rows,
        'branch': {**branch_comparison, 'selection_seed': branch_design['subset']['selection_seed'],
                   'source_quotas': quotas, 'selected_sample_ids_sha256': ids_digest,
                   'ledger_differing_sample_ids': ledger_differences,
                   'revised_positions_in_branch': sorted(set(revised_manifest['replaced_sample_ids']) & set(branch_ids))},
        'calibration': {**calibration_comparison, 'selection': calibration,
                        'selection_sha256': selection_digest, 'sample_ids_sha256': calibration_ids_digest,
                        'revised_positions_in_calibration': sorted(set(revised_manifest['replaced_sample_ids']) & set(calibration_ids)),
                        'gradient_groups': [calibration[index:index + 4] for index in range(0, 32, 4)]},
        'all_selected_field_values_equal_to_revised_primary': not (
            ledger_differences or branch_comparison['differing_rows'] or calibration_comparison['differing_rows']),
        'datasets_mutated': False, 'historical_gradients_reused': False, 'gpu_work_started': False,
        'boundary': 'Actual input and full-group equality only, not numerical calibration, source-checkpoint readiness, formal branch or scientific completion.',
    }
    for path, expected in bindings.items():
        require(file_identity(path) == expected, 'Input changed during the read-only data audit')
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    require(result['all_selected_field_values_equal_to_revised_primary'],
            'Fixed branch/calibration differs from revised primary; result preserved, no replacements selected')
    print(json.dumps({'output': str(output), **file_identity(output), 'bound_files': len(bindings),
                      'branch_rows': branch_comparison['rows'], 'calibration_rows': calibration_comparison['rows'],
                      'columns': len(branch_comparison['columns']), 'all_fields_equal': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output.resolve())
