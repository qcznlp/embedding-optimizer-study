"""Independent standard-library byte and table/figure checks; no new inference."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MANIFEST_SHA = 'cddee2dcfb68c25e42931c82f129c0f57590434cecf2b48bbc106da0dfc38c2e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def data(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Recovered inputs must be ordinary, nonsymlinked files')
    return path.read_bytes()


def compare_rows(path, expected):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        header, actual = reader.fieldnames, list(reader)
    require(len(header) == len(set(header)) and len(actual) == len(expected), 'CSV shape differs')
    numeric = 0
    for row, reference in zip(actual, expected):
        require(set(row) == set(reference), 'CSV columns differ')
        for key, value in reference.items():
            if type(value) in (int, float):
                require(float(row[key]) == value, 'Recovered CSV number differs from native record')
                numeric += 1
            else:
                require(row[key] == value, 'Recovered CSV label differs from native record')
    return len(actual), numeric


def verify(root):
    raw = data(root / 'artifact_manifest.json')
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'External manifest anchor differs')
    manifest = json.loads(raw)
    files = manifest['files']
    require(len(files) == 15 and manifest['contrasts'] == 6 and manifest['families'] == 2
            and manifest['new_inference'] is False and manifest['source_code_included'] is False
            and manifest['scientific_completion'] is False, 'Wrong snapshot scope')
    expected_names = set(files) | {'artifact_manifest.json'}
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == expected_names, 'Recovered file population differs')
    count_bytes = len(raw)
    for name, identity in files.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name,
                'Unsafe recovered manifest path')
        payload = data(root / name)
        require(len(payload) == identity['bytes']
                and hashlib.sha256(payload).hexdigest() == identity['sha256']
                and hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
                == identity['git_blob_sha1'], 'Recovered bytes differ')
        count_bytes += len(payload)
    native = json.loads(data(root / 'tables/readout.json'))
    moved = json.loads(data(root / 'provenance/source-relocated-readout.json'))
    require({k: v for k, v in native.items() if k != 'observed_at_utc'}
            == {k: v for k, v in moved.items() if k != 'observed_at_utc'},
            'Original relocated receipt differs beyond its declared observation time')
    require(native['runs'] == 12 and native['task_cells'] == 168
            and native['original_whole_grid_admission_passed'] is False
            and native['source_release'] is False, 'Wrong native scope')
    table_rows = numeric_fields = 0
    for name, rows in native['tables'].items():
        count, numbers = compare_rows(root / 'tables' / name, rows)
        table_rows += count
        numeric_fields += numbers
    render = json.loads(data(root / 'figures/rendering.json'))
    plotted = render['plotted_rows']
    require(len(plotted) == 6, 'Incomplete figure')
    figure_rows, figure_numbers = compare_rows(root / 'figures/figure_data.csv', plotted)
    for row in plotted:
        stats = next(value for value in native['tables'][f'{row["family"]}_summary.csv']
                     if value['treatment'] == row['treatment'] and value['baseline'] == row['baseline'])
        for source, target in [('mean_delta_ndcg_at_10', 'difference_points'),
                               ('simultaneous_ci_95_lower', 'simultaneous_lower_points'),
                               ('simultaneous_ci_95_upper', 'simultaneous_upper_points')]:
            require(stats[source] * 100 == row[target], 'Figure unit conversion differs')
        lower, upper = stats['simultaneous_ci_95_lower'], stats['simultaneous_ci_95_upper']
        support = 'positive' if lower > 0 else 'negative' if upper < 0 else 'inconclusive'
        require(stats['support'] == row['support'] == support, 'Reported support differs')
    return {'scope': 'independent-recovered-byte-and-numeric-table-figure-mapping',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'manifest_sha256': MANIFEST_SHA, 'files': len(expected_names), 'bytes': count_bytes,
            'native_csv_rows': table_rows, 'native_csv_numeric_fields': numeric_fields,
            'figure_rows': figure_rows, 'figure_numeric_fields': figure_numbers,
            'interval_figure_links': 6, 'relocation_receipts_equal_except_time': True,
            'all_checks_passed': True, 'new_bootstrap_or_model_execution': False,
            'physical_second_host_experiment': False, 'scientific_completion': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.root.is_absolute() and args.output.is_absolute(), 'Use explicit absolute paths')
    result = verify(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
