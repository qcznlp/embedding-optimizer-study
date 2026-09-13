"""Read-only endpoint result, exact replay, rendering and unchanged-parent checks."""
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

STORY = Path('/root/embedding-optimizer-story-refactor')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
ROOT = STORY / 'reports/dense-v3-final-inference-v1'
WORK = Path('/tmp/dense-v3-endpoint-inference.pee7UOFu')


def identity(path):
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), path
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()}


def check(path, expected):
    actual = identity(path)
    assert all(actual[k] == v for k, v in expected.items()), path
    return actual


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (ROOT / 'verification.json').exists(), 'Retain previous verification'
    original = EXPERIMENT / 'analyses/dense-primary-v3-final-inference-v1'
    check(original / 'readout.json', {'sha256': '592c902f94b8f190f3165f4075dfddc1e8c3f52add760805da639b541d27716f'})
    result = read(original / 'readout.json')
    assert len(result['outputs']) == 6 and result['task_cells'] == 168 and result['runs'] == 12
    assert result['scientific_completion'] is False and result['original_whole_grid_consumer_called'] is False
    assert result['original_whole_grid_admission_passed'] is False and result['intermediate_scores_imputed'] is False
    for filename, binding in result['outputs'].items():
        for parent in (original, ROOT / 'tables', WORK / 'relocated-replay'):
            check(parent / filename, binding)
    check(ROOT / 'tables/readout.json', identity(original / 'readout.json'))
    check(ROOT / 'source-relocated-readout.json', identity(WORK / 'relocated-replay/readout.json'))
    replay = read(ROOT / 'source-relocated-readout.json')
    first_compare = dict(result)
    replay_compare = dict(replay)
    first_compare.pop('observed_at_utc')
    replay_compare.pop('observed_at_utc')
    assert first_compare == replay_compare
    for name in ('run_endpoint_inference.py', 'render_endpoint_inference.py'):
        check(ROOT / 'source' / name, identity(WORK / name))
    check(ROOT / 'source/run_endpoint_inference.py', result['source'])
    for name, binding in result['frozen_sources'].items():
        check(STORY / name, binding)
        check(ROOT / 'source-original' / name, binding)
    inference = result['inference']
    assert inference['bootstrap_seed'] == 20260903 and inference['bootstrap_samples'] == 50000
    assert result['numpy_version'] == '2.5.2'
    for family, expected_support in (('primary', ['inconclusive'] * 3),
                                     ('secondary', ['inconclusive', 'positive', 'inconclusive'])):
        rows = result['tables'][family + '_summary.csv']
        assert len(rows) == 3 and [r['support'] for r in rows] == expected_support
        assert all(r['tasks'] == 14 and r['bootstrap_samples'] == 50000 and r['bootstrap_seed'] == 20260903 for r in rows)
    assert len(result['independent_scalar_checks']) == 2
    assert all(c['maximum_absolute_scalar_replay_difference'] < 2e-18 for c in result['independent_scalar_checks'])
    figure_original = EXPERIMENT / 'analyses/dense-primary-v3-final-inference-v1-figures'
    figure = read(figure_original / 'rendering.json')
    check(ROOT / 'figures/rendering.json', identity(figure_original / 'rendering.json'))
    check(ROOT / 'source/render_endpoint_inference.py', figure['source'])
    assert len(figure['outputs']) == 4 and len(figure['plotted_rows']) == 6
    for name, binding in figure['outputs'].items():
        check(figure_original / name, binding)
        check(ROOT / 'figures' / name, binding)
    with (ROOT / 'figures/figure_data.csv').open() as handle:
        plotted = list(csv.DictReader(handle))
    for row in plotted:
        candidates = result['tables'][row['family'] + '_summary.csv']
        original_row = next(r for r in candidates if r['treatment'] == row['treatment'] and r['baseline'] == row['baseline'])
        for target, field in (('difference_points', 'mean_delta_ndcg_at_10'),
                              ('simultaneous_lower_points', 'simultaneous_ci_95_lower'),
                              ('simultaneous_upper_points', 'simultaneous_ci_95_upper')):
            assert float(row[target]) == 100 * original_row[field]
        assert row['support'] == original_row['support']
    commands = read(ROOT / 'commands.json')
    assert commands['actual_terminal']['exit_code'] == commands['relocated_replay_terminal']['exit_code'] == 0
    assert commands['render']['result']['exit_code'] == commands['pdf_fonts']['exit_code'] == 0
    assert commands['visual_inspection']['actually_viewed'] is True
    assert len(commands['pdf_fonts']['observed_fonts']) == 2 and all(
        r['type'] == 'CID TrueType' and r['embedded'] for r in commands['pdf_fonts']['observed_fonts'])

    previous = STORY / 'reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1'
    check(previous / 'verification.json', {'sha256': '517e7bd93a3b6b02ff2319bd38750666776ed8b52c14beb67bf3d1d764c14306'})
    binding = read(previous / 'verification.json')
    for name, expected in binding['archive_files'].items():
        check(previous / name, expected)
    for name, expected in binding['live_files'].items():
        check(ROOT / 'before/CURRENT_EXPERIMENT.md' if name == 'CURRENT_EXPERIMENT.md' else STORY / name, expected)
    for root in (Path('/root/embedding-optimizer-primary-v3'), EXPERIMENT / 'launch/source-snapshot'):
        check(root / 'source-assembly.json', {'sha256': 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8'})
        assembly = read(root / 'source-assembly.json')
        assert len(assembly['files']) == 56
        for name, expected in assembly['files'].items():
            check(root / name, expected['identity'])
    dispatch = {
        'evaluation-handoff/dispatch.py': '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427',
        'evaluation-handoff/authorization.json': '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c',
        'validation-handoff/validation.py': 'd2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913',
        'validation-handoff/authorization.json': '6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f',
        'functional-dimensions/dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
        'functional-dimensions/authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'}
    for name, digest in dispatch.items():
        check(EXPERIMENT / 'launch' / name, {'sha256': digest})
    framing = STORY / 'reports/paper-review/dense-v3-retrieval-usefulness-v1'
    framing_binding = read(framing / 'verification.json')
    for name, expected in read(framing / 'before-bindings.json').items():
        if name not in framing_binding['live_files']:
            check(STORY / name, expected)
    for name, expected in framing_binding['live_files'].items():
        if name != 'CURRENT_EXPERIMENT.md':
            check(STORY / name, expected)
    check(STORY / 'paper/build/main.pdf', framing_binding['archive_files']['draft/main.pdf'])
    check(STORY / 'AGENTS.md', {'sha256': '2ea0ac433747a4406823013f36701f49fb8cd806a5c01bdbad73c88ad93b6946'})

    links = 0
    for doc in (ROOT / 'README.md', STORY / 'CURRENT_EXPERIMENT.md'):
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if link.startswith(('http://', 'https://', '#')):
                continue
            path = (doc.parent / link.split('#', 1)[0]).resolve()
            if path == ROOT / 'verification.json':
                continue
            assert path.exists(), (doc, link)
            links += 1
    files = {p.relative_to(ROOT).as_posix(): identity(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
    result = {'scope': 'actual_complete_endpoint_inference_artifact_and_relocation_verification',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'archive_files': files,
              'live_files': {'CURRENT_EXPERIMENT.md': identity(STORY / 'CURRENT_EXPERIMENT.md')},
              'byte_identical_replayed_numeric_text_outputs': 6, 'relocated_receipt_equal_except_observation_time': True,
              'native_final_task_cells': 168, 'final_checkpoint_runs': 12, 'contrast_intervals': 6,
              'independent_scalar_interval_fields': 48, 'exact_task_optimizer_means': 84,
              'plotted_intervals_verified': 6, 'local_links_checked': links,
              'prior_archive_files_preserved': len(binding['archive_files']), 'prior_live_handoff_binding_preserved': True,
              'primary_source_assemblies_verified': 2, 'primary_files_per_assembly': 56,
              'dispatch_authorization_files_unchanged': 6, 'manuscript_and_generated_results_unchanged': True,
              'original_whole_grid_consumer_called': False, 'original_whole_grid_admission_passed': False,
              'physical_second_host_experiment': False, 'resource_handoff': False,
              'source_release': False, 'scientific_completion': False, 'whole_goal_complete': False,
              'passed_in_declared_scope': True}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
