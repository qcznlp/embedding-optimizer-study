"""Read back the complete bounded archive; no new model/gradient execution."""

import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from embed_optim.primary_contract import file_identity, read_json, verify_file

ARCHIVE = Path(__file__).resolve().parent
WORK = Path('/tmp/dense-v3-factorial-calibration.U5gTyB')
STORY = Path('/root/embedding-optimizer-story-refactor')
BLOCK = "**Genuine v3 calibration component checked; GPU execution still pending — 2026-09-10.**\n\nThe new internal `factorial_v3_inputs.py` and `factorial_v3_calibration.py` connect\nthe genuine corrected source/data evidence to fresh gradient-history production,\ncomplete direction-norm reading and all twelve reset continuation cells. The four\nhidden rates are derived from all 88 matrix norms by the unchanged global\nFrobenius rule, not hand-entered or selected using retrieval. Gradient and final\ncalibration receipts require externally supplied content bindings. Neither saved\ncalibration moments nor historical gradients become branch initialization.\n\nBoth actual-input consumer reads passed on their own source versions. The final\n`actual/actual-inputs-third.json` has SHA\n**3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f**.\nIt authenticates 66 assembled files, including all 56 unchanged primary files,\nboth native source checkpoints and the exact 50K / 32-row input evidence.\nThe final **57 focused tests** pass. Actual tiny-tensor replay is checked against\nan independent FP64 Adam reference and eight actual primary Muon updates; complete\ncalibration-to-twelve-cell wiring remains explicitly synthetic. Initial 45- and\n56-case results, original source versions, and the first audit-wrapper schema\nfailure are preserved, not aggregated into extra experiments.\n\nRead `reports/engineering-archive/dense-v3-factorial-calibration-v1/README.md` in\nthe story-refactor tree. All its CPU test/input calls are terminal. Do not repeat\nthem as missing work. No GPU calibration, calibrated primary rate or formal branch\nis claimed, and the component has no scheduler. Its internal GPU functions require\na separately admitted exclusive worker; they do not waive source/runtime/release,\nresource or primary-completion gates. Real GPU save/readback, four-GPU DenseOn\nbranch verification, formal run identity and whole-run consumers remain required.\n\nAt the exact observer's **09:27:30 UTC** snapshot, primary BEIR is **12 / 840**,\nbaseline **14 / 14**, and all eight exact primary workers are live/R with no failure\nreceipt. Existing dispatch and numerical sources are unchanged. The unanswered\none-GPU request still is not authorization for a handoff. Complete primary training\nand all sixty remote backups remain done; no scientific winner is claimed.\n\n"
OPS = [
    (STORY / 'AGENTS.md', 'AGENTS.md'),
    (STORY / 'PROJECT_STATUS.md', 'PROJECT_STATUS.md'),
    (Path('/root/embedding-optimizer-v3-experiment/launch/RUNNING.md'), 'PRIMARY_RUNNING.md'),
    (Path('/root/embedding-optimizer-v3-experiment/launch/view-history-continuation/RUNNING.md'), 'CONTINUATION_RUNNING.md'),
]


def main():
    output = ARCHIVE / 'verification.json'
    if output.exists():
        raise ValueError('Preserve previous archive verification')
    copied = {}
    for original in sorted(WORK.rglob('*')):
        if original.is_file():
            relative = original.relative_to(WORK)
            expected = file_identity(original)
            verify_file(ARCHIVE / 'actual' / relative, expected)
            copied[relative.as_posix()] = expected
    cases = {}
    for name, expected in [('tests-first.xml', 45), ('tests-second.xml', 56), ('tests-third.xml', 57)]:
        rows = ET.parse(ARCHIVE / 'actual' / name).getroot().findall('.//testcase')
        if len(rows) != expected or any(any(r.find(k) is not None for k in ('failure', 'error', 'skipped')) for r in rows):
            raise ValueError('A focused test attempt is incomplete or failed')
        cases[name] = expected
    native = {}
    for version, sha in [('second', 'c75a198d40f245cea98065f942d57a53ff63d2cf9bfb0fa9157ec1f2964ccb00'),
                         ('third', '3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f')]:
        path = ARCHIVE / 'actual' / f'actual-inputs-{version}.json'
        if file_identity(path)['sha256'] != sha:
            raise ValueError('Actual input reader output changed')
        record = read_json(path)
        if (record['genuine_source_states'] != 2 or record['branch_rows'] != 50000
            or record['calibration_rows'] != 32 or record['actual_model_or_gradient_execution'] is not False
            or record['scientific_completion'] is not False or record['primary_files_unchanged'] != 56):
            raise ValueError('Actual input consumer scope differs')
        for relative, expected in record['source_files'].items():
            verify_file(WORK / f'source-{version}' / relative, expected)
            verify_file(ARCHIVE / 'actual' / f'source-{version}' / relative, expected)
        native[version] = {'receipt': file_identity(path), 'assembled_source_files': len(record['source_files'])}
    final = read_json(ARCHIVE / 'actual/actual-inputs-third.json')
    verify_file(STORY / 'reports/engineering-archive/dense-v3-factorial-calibration-v1/audit_actual_inputs.py',
                final['audit_source'])
    primary = read_json(STORY / 'reports/engineering-archive/dense-v3-primary-launch-v1/source-assembly.json')
    for relative, expected in primary['files'].items():
        verify_file(Path('/root/embedding-optimizer-primary-v3') / relative, expected['identity'])
    for relative in ('src/embed_optim/factorial_v3_inputs.py', 'src/embed_optim/factorial_v3_calibration.py'):
        verify_file(STORY / relative, final['source_files'][relative])
    edits = []
    for target, name in OPS:
        before = ARCHIVE / 'ops-before' / name
        current, previous = target.read_text(), before.read_text()
        if current.count(BLOCK) != 1 or current.replace(BLOCK, '', 1) != previous:
            raise ValueError('Operational edit changed more than its one bounded addition')
        edits.append({'path': str(target), 'before': file_identity(before), 'after': file_identity(target)})
    if (WORK / 'actual-inputs-first.json').exists():
        raise ValueError('Original pre-admission failure was relabelled successful')
    files = {str(p.relative_to(ARCHIVE)): file_identity(p) for p in sorted(ARCHIVE.rglob('*')) if p.is_file()}
    result = {'scope': 'bounded-genuine-v3-calibration-component-archive',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'final_focused_cases_passed': 57, 'prior_focused_case_counts_not_additive': cases,
              'actual_input_reads': native, 'original_files_exactly_preserved': len(copied),
              'primary_source_files_unchanged': len(primary['files']),
              'reversible_operational_edits': edits, 'files': files,
              'first_input_wrapper_attempt_exit_code': 1,
              'first_failure_stage': 'new wrapper parent-inventory field lookup before input admission',
              'all_owned_test_and_input_calls_terminal': True,
              'fresh_model_gradient_or_direction_computation_by_this_verifier': False,
              'actual_gpu_calibration_completed': False, 'formal_branches_admitted': 0,
              'scientific_completion': False}
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'verification': str(output), **file_identity(output),
                      'archived_files': len(files), 'original_files_preserved': len(copied),
                      'final_focused_cases': 57, 'gpu_calibration_completed': False}), flush=True)


if __name__ == '__main__':
    main()
