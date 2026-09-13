"""Retain genuine local replay evidence and the preceding hosted failure.

Run only after the exact owned handles are observed terminal. No scientific
source, comparison, output, system package or existing archive is changed.
"""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from embed_optim.distribution_audit import _credential_findings

ROOT = Path('/root/embedding-optimizer-story-refactor')
WORK = Path(__file__).resolve().parent
MANUAL = Path('/tmp/ci-python-runtime.rXB7OKBE')
SDE = Path('/tmp/ci-sde-runtime.lJfWrfwu')
OUT = ROOT / 'reports/engineering-archive/dense-v3-ci-subprocess-runtime-v1'


def binding(path):
    return {'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def replay(root):
    complete = read(root / 'complete.json')
    assert complete['complete'] is True
    assert complete['numerical_and_reviewed_document_complete'] is True
    assert complete['original_contracts_modified'] is False
    assert complete['repository_publication_complete'] is False
    assert complete['document_snapshot_sha256'] == '6ef28bcf1f2cd873bcb0a663c2baea57ec6dc98a4e441d0850608a61f46ae6b9'
    assert complete['input_manifest_sha256'] == '746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7'
    assert complete['numerical_completion'] == binding(root / 'numerical/complete.json')
    assert complete['numerical_exit'] == read(root / 'numerical-exit.json')
    assert complete['numerical_exit']['exit_code'] == 0
    numerical = read(root / 'numerical/complete.json')
    assert numerical['all_original_scientific_rules_unchanged'] is True
    assert numerical['complete_paper_source_and_extracted_text_exact'] is True
    assert numerical['factorial_counts'] == {
        'beir_seed_task_scores': 168, 'estimand_seed_task_contrasts': 126,
        'estimand_summary': 3, 'factorial_cell_summary': 4,
        'probe_checkpoint_metrics': 60, 'probe_task_metrics': 840}
    assert numerical['primary_completion'] == binding(root / 'numerical/primary/reconstructed/complete.json')
    primary = read(root / 'numerical/primary/reconstructed/complete.json')
    for key in ('all_84_weight_and_24_functional_controls_reconstructed',
                'all_nine_functional_tables_and_decisions_reconstructed',
                'all_ten_primary_outcome_tables_reconstructed',
                'original_nine_and_exact_five_weight_predictors_reconstructed'):
        assert primary[key] is True
    assert len(primary['all_17_original_publication_outputs_byte_exact']) == 17
    for name, expected in primary['all_17_original_publication_outputs_byte_exact'].items():
        assert binding(root / 'numerical/primary/reconstructed' / name) == expected
    assert len(complete['shared_inputs']) == 8
    for name, expected in complete['shared_inputs'].items():
        assert binding(ROOT / 'paper/current' / name) == expected
        assert binding(root / 'reviewed/paper' / name) == expected
    reviewed = complete['reviewed_document']
    assert reviewed['document_reproduction_complete'] is True
    assert reviewed['pdf_pages'] == 13 and reviewed['main_end_page'] == 8
    assert reviewed['abstract_words'] == 158 and reviewed['font_count'] == 25
    assert reviewed['document_receipt'] == binding(root / 'reviewed/document.json')
    assert complete['pdf'] == binding(root / 'reviewed/paper/build/main.pdf')
    for name in ('numerical/primary-exited.json', 'numerical/document/compiler-exited.json',
                 'reviewed/compiler-exited.json'):
        assert read(root / name)['exit_code'] == 0
    assert not (root / 'failed.json').exists()
    return {'receipt': binding(root / 'complete.json'), 'shared_inputs': 8,
            'pdf_pages': 13, 'original_comparisons_unchanged': True}


def main():
    # Written from actual terminal tool observations, never from artifact existence.
    terminals = read(WORK / 'observed-terminal-exits.json')
    for key in ('manual_complete_replay', 'repository_builder', 'repository_fd_manual_runtime',
                'repository_fd_built_runtime', 'built_runtime_functional',
                'built_runtime_complete_replay', 'complete_source_role_tests',
                'final_distribution_build', 'final_distribution_audit'):
        assert terminals[key]['exit_code'] == 0, key
    manual = replay(MANUAL / 'complete-paper')
    built = replay(WORK / 'complete-paper')
    runtime = read(WORK / 'runtime/runtime.json')
    assert runtime['complete'] is True and runtime['source_modified'] is False
    assert runtime['package_installation_performed'] is False
    assert runtime['system_environment_modified'] is False
    assert runtime['configure_environment']['ac_cv_func_close_range'] == 'no'
    assert runtime['python_sha256'] == binding(Path(runtime['python']))['sha256']
    assert runtime['source_sha256'] == '56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1'
    assert binding(WORK / 'runtime/Python-3.12.3.tar.xz')['sha256'] == runtime['source_sha256']
    commands = read(WORK / 'runtime/commands.json')
    assert [row['name'] for row in commands] == ['configure', 'build', 'install', 'runtime']
    assert all(row['exit_code'] == 0 for row in commands)
    for name in ('fd-entry.log', 'fd-built-runtime.log'):
        result = json.loads((WORK / name).read_text().splitlines()[-1])
        assert result['complete'] is True
        assert [row['keep_one_explicit_fd'] for row in result['cases']] == [False, True]
        assert all(row['exit_code'] == 0 for row in result['cases'])
    functional = read(WORK / 'functional.json')
    assert functional['functional_tables_exact'] is functional['decisions_exact'] is True
    assert functional['differences'] == [] and functional['acceptance_override'] is False
    tests = read(WORK / 'source-role-tests/summary.json')
    assert tests['complete'] is True and tests['cases'] == 3800 and tests['test_modules'] == 211
    assert {row['role']: row['cases'] for row in tests['roles']} == {
        'current': 2873, 'original-analysis': 733, 'original-factorial': 194}
    assert all(row['exit_code'] == row['failure'] == row['error'] == row['skipped'] == 0
               for row in tests['roles'])
    assert tests['role_file_sha256'] == binding(ROOT / 'configs/source_test_roles.json')['sha256']
    previous = SDE / 'ci-b0fa533f-receipts.zip'
    assert binding(previous) == {'bytes': 15742, 'sha256': '9ed168a34b37b463f7dc45006c90136639c9c62ea7ce5b5ae87d52edb6161842'}
    with zipfile.ZipFile(previous) as archive:
        remote_probe = json.loads(archive.read('cpu-replay-diagnostics/functional.json'))
        assert remote_probe['functional_tables_exact'] is remote_probe['decisions_exact'] is True
        assert remote_probe['differences'] == []
        assert json.loads(archive.read('complete-paper/numerical-exit.json'))['exit_code'] == 255
        # This launcher failure is emitted by the parent before child logging starts.
        assert b'PreparePindForFollowExecve' in (SDE / 'ci-b0fa533f-job.log').read_bytes()
        for info in archive.infolist():
            if not info.is_dir():
                assert not _credential_findings('previous-hosted-zip', info.filename, archive.read(info))
    files = {Path(__file__): 'archive_local_verification.py', previous: previous.name,
             SDE / 'ci-b0fa533f-job.log': 'ci-b0fa533f-job.log',
             WORK / 'observed-terminal-exits.json': 'observed-terminal-exits.json'}
    for name in ('archive-local-verification-attempt1.py', 'archive-verification.log'):
        files[WORK / name] = 'collector-failed-attempt/' + name
    for name in ('full-forced-skx.log', 'follow-child-minimal.log', 'follow-subprocess-minimal.log',
                 'follow-exec-replace.log', 'follow-fds-open.log', 'follow-no-vfork.log',
                 'follow-pind-enabled.log', 'follow-pind-enabled.txt', 'follow-pind-debug.log',
                 'follow-pind-debug.txt', 'follow-pind-grouped.log', 'follow-pind-grouped.txt',
                 'follow-high-fd.log', 'paper-pin-log.txt'):
        files[SDE / name] = 'preceding-controls/' + name
    for name in ('failed.json', 'numerical-exit.json', 'numerical.log'):
        files[SDE / 'complete-paper-forced' / name] = 'preceding-full-failure/' + name
    for source in MANUAL.glob('*.log'):
        files[source] = 'manual-build/' + source.name
    for name in ('source-verified.json', 'source.spdx.json', 'prepare.py',
                 'check_fd_isolation.py', 'check_numerical_runtime.py'):
        files[MANUAL / name] = 'manual-build/' + name
    for name in ('commands.json', 'runtime.json', 'configure.log', 'build.log', 'install.log', 'runtime.log'):
        files[WORK / 'runtime' / name] = 'repository-build/' + name
    files[WORK / 'runtime/Python-3.12.3/LICENSE'] = 'repository-build/CPython-LICENSE'
    for name in ('build-entry.log', 'fd-entry.log', 'fd-built-runtime.log', 'functional.log',
                 'functional.json', 'full-numerical-paper.log', 'source-path-control.log',
                 'source-role-driver.log', 'focused-tests.log', 'focused-tests.xml'):
        files[WORK / name] = 'repository-entry/' + name
    for source in (WORK / 'source-role-tests').iterdir():
        if source.is_file() and source.suffix in ('.json', '.xml', '.log'):
            files[source] = 'source-role-tests/' + source.name
    for source in (WORK / 'before-handoff').iterdir():
        assert source.is_file()
        files[source] = 'before-handoff/' + source.name
    for name in ('distribution-build.log', 'distribution-audit.log', 'portable-evidence.log',
                 'distribution-final-build.log', 'distribution-final-audit.log',
                 'final-workflow-tests.log', 'final-workflow-tests.xml'):
        files[WORK / name] = 'local-release/' + name
    for name in ('scripts/build_ci_python.py', 'scripts/check_cpu_replay_fds.py',
                 '.github/workflows/ci.yml', 'tests/test_source_roles.py'):
        files[ROOT / name] = 'executed-entry/' + name
    receipts = ('complete.json', 'numerical-exit.json', 'numerical.log', 'numerical/complete.json',
                'numerical/primary-exited.json', 'numerical/primary-replay.log',
                'numerical/primary/reconstructed/complete.json',
                'numerical/factorial/independent_verification.json',
                'numerical/factorial/tables.json', 'numerical/document/compiler-exited.json',
                'numerical/document/document.json', 'reviewed/current-paper.json',
                'reviewed/document-snapshot.json', 'reviewed/document.json',
                'reviewed/compiler-exited.json', 'reviewed/paper/build/main.pdf')
    for root, label in ((MANUAL / 'complete-paper', 'manual-full'),
                        (WORK / 'complete-paper', 'repository-full')):
        for name in receipts:
            files[root / name] = label + '/' + name
    assert OUT.is_dir() and set(p.name for p in OUT.iterdir()) == {'README.md'}
    for source, name in files.items():
        assert source.is_file() and not source.is_symlink(), str(source)
        assert not _credential_findings('local-evidence', name, source.read_bytes()), name
        assert not (OUT / name).exists()
    for source, name in files.items():
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert binding(source) == binding(destination)
    verified = {'scope': 'original-CPython-FD-path-and-complete-local-CPU-reproduction',
                'manual_replay': manual, 'repository_built_runtime_replay': built,
                'source_role_cases': 3800, 'source_role_modules': 211,
                'launcher_prefix': 'sde64 -skx -force_emulate skx --',
                'current_document_pythonpath': [str(ROOT / 'src'), str(ROOT)],
                'original_numerical_child_pythonpath': '',
                'numerical_threads': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                      'MKL_NUM_THREADS': '1'},
                'preceding_hosted_run': 34769598250, 'preceding_hosted_full_replay_passed': False,
                'hosted_replacement_pass_claimed': False, 'published': False,
                'source_or_assertions_modified': False, 'gpu_execution': False,
                'system_packages_or_security_modified': False, 'sde_binary_redistributed': False}
    (OUT / 'verified.json').write_text(json.dumps(verified, indent=2, sort_keys=True) + '\n')
    manifest = {'files': {p.relative_to(OUT).as_posix(): binding(p)
                          for p in sorted(OUT.rglob('*')) if p.is_file()}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files': len(manifest['files']), 'manifest': binding(OUT / 'manifest.json')}))


if __name__ == '__main__':
    main()
