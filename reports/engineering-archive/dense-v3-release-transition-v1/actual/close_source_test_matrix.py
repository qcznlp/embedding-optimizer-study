"""Join separately executed, explicit source-version test roles without suppressing failures."""
import hashlib
import json
from pathlib import Path

from scripts import test_source_roles as runner

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
first = work / 'full-source-tests-v1'
current = work / 'current-role-final'
old_input = json.loads((first / 'inputs.json').read_text())
new_input = json.loads((current / 'inputs.json').read_text())
roles = runner.load_roles(root)
results = {}
identifiers = []
for name, role in roles.items():
    directory = current if name == 'current' else first
    inputs = new_input if name == 'current' else old_input
    report = directory / (name + '-result.json')
    result = json.loads(report.read_text())
    assert result['complete'] is True and result['exit_code'] == 0
    assert result['failure'] == result['error'] == result['skipped'] == 0
    assert inputs['roles'][name]['tests'] == role['tests']
    assert inputs['roles'][name]['changes'] == role['changes']
    actual_root = Path(inputs['roles'][name]['root'])
    # Every executed test is still byte-identical to the final checkout. Historical
    # copies retain their explicit original dependencies, rather than adopting the
    # current training version. No failing result is changed or called successful.
    for label in role['tests']:
        expected = inputs['files'][label]['sha256']
        assert runner.sha(root / label) == expected
        assert runner.sha(actual_root / label) == expected
    for label, replacement in role['changes'].items():
        assert runner.sha(actual_root / label) == replacement['sha256']
        assert runner.sha(root / replacement['origin']) == replacement['sha256']
    for label, binding in inputs['files'].items():
        # Executable study code and frozen configuration are common between the
        # separately launched roles apart from their declared original overlays.
        if label.startswith(('src/', 'scripts/', 'configs/')) and label.endswith(('.py', '.json', '.yaml', '.yml')):
            assert runner.sha(root / label) == binding['sha256']
    xml = runner.read_results(directory / (name + '.xml'))
    assert xml['identifiers'] == result['identifiers']
    identifiers.extend(result['identifiers'])
    results[name] = {'cases': result['cases'], 'test_modules': len(role['tests']),
                     'result_path': str(report), 'result_sha256': runner.sha(report),
                     'input_inventory_sha256': runner.sha(directory / 'inputs.json'),
                     'exit_code': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
assert len(identifiers) == len(set(identifiers))
assert len(identifiers) == 3800
deltas = sorted(name for name, binding in new_input['files'].items() if old_input['files'].get(name) != binding)
result = {'complete': True, 'cases': len(identifiers), 'test_modules': sum(len(role['tests']) for role in roles.values()),
          'roles': results, 'current_role_rerun': True, 'original_failed_combined_run_preserved': True,
          'historical_roles_restarted': False, 'changed_files_between_role_launches': deltas,
          'role_file_sha256': runner.sha(root / runner.ROLE_FILE),
          'scope': 'Complete per-source-version regression matrix; not a passing bare mixed-source pytest invocation or fresh training admission',
          'fresh_training_worker_admission': False, 'remote_publication': False}
with (work / 'complete-source-test-matrix.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
