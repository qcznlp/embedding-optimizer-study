"""Host-local preservation audit for this preparation milestone; no model work."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def identity(path):
    assert path.is_file() and not path.is_symlink(), str(path)
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    root = Path(__file__).parents[1]
    story = Path('/root/embedding-optimizer-story-refactor')
    primary = Path('/root/embedding-optimizer-primary-v3')
    experiment = Path('/root/embedding-optimizer-v3-experiment')
    output = root / 'verification.json'
    assert not output.exists(), 'Preserve prior verification'
    protected = {
        story / 'AGENTS.md': '2ea0ac433747a4406823013f36701f49fb8cd806a5c01bdbad73c88ad93b6946',
        primary / 'source-assembly.json': 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8',
        experiment / 'launch/evaluation-handoff/dispatch.py': '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427',
        experiment / 'launch/evaluation-handoff/authorization.json': '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c',
        experiment / 'launch/functional-dimensions/dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
        experiment / 'launch/functional-dimensions/functional.py': 'c55d3fa101fa681070509a95d987023b7ed442d89ce385a8885f940ea3cc05b8',
        experiment / 'launch/functional-dimensions/inputs.json': 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067',
        experiment / 'launch/functional-dimensions/authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336',
        experiment / 'launch/functional-dimensions/run/failed.json': 'c84f93bc431612be2b885d7a467828d08a18d5f259829c4b925a4f23ea7def2e',
        story / 'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
        story / 'paper/build/main.pdf': '94f47113cbfc0e5651bb0d12ef5d8aa16ae5aceb35991ffd5d96626d59de8a3c',
    }
    checked = {}
    for path, digest in protected.items():
        checked[str(path)] = identity(path)
        assert checked[str(path)]['sha256'] == digest, str(path)
    assembly = json.loads((primary / 'source-assembly.json').read_text())
    assembly_count = 0
    for base in (primary, experiment / 'launch/source-snapshot'):
        for relative, entry in assembly['files'].items():
            got = identity(base / relative)
            assert all(got[key] == entry['identity'][key] for key in ('bytes', 'sha256'))
            assembly_count += 1
    assert assembly_count == 112
    copied_before = identity(root / 'before/CURRENT_EXPERIMENT.md')
    assert copied_before['sha256'] == '4aeeff6f2c004dd4aa5621be05b58d0e970b67cdb4448fe66a4d2bf5f621aa51'
    checks = {}
    for name, tests, failures in [('first-attempt/adapter-tests.xml', 80, 54),
                                  ('actual/adapter-policy-tests.xml', 87, 0),
                                  ('actual/adapter-final-tests.xml', 97, 0),
                                  ('actual/combined-final-tests.xml', 124, 0)]:
        suites = ET.parse(root / name).getroot().findall('testsuite')
        counts = {key: sum(int(s.attrib[key]) for s in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        assert counts == {'tests': tests, 'failures': failures, 'errors': 0, 'skipped': 0}
        checks[name] = counts
    first = json.loads((root / 'actual/available-inputs.json').read_text())
    second = json.loads((root / 'actual/relocated-available-inputs.json').read_text())
    compared = 0
    for key in first:
        if key == 'observed_at_utc':
            continue
        if key == 'frozen_sources':
            for name, item in first[key].items():
                assert {k: item[k] for k in ('bytes', 'sha256')} == {
                    k: second[key][name][k] for k in ('bytes', 'sha256')}
        else:
            assert first[key] == second[key], key
        compared += 1
    assert first['actual_complete_checkpoints_checked'] == 54
    assert first['actual_task_scores_checked'] == 756 and first['incomplete_grid_refused'] is True
    assert first['full_grid_summary_executed'] is False
    payloads, links = {}, []
    secret = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})')
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), str(path)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        assert not secret.search(raw), 'Credential-like string in new archive'
        payloads[path.relative_to(root).as_posix()] = identity(path)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', raw.decode()):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                local = target.split('#', 1)[0]
                assert (path.parent / local).exists(), (path, local)
                links.append({'document': str(path), 'target': local})
    report = {'scope': 'complete_trajectory_candidate_host_local_preservation',
              'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'payloads': payloads,
              'protected_inputs': checked, 'original_source_assembly_files_unchanged': assembly_count,
              'test_xml_counts': checks, 'available_input_replay_fields_compared': compared,
              'markdown_local_links_checked': len(links), 'credential_findings': 0,
              'current_handoff': identity(story / 'CURRENT_EXPERIMENT.md'),
              'original_handoff_preserved': copied_before,
              'actual_complete_grid_executed': False, 'model_or_scheduler_started': False,
              'source_or_remote_release': False, 'scientific_completion': False}
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'verification': str(output), 'identity': identity(output),
                      'sealed_payloads': len(payloads), 'source_files_unchanged': assembly_count,
                      'protected_inputs_unchanged': len(checked), 'test_cases_passed': 124,
                      'first_failed_cases_preserved': 54, 'replay_fields_compared': compared,
                      'markdown_local_links_checked': len(links), 'scientific_completion': False},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
