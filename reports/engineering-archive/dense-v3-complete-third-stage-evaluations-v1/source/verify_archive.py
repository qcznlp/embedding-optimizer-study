"""One-shot, CPU-only local archive check; not a model or publication audit."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-complete-third-stage-evaluations-v1'
WORK = Path('/tmp/dense-v3-complete-third-stage-readback.w0XgOLcY')
EXPECTED = {
    'README.md', 'commands.json', 'observations.json',
    'before/CURRENT_EXPERIMENT.md',
    'source/readback.py', 'source/independent_readback.py', 'source/verify_archive.py',
    'actual/complete-checkpoint-readback.json', 'actual/independent-raw-reconstruction.json',
    'actual/archived-source-replay.json', 'actual/native-primary-exits.json',
    'tables/third-stage-checkpoint-scores.csv', 'tables/third-stage-task-scores.csv',
    'tables/third-stage-checkpoint-scores.md',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path):
    path = Path(path)
    require('gpu.py' not in path.parts, 'Protected helper is outside this audit')
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Require a regular bound file')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in keys), 'Input moved')
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, raw


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for archive checks')
    destination = ARCHIVE / 'verification.json'
    require(not destination.exists(), 'Preserve the original verification; do not rerun in place')
    listed = subprocess.run(['rg', '--files', '--hidden', str(ARCHIVE)],
                            check=True, capture_output=True, text=True)
    actual = {str(Path(p).relative_to(ARCHIVE)) for p in listed.stdout.splitlines()}
    require(actual == EXPECTED, 'Unexpected archive inventory')
    payloads, contents = {}, {}
    for relative in sorted(EXPECTED):
        payloads[relative], contents[relative] = snapshot(ARCHIVE / relative)

    copies = {
        'source/readback.py': WORK / 'source/readback.py',
        'source/independent_readback.py': WORK / 'source/independent_readback.py',
        'actual/complete-checkpoint-readback.json': WORK / 'complete-checkpoint-readback.json',
        'actual/independent-raw-reconstruction.json': WORK / 'independent/independent-raw-reconstruction.json',
        'actual/archived-source-replay.json': WORK / 'archive-replay/independent-raw-reconstruction.json',
    }
    for name in ('third-stage-checkpoint-scores.csv', 'third-stage-task-scores.csv',
                 'third-stage-checkpoint-scores.md'):
        copies['tables/' + name] = WORK / 'independent' / name
        require(snapshot(WORK / 'independent' / name)[1]
                == snapshot(WORK / 'archive-replay' / name)[1], 'Replayed table bytes differ')
    for relative, original in copies.items():
        require(contents[relative] == snapshot(original)[1], 'Archive copy differs')
    require(payloads['before/CURRENT_EXPERIMENT.md']['sha256']
            == '589778f95c9f2e06d595a3cb1b1fb7c01b55a77b5084512966fa74d7f63f2992',
            'Prior handoff differs')

    summary = json.loads(contents['actual/independent-raw-reconstruction.json'])
    replay = json.loads(contents['actual/archived-source-replay.json'])
    require(summary['complete_checkpoint_records'] == replay['complete_checkpoint_records'] == 48
            and summary['new_checkpoint_records'] == 6 and summary['new_raw_task_values'] == 84,
            'Wrong complete/new checkpoint coverage')
    require(summary['raw_task_scores_checked'] == summary['native_exit_zero_workers_checked'] == 672
            and summary['raw_score_and_metadata_snapshots_checked'] == 768
            and summary['third_stage_checkpoint_rows'] == 12
            and summary['third_stage_raw_task_values'] == 168
            and summary['previous_forty_two_records_exactly_unchanged'] is True
            and summary['all_original_rational_means_exact'] is True,
            'Wrong independent reconstruction scope')
    for name, original in summary['generated_artifacts'].items():
        current = payloads['tables/' + name]
        require(all(current[k] == original[k] == replay['generated_artifacts'][name][k]
                    for k in ('bytes', 'sha256')), 'Generated artifact identity differs')
    for relative, expected in (
        ('actual/complete-checkpoint-readback.json', summary['native_bundle']),
        ('source/independent_readback.py', summary['source']),
    ):
        require(all(payloads[relative][k] == expected[k] for k in ('bytes', 'sha256')),
                'Recorded native/source identity differs')
    commands = json.loads(contents['commands.json'])
    require(commands['native_readback']['polls'][-1]['exit_code'] == 0
            and commands['independent_readback']['result']['exit_code'] == 0
            and commands['archived_source_replay']['result']['exit_code'] == 0,
            'Missing original terminal success records')
    exits = json.loads(contents['actual/native-primary-exits.json'])
    records = [r for segment in exits['segments'] for r in segment['native_receipts']]
    require(len(records) == len({r['path'] for r in records}) == 46
            and all(r['exit_code'] == 0 for r in records), 'Wrong observed original exit set')
    for record in records:
        bound, raw = snapshot(record['path'])
        require(all(bound[k] == record[k] for k in ('bytes', 'sha256'))
                and json.loads(raw)['exit_code'] == 0, 'Original exit record changed')

    protected = {
        STORY / 'AGENTS.md': '2ea0ac433747a4406823013f36701f49fb8cd806a5c01bdbad73c88ad93b6946',
        PRIMARY / 'AGENTS.md': 'ae76d597a42f8b1a688a4da116de8e8b4ad735411fff1b2b6d792a82a690b1df',
        PRIMARY / 'source-assembly.json': 'e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8',
        STORY / 'paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
        STORY / 'paper/build/main.pdf': '94f47113cbfc0e5651bb0d12ef5d8aa16ae5aceb35991ffd5d96626d59de8a3c',
        EXPERIMENT / 'launch/evaluation-handoff/dispatch.py': '5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427',
        EXPERIMENT / 'launch/evaluation-handoff/authorization.json': '2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c',
        EXPERIMENT / 'launch/functional-dimensions/dispatch.py': '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5',
        EXPERIMENT / 'launch/functional-dimensions/authorization.json': 'd72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336',
        EXPERIMENT / 'launch/functional-dimensions/run/failed.json': 'c84f93bc431612be2b885d7a467828d08a18d5f259829c4b925a4f23ea7def2e',
        EXPERIMENT / 'launch/functional-dimensions/functional.py': 'c55d3fa101fa681070509a95d987023b7ed442d89ce385a8885f940ea3cc05b8',
        STORY / 'reports/engineering-archive/dense-v3-third-stage-pool-b-evaluations-v1/verification.json': '71d559b1e5602ba86239d6ad0bfccd0bdc14dda511a78814c980e8b62ee62ac2',
        STORY / 'reports/engineering-archive/dense-v3-third-stage-pool-b-artifact-backup-v1/verification.json': '3d1cec7ae87c44ce46534048cdfe4296343a3e8ef1b48d6d39b16de7e959f695',
    }
    protected_records = []
    for path, expected_sha in protected.items():
        bound, _ = snapshot(path)
        require(bound['sha256'] == expected_sha, 'Protected original changed')
        protected_records.append(bound)
    prior, _ = snapshot(summary['prior_forty_two_bundle']['path'])
    require(prior == summary['prior_forty_two_bundle'], 'Original 42-state bundle changed')
    assembly = json.loads(snapshot(PRIMARY / 'source-assembly.json')[1])
    source_checks = 0
    for root in (PRIMARY, EXPERIMENT / 'launch/source-snapshot'):
        for relative, record in assembly['files'].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                    'Invalid bound source path')
            bound, _ = snapshot(root / relative)
            require(all(bound[k] == record['identity'][k] for k in ('bytes', 'sha256')),
                    'Original assembled source changed')
            source_checks += 1
    require(source_checks == 112, 'Wrong original source assembly coverage')
    current, current_raw = snapshot(STORY / 'CURRENT_EXPERIMENT.md')
    for marker in (b'672 / 840 task cells', b'[48 / 60]', b'locally archived,',
                   b'not yet added to the earlier HF backups'):
        require(marker in current_raw, 'Current handoff misses a material scope statement')
    secret = re.compile(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,})')
    require(not any(secret.search(raw) for raw in [*contents.values(), current_raw]),
            'Possible credential: preserve locally and stop publication')
    links = 0
    deferred_self_links = 0
    documents = [(ARCHIVE / 'README.md', contents['README.md']),
                 (STORY / 'CURRENT_EXPERIMENT.md', current_raw),
                 (ARCHIVE / 'tables/third-stage-checkpoint-scores.md',
                  contents['tables/third-stage-checkpoint-scores.md'])]
    for origin, raw in documents:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', raw.decode()):
            target = target.strip('<>').split('#', 1)[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            path = (origin.parent / target).resolve()
            require('gpu.py' not in path.parts, 'Protected helper link is outside this audit')
            if path == destination:
                deferred_self_links += 1
            else:
                require(path.exists(), 'Broken local documentation link')
            links += 1
    for relative, bound in payloads.items():
        require(snapshot(ARCHIVE / relative)[0] == bound, 'Archive payload changed during checks')
    require(snapshot(STORY / 'CURRENT_EXPERIMENT.md')[0] == current, 'Current handoff moved')
    result = {
        'scope': 'complete_third_stage_local_archive_verification',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'payloads': payloads, 'exact_working_copies': len(copies),
        'prior_handoff_preserved': True, 'protected_originals': protected_records,
        'original_assembled_source_files_unchanged': source_checks,
        'previous_forty_two_records_exactly_unchanged': True,
        'complete_checkpoint_outcomes': 48, 'new_checkpoint_outcomes': 6,
        'raw_task_values': 672, 'new_raw_task_values': 84,
        'three_archived_replay_tables_byte_identical': True,
        'local_document_links_checked': links, 'deferred_verification_self_links': deferred_self_links,
        'credential_pattern_matches': 0, 'current_handoff': current,
        'native_readback_session': 78814, 'native_readback_exit_code': 0,
        'new_model_or_statistical_execution': False, 'new_outcomes_uploaded_to_hf': False,
        'functional_recovery_authorized': False, 'functional_recovery_launched': False,
        'source_code_published': False, 'scientific_completion': False,
    }
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'verification': snapshot(destination)[0],
                      'payloads': len(payloads), 'exact_working_copies': len(copies),
                      'original_source_files_unchanged': source_checks,
                      'protected_originals_unchanged': len(protected_records),
                      'links_checked': links, 'scientific_completion': False}))


if __name__ == '__main__':
    main()
