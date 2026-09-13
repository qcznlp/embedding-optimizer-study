"""Verify the original audit archive; add separate dated documentation copies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

STORY = Path('/root/embedding-optimizer-story-refactor')
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-final-tracking-audit-v1'
MANIFEST_SHA = 'f9ed436522827830749f8eceb19a8436aafc41aa2760827d8ca903e97c54ba12'
FROZEN = {
    '/tmp/dense-v3-factorial-evaluation.IOV7MK93/evaluate.py': '45577537257e1dff2b7c3a26ee12f7564fbb476daa256e7b0f09300305b68cba',
    '/tmp/dense-v3-factorial-summary.BQ08HjeP/summarize.py': '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1',
    '/tmp/dense-v3-document-integration.Xz1qvTME/author.py': 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d',
    '/tmp/dense-v3-combined-paper-replay.URO4uk9L/combined.py': '1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566',
    '/tmp/dense-v3-resume-verification.wcsmQnDm/resume.py': '1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2',
    '/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1/factorial_dispatch.py': 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53',
    '/root/embedding-optimizer-story-refactor/paper/main.tex': '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
}


def identify(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Nonordinary input')
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    manifest = ARCHIVE / 'copy-inventory.json'
    if identify(manifest)['sha256'] != MANIFEST_SHA:
        raise ValueError('Original archive manifest changed')
    old = json.loads(manifest.read_bytes())
    for row in old['files']:
        if identify(ARCHIVE / row['path']) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Original archive copy changed')
    for path, expected in FROZEN.items():
        if identify(path)['sha256'] != expected:
            raise ValueError('Original frozen source changed')
    target = ARCHIVE / 'final-ops'
    target.mkdir(exist_ok=False)
    sources = [STORY / p for p in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md')]
    sources.append(Path(__file__))
    rows = []
    for source in sources:
        data = source.read_bytes()
        if re.search(rb'wandb_v1_[A-Za-z0-9_-]{40,}|\bhf_[A-Za-z0-9]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', data):
            raise ValueError('Credential-shaped documentation refused')
        copy = target / source.name
        with copy.open('xb') as stream:
            stream.write(data)
        if identify(copy) != identify(source):
            raise ValueError('Documentation copy differs')
        rows.append({'source': str(source), 'path': copy.relative_to(ARCHIVE).as_posix(), **identify(copy)})
    result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'goal_turn': 'PROGRESS',
              'original_archive_manifest_sha256': MANIFEST_SHA, 'original_archive_files_verified': len(old['files']),
              'additional_files': rows, 'frozen_sources_unchanged': FROZEN,
              'archive_tool_terminal': 'd03e7d', 'archive_tool_exit_code': 0,
              'documentation_diff_check_terminal': '27afc2', 'documentation_diff_check_exit_code': 0,
              'scientific_runs_complete': 24, 'immutable_model_checkpoints_verified': 120,
              'beir_observation': 'work/evaluation-third.json', 'beir_task_cells_at_observation': 10,
              'beir_task_cells_expected': 168, 'tracking_inventory_complete': True,
              'full_goal_complete': False, 'source_release': False, 'authoritative_manuscript_changed': False}
    output = ARCHIVE / 'final-handoff.json'
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'final_handoff': identify(output), 'original_copies_verified': len(old['files']),
                      'additional_copies': len(rows), 'frozen_sources_verified': len(FROZEN),
                      'goal_turn': 'PROGRESS', 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
