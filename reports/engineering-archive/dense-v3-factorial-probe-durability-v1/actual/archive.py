"""Preserve actual complete durability/replay evidence; bulky data stay on HF."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
TARGET = STORY / 'reports/engineering-archive/dense-v3-factorial-probe-durability-v1'
TARGET.mkdir(parents=True, exist_ok=False)
files = {}


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Nonordinary evidence')
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def copy(source, name):
    binding = identity(source)
    destination = TARGET / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as original, destination.open('xb') as out:
        shutil.copyfileobj(original, out)
    if identity(source) != binding or identity(destination) != binding:
        raise ValueError('Evidence copy differs')
    files[name] = {'origin': str(source), **binding}


for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    copy(STORY / name, 'before/' + name)
copy(STORY / 'scripts/restore_factorial_probes.py', 'restoration/restore_factorial_probes.py')
for path in sorted(HERE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and not path.is_relative_to(HERE / 'download'):
        copy(path, 'actual/' + path.relative_to(HERE).as_posix())
download = json.loads((HERE / 'download-verified.json').read_text())
remote = json.loads((HERE / 'remote-verified.json').read_text())
replay = json.loads((HERE / 'numerical-replay/actual-replay.json').read_text())
if not (download['all_checksums_match'] is True and download['downloaded_anonymously'] is True
        and remote['all_remote_files_verified'] is True and remote['old_root_and_subtrees_preserved'] is True
        and download['files'] == remote['files'] == 638 and download['bytes'] == remote['bytes'] == 586119383
        and download['revision'] == remote['revision'] == 'cff3f190e169548931fbd33eadcf1279439798e1'
        and replay['actual_states'] == 61 and replay['exact_original_metric_values'] == 5490
        and len(replay['observations']) == 61 and all(row['all_scores_exact'] and row['all_six_metrics_exact']
            and row['raw_to_scoring_arrays_exact'] for row in replay['observations'])):
    raise ValueError('Incomplete actual durability or replay')
with (TARGET / 'archive-manifest.json').open('x') as stream:
    json.dump({'archived_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'copied_files': len(files), 'copied_bytes': sum(v['bytes'] for v in files.values()),
        'copy_hashes_verified': True, 'remote_probe_files': 638, 'remote_probe_bytes': 586119383,
        'all_remote_payloads_anonymously_downloaded': True, 'numerical_replay_states': 61,
        'bulky_npz_payloads_not_duplicated_in_repository': True,
        'source_code_uploaded': False, 'full_factorial_beir_complete': False,
        'full_goal_complete': False}, stream, sort_keys=True, indent=2)
    stream.write('\n')
print({'archive': str(TARGET), 'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
       'manifest': identity(TARGET / 'archive-manifest.json')})
