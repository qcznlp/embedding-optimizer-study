"""Final local evidence inventory; not a remote publication or scientific gate."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.resolve()
STORY = ROOT.parents[2]
assert STORY.name == 'embedding-optimizer-story-refactor'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if (ROOT / 'archive-manifest.json').exists():
    raise ValueError('Preserve an existing seal')
completion = json.loads((ROOT / 'actual/completed.json').read_text())
independent = json.loads((ROOT / 'independent-hf-readback.json').read_text())
assert completion['states'] == independent['states'] == 61
assert completion['task_states'] == independent['task_states'] == 854
assert completion['original_task_mean_attributions_exact']
assert independent['original_producer_inputs_used'] is False
assert sha(ROOT / 'cosine_readout.py') == completion['source_sha256']
assert sha(ROOT / 'verify_cosine_readout.py') == independent['source_sha256']
inputs = json.loads((ROOT / 'actual/input_bindings.json').read_text())
for name, expected in inputs['frozen_sources'].items():
    assert sha(Path(name)) == expected
    assert sha(ROOT / 'frozen-source' / Path(name).relative_to(STORY)) == expected
proposal = json.loads((ROOT / 'deferred-prose-candidate.json').read_text())
main = (STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/paper-preview-v2/main.tex').read_text()
bib = (STORY / 'paper/references.bib').read_text()
for replacement in proposal['replacements']:
    assert (bib if replacement['surface'] == 'bibliography' else main).count(replacement['old']) == 1
(ROOT / 'ops-after').mkdir()
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    shutil.copy2(STORY / name, ROOT / 'ops-after' / name)
patterns = [re.compile(p) for p in (
    r'wandb_v1_[A-Za-z0-9_-]{25,}', r'hf_[A-Za-z0-9]{25,}',
    r'github_pat_[A-Za-z0-9_]{30,}', r'gh[pousr]_[A-Za-z0-9]{25,}',
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
)]
files = {}
for path in sorted(ROOT.rglob('*')):
    if path.is_symlink():
        raise ValueError('Unexpected report symlink')
    if not path.is_file():
        continue
    if path.suffix in ('.py', '.md', '.json', '.tex', '.bib', '.csv'):
        if any(p.search(path.read_text()) for p in patterns):
            raise ValueError('Credential-shaped content found; do not publish')
    files[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
record = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'actual_post_result_cosine_sensitivity_and_deferred_scientific_wording',
          'files': files, 'credential_shaped_findings': 0,
          'frozen_original_sources_unchanged': True, 'prose_replacements_installed': False,
          'scientific_completion': False, 'remote_source_publication': False}
with (ROOT / 'archive-manifest.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
for name, binding in files.items():
    assert sha(ROOT / name) == binding['sha256']
print(json.dumps({'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
                  'manifest_sha256': sha(ROOT / 'archive-manifest.json'),
                  'all_retained_files_rehashed': True, 'frozen_sources_unchanged': True,
                  'credential_shaped_findings': 0, 'full_goal_complete': False}))
