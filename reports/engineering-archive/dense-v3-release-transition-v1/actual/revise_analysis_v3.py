"""Preserve failed v2; restore three exact historical publication dependencies."""
import hashlib
import json
from pathlib import Path
import shutil

work = Path(__file__).resolve().parent
root = Path('/root/embedding-optimizer-story-refactor')
old = json.loads((work / 'analysis-v2-role.json').read_text())
new_root = work / 'original-analysis-v3'
new_root.mkdir(exist_ok=False)
updates = {
    'src/embed_optim/short_branch.py': ('reports/engineering-archive/dense-v3-numerical-source-integration-v1/before/src/embed_optim/short_branch.py', '828345379b4cd4312b6b8a1881a84a7b3116b16fa09ad402e2c1b0e0321c2a4f'),
    'paper/main.tex': ('reports/paper-review/dense-v3-retrieval-usefulness-v1/before/paper/main.tex', '4d59c3219589204ec396d5b9d8c6269df1dfc89f29361ddf70ada26100dbbb2b'),
    'paper/Makefile': ('reports/engineering-archive/dense-v3-current-paper-entry-v1/actual/before/paper/Makefile', '70f96295cba43d07ddefb0c895a55937c4c720ef67d84bfc18b36a597097af92'),
}
for name, (origin, expected) in updates.items():
    assert hashlib.sha256((root / origin).read_bytes()).hexdigest() == expected
for source in sorted(Path(old['root']).rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts or '.pytest_cache' in source.parts:
        continue
    assert not source.is_symlink()
    name = str(source.relative_to(old['root']))
    actual = root / updates[name][0] if name in updates else source
    target = new_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(actual, target)
    assert target.read_bytes() == actual.read_bytes()
changes = {**old['changes'], **{name: {'origin': str(root / origin), 'sha256': sha} for name, (origin, sha) in updates.items()}}
record = {**old, 'root': str(new_root), 'changes': changes, 'prior_v2_failure_preserved': True}
with (work / 'analysis-v3-role.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
print(json.dumps({'source_deltas': len(changes), 'root': str(new_root)}), flush=True)
