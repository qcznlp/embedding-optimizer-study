"""Archive the union of two already-bound source closures for the new probe."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PARENTS = {
    STORY / 'reports/dense-v3-functional-inference-v1/actual/readout.json':
        ('608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e', 'source_bindings'),
    Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions/inputs.json'):
        ('be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067', 'sources'),
}


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Nonordinary source')
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


bindings = {}
for parent, (digest, field) in PARENTS.items():
    if identity(parent)['sha256'] != digest:
        raise ValueError('Source parent differs')
    for name, binding in json.loads(parent.read_bytes())[field].items():
        if name.startswith(str(STORY / 'src') + '/'):
            if bindings.setdefault(name, binding) != binding:
                raise ValueError('Original source parents disagree')
bindings[str(STORY / 'src/embed_optim/__init__.py')] = dict(bytes=84,
    sha256='ddfea072915535be4c693c1263f18e17a47ecc7c25d2f6dbe86ff6d2fad7646e')
copied = {}
for name, binding in sorted(bindings.items()):
    source = Path(name)
    target = HERE / 'source' / source.relative_to(STORY)
    if identity(source) != binding or target.exists():
        raise ValueError('Source changed or target already exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if identity(target) != binding or identity(source) != binding:
        raise ValueError('Copy differs')
    copied[str(target)] = binding
with (HERE / 'source-copy.json').open('x') as stream:
    json.dump(dict(parents={str(p): d for p, (d, _) in PARENTS.items()}, copied_files=copied,
                   scientific_completion=False), stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(copied_unchanged_sources=len(copied), source_copy=identity(HERE / 'source-copy.json'))))
