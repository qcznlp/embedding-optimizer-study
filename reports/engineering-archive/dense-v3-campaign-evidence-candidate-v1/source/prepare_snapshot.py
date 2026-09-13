"""One-shot, local data-only copy for isolated completion-reader checks."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import campaign_evidence as c

LAUNCH = Path('/root/embedding-optimizer-v3-experiment/launch')
NATIVE = Path('/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-training-observations-v1')


def main():
    target = Path(sys.argv[1])
    c.require(target.is_absolute() and not target.exists() and not target.is_symlink(), 'Require new absolute snapshot')
    anchors = {
        'admission.json': NATIVE / 'inputs/admission.json',
        'input-view-audit.json': LAUNCH / 'view-history-continuation/input-view-audit.json',
        'source-assembly.json': LAUNCH / 'source-snapshot/source-assembly.json',
        'evaluation-authorization.json': LAUNCH / 'evaluation-handoff/authorization.json',
    }
    values = {name: c.read_local(path.parent, path.name, sha, size)
              for name, path in anchors.items() for size, sha in [c.ANCHORS[name]]}
    roles = c._roles(values)
    sources = {f'anchors/{name}': (path, {'bytes': c.ANCHORS[name][0], 'sha256': c.ANCHORS[name][1]})
               for name, path in anchors.items()}
    for name, binding in roles.items():
        if name.startswith('native/'):
            source = NATIVE / 'inputs' / name
        else:
            basename = name.removeprefix('receipts/')
            run, kind, _ = basename.split('.')
            if kind == 'proof':
                source = (LAUNCH / 'orphan-completion-recovery/receipts' / f'{run}.artifacts-verified.json'
                          if run in c.TERMINAL else LAUNCH / 'logs' / f'{run}.view-verified.json')
            elif kind == 'terminated':
                source = LAUNCH / 'orphan-completion-recovery/receipts' / f'{run}.process-terminated.json'
            else:
                source = LAUNCH / 'logs' / basename
        sources[name] = source, binding
    c.require(len(sources) == 172, 'Wrong original source population')
    checked = {}
    for name, (source, expected) in sources.items():
        c.read_local(source.parent, source.name, expected['sha256'], expected.get('bytes'))
        raw = source.read_bytes()
        c.require(re.search(rb'(?:wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})', raw) is None,
                  'Credential-shaped content refused')
        checked[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        c.same(checked[name]['sha256'], expected['sha256'], 'Source changed after read')
    target.mkdir()
    for name, (source, _) in sources.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        c.require(not destination.exists(), 'Refuse overwrite')
        shutil.copyfile(source, destination)
        c.read_local(target, name, checked[name]['sha256'], checked[name]['bytes'])
    print(json.dumps({'scope': 'isolated-local-training-evidence-copy', 'files': len(sources),
                      'bytes': sum(x['bytes'] for x in checked.values()), 'target': str(target),
                      'copied_files': checked, 'new_authority': False, 'scientific_completion': False}, sort_keys=True))


if __name__ == '__main__':
    main()
