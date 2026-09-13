"""Preserve an isolated source predecessor and record only explicit component replacements."""

import argparse
import json
from pathlib import Path
import shutil

from integration import COMPONENTS, WORKER, identity, now, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--replace', nargs=2, action='append', default=[])
    args = parser.parse_args()
    if identity(args.parent)['sha256'] != args.parent_sha:
        raise ValueError('The original source manifest changed')
    parent = json.loads(args.parent.read_text())
    previous = Path(parent['root'])
    replacements = dict(args.replace)
    if len(replacements) != len(args.replace) or not replacements:
        raise ValueError('Require distinct explicit component replacements')
    if set(replacements) - {*COMPONENTS, WORKER}:
        raise ValueError('The primary source is immutable in this diagnostic')
    for relative, row in parent['files'].items():
        if identity(previous / relative) != {k: row[k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Original isolated source changed: {relative}')
    args.destination.mkdir(parents=False, exist_ok=False)
    changes = {}
    for relative, row in parent['files'].items():
        source = Path(replacements[relative]).resolve() if relative in replacements else previous / relative
        source_id = identity(source)
        if relative in replacements:
            if source_id == {k: row[k] for k in ('bytes', 'sha256')}:
                raise ValueError('A replacement must represent a genuine recorded change')
            changes[relative] = {'before': row, 'after': {'origin': str(source), **source_id}}
            parent['files'][relative] = changes[relative]['after']
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if identity(destination) != source_id:
            raise ValueError('Mechanical source copy mismatch')
    parent.update(root=str(args.destination.resolve()), created_at_utc=now(),
                  previous_manifest={'path': str(args.parent.resolve()), **identity(args.parent)},
                  explicit_changes=changes,
                  revision_tool={'path': str(Path(__file__).resolve()), **identity(__file__)})
    save(args.manifest, parent)
    print(json.dumps({'manifest': str(args.manifest), **identity(args.manifest),
                      'changed_paths': sorted(changes)}), flush=True)


if __name__ == '__main__':
    main()
