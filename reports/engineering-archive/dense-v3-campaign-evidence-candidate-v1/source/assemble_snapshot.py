"""Compose a new local proof snapshot from recovered HF telemetry and 39 records.

No download, model computation, authorization or replacement of an existing root.
"""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil

import campaign_evidence as c

TRAINING_MANIFEST = '9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e'


def assemble(training_root, additional_root, target):
    training_root, additional_root = c.ordinary_root(training_root), c.ordinary_root(additional_root)
    target = Path(target)
    c.require(target.is_absolute() and '..' not in target.parts
              and not target.exists() and not target.is_symlink()
              and all(not p.is_symlink() for p in target.parents), 'Require new absolute destination')
    c.read_local(training_root, 'artifact_manifest.json', TRAINING_MANIFEST)
    additional_before = c.inventory(additional_root)
    sources = {}
    anchors = {}
    for name, (size, sha) in c.ANCHORS.items():
        root, role = ((training_root, 'native/inputs/admission.json') if name == 'admission.json'
                      else (additional_root, f'anchors/{name}'))
        anchors[name] = c.read_local(root, role, sha, size)
        sources[f'anchors/{name}'] = (root, role, {'bytes': size, 'sha256': sha})
    for name, binding in c._roles(anchors).items():
        root, role = ((training_root, 'native/inputs/' + name) if name.startswith('native/')
                      else (additional_root, name))
        c.read_local(root, role, binding['sha256'], binding.get('bytes'))
        sources[name] = (root, role, binding)
    additional_names = {role for root, role, binding in sources.values() if root == additional_root}
    c.require(len(additional_names) == 39 and len(sources) == 172, 'Wrong composition population')
    directories = {''} | {p.as_posix() for name in additional_names for p in PurePosixPath(name).parents if p.as_posix() != '.'}
    c.same(sorted(additional_before), sorted(additional_names | directories), 'Unexpected additional records')
    target.mkdir()
    for name, (root, role, binding) in sources.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive output and anchored post-copy reads preserve any partial failed attempt.
        with (root / role).open('rb') as source, destination.open('xb') as output:
            shutil.copyfileobj(source, output)
        c.read_local(target, name, binding['sha256'], binding.get('bytes'))
    c.same(c.inventory(additional_root), additional_before, 'Additional source records changed')
    return c.read_complete_training_population(target)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recovered_training_prefix', type=Path)
    parser.add_argument('additional_records', type=Path)
    parser.add_argument('new_destination', type=Path)
    args = parser.parse_args()
    result = assemble(args.recovered_training_prefix, args.additional_records, args.new_destination)
    print(json.dumps({k: v for k, v in result.items() if k != 'original_completion_rows'}, sort_keys=True))


if __name__ == '__main__':
    main()
