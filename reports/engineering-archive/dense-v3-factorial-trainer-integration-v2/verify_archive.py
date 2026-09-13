"""Independent immutable-file/copy and reversible handoff check; no tensor replay."""

import argparse
import json
from pathlib import Path

from integration import PARENT, PARENT_SHA, PRIMARY, identity, now, save


OPS = (
    ('/root/embedding-optimizer-story-refactor/AGENTS.md', 'AGENTS.md',
     '5faca6a21b5841e40012361ac45f2a3b2d30d4ebbc71cb9a46c1871b4769a166'),
    ('/root/embedding-optimizer-story-refactor/PROJECT_STATUS.md', 'PROJECT_STATUS.md',
     '17538af03837c74db8707974391f51779b69c740f37624c87d735e39c9f7c20d'),
    ('/root/embedding-optimizer-v3-experiment/launch/RUNNING.md', 'PRIMARY_RUNNING.md',
     '4a34b1a9244b1ff678b32419a7f09d1ec4a17af4b44b8107f64b129ec0e91597'),
    ('/root/embedding-optimizer-v3-experiment/launch/view-history-continuation/RUNNING.md',
     'CONTINUATION_RUNNING.md', '54aac766169db4a0c5562ff62b804e986a44d890a94e25709f3ad9bcd6b1dc95'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    archive = args.manifest.resolve().parent
    if identity(args.manifest)['sha256'] != args.manifest_sha:
        raise ValueError('The complete bounded receipt changed')
    manifest = json.loads(args.manifest.read_text())
    for path, expected in manifest['immutable_files'].items():
        if identity(path) != expected:
            raise ValueError(f'Archived immutable file changed: {path}')
    original = Path('/tmp/dense-v3-factorial-trainer-integration.dmkUEso6')
    original_files = {p.relative_to(original) for p in original.rglob('*') if p.is_file()}
    copy_root = archive / 'actual'
    copied_files = {p.relative_to(copy_root) for p in copy_root.rglob('*') if p.is_file()}
    if original_files != copied_files:
        raise ValueError('Mechanical archive omitted or added an execution file')
    for relative in original_files:
        if identity(original / relative) != identity(copy_root / relative):
            raise ValueError(f'Original/archive bytes differ: {relative}')
    source_rows = 0
    for name in ('first', 'second', 'third', 'gradient'):
        source_manifest = json.loads((copy_root / f'source-{name}.json').read_text())
        for relative, expected in source_manifest['files'].items():
            target = {key: expected[key] for key in ('bytes', 'sha256')}
            if (identity(copy_root / f'source-{name}' / relative) != target or
                    identity(Path(source_manifest['root']) / relative) != target):
                raise ValueError('A preserved source assembly changed')
            source_rows += 1
    if identity(PARENT)['sha256'] != PARENT_SHA:
        raise ValueError('The original primary source manifest changed')
    primary = json.loads(PARENT.read_text())
    for relative, row in primary['files'].items():
        if identity(PRIMARY / relative) != row['identity']:
            raise ValueError('The running primary source changed')
    anchor = 'protected-helper prohibition, GitHub 403 and rejected historical HF erasure limits.\n'
    additions, operations = [], []
    for live, filename, expected_before in OPS:
        before_path = archive / 'ops-before' / filename
        if identity(before_path)['sha256'] != expected_before:
            raise ValueError('Original operational preimage differs')
        before, after = before_path.read_text(), Path(live).read_text()
        if before.count(anchor) != 1 or after.count(anchor) != 1:
            raise ValueError('Operational insertion is not uniquely scoped')
        prefix, suffix = before.split(anchor)
        prefix += anchor
        if not after.startswith(prefix) or not after.endswith(suffix):
            raise ValueError('Unrelated operational text changed')
        addition = after[len(prefix):len(after) - len(suffix)]
        if not addition.startswith('\n**CPU factorial Trainer integration complete in its bounded scope'):
            raise ValueError('Unexpected operational addition')
        if prefix + after[len(prefix) + len(addition):] != before:
            raise ValueError('Operational addition cannot be reversed exactly')
        additions.append(addition)
        operations.append({'path': live, 'before': identity(before_path), 'after': identity(live)})
    if len(set(additions)) != 1:
        raise ValueError('Operational handoffs disagree')
    result = {
        'scope': 'bounded-cpu-factorial-archive-and-operational-readback',
        'scientific_admission': False, 'observed_at_utc': now(),
        'complete_manifest': {'path': str(args.manifest.resolve()), **identity(args.manifest)},
        'archived_file_identities_verified': len(manifest['immutable_files']),
        'original_execution_files_copied_exactly': len(original_files),
        'source_assembly_rows_verified_in_original_and_archive': source_rows,
        'frozen_primary_sources_unchanged': len(primary['files']),
        'operational_addition': additions[0], 'reversible_operational_edits': operations,
        'reader': {'path': str(Path(__file__).resolve()), **identity(__file__)},
        'fresh_tensor_or_gradient_replay': False,
    }
    save(args.output, result)
    print(json.dumps({'verification': str(args.output), **identity(args.output),
                      'archived_files': len(manifest['immutable_files']),
                      'exact_copied_files': len(original_files),
                      'source_rows_both_copies': source_rows, 'reversible_operations': len(operations)}),
          flush=True)


if __name__ == '__main__':
    main()
