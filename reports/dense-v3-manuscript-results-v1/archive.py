"""Preserve the actual fragment construction and exact copied input closure."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/dense-v3-publication-close-loop.462cejZZ')


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Nonordinary archive input')
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=sha)


def main():
    records = {}
    pairs = []
    for origin, target in ((WORK / 'bundle', HERE / 'bundle'),
                           (WORK / 'render-first', HERE / 'actual-first')):
        if target.exists():
            raise ValueError('Archive destination exists')
        for source in sorted(origin.rglob('*')):
            if source.is_symlink():
                raise ValueError('Symlinked archive input')
            if source.is_file():
                pairs.append((source, target / source.relative_to(origin)))
    for name in ('render.py', 'prepare.py', 'test_render.py', 'preview.tex'):
        pairs.append((HERE / name, HERE / 'first-source' / name))
    for name in ('preview.pdf', 'preview.log', 'preview.aux',
                 'training-1800.json', 'evaluation-1800.json', 'probe-1800.json', 'summary-1800.json',
                 'training-1810.json'):
        pairs.append((WORK / name, HERE / 'first-preview-and-observations' / name))
    for source, target in pairs:
        before = identity(source)
        if target.exists():
            raise ValueError('Existing archive file')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
        if identity(target) != before or identity(source) != before:
            raise ValueError('Copied archive bytes differ')
        records[str(target.relative_to(HERE))] = dict(origin=str(source), binding=before)
    receipt = dict(scope='actual-result-fragment-first-preservation', copies=records,
        original_inputs_and_numerical_sources_unchanged=True,
        first_preview_pages_visually_inspected=[1, 2, 3, 4, 5],
        first_preview_font_records_all_embedded_type1=14,
        manuscript_installed=False, scientific_completion=False)
    with (HERE / 'first-preservation.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copied=len(records), receipt=identity(HERE / 'first-preservation.json'))))


if __name__ == '__main__':
    main()
