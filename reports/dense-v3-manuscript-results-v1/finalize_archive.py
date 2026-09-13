"""Archive actual generated fragments, preview and unchanged ops preimages."""
import json
import shutil
from pathlib import Path
from archive import HERE, WORK, identity


def main():
    pairs = []
    for source in sorted((WORK / 'render-final').iterdir()):
        pairs.append((source, HERE / 'actual' / source.name))
    for name in ('render.py', 'prepare.py', 'test_render.py', 'preview.tex'):
        pairs.append((HERE / name, HERE / 'source-final' / name))
    for suffix in ('pdf', 'log', 'aux'):
        pairs.append((WORK / ('preview-complete.' + suffix), HERE / 'preview' / ('results.' + suffix)))
    for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
        pairs.append((HERE.parents[1] / name, HERE / 'before' / name))
    for name in ('training-1815.json', 'training-1817.json'):
        pairs.append((WORK / name, HERE / 'observations' / name))
    records = {}
    for source, target in pairs:
        before = identity(source)
        if target.exists():
            raise ValueError('Existing archive target')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
        if identity(source) != before or identity(target) != before:
            raise ValueError('Copy differs')
        records[str(target.relative_to(HERE))] = dict(origin=str(source), binding=before)
    if any(x in (HERE / 'preview/results.log').read_text() for x in
           ('Overfull', 'undefined references', 'undefined citations', 'Label(s) may have changed')):
        raise ValueError('Unresolved preview compiler condition')
    receipt = dict(scope='actual-scientific-result-fragments-not-paper-release', copies=records,
        first_preservation=identity(HERE / 'first-preservation.json'),
        input_manifest=identity(HERE / 'bundle/inputs.json'),
        final_tests_passed=20, actual_test_terminal_chunk='f0b42a',
        actual_first_preparation_chunk='60f91c', actual_final_generation_chunk='fc4623',
        actual_fresh_verification_chunk='f10772',
        final_preview_actual_compiler_chunk='c9c0b1',
        final_preview_all_five_pages_visually_inspected=True,
        final_preview_all_14_font_records_embedded_type1=True,
        full_manuscript_or_gpu_admission=False, scientific_completion=False)
    with (HERE / 'verification.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copies=len(records), receipt=identity(HERE / 'verification.json'))))


if __name__ == '__main__':
    main()
