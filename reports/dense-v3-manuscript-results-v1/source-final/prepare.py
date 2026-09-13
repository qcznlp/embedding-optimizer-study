"""Copy actual, hash-bound numerical evidence; never admit models or install paper files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
READOUTS = {
    'trajectory': ('dense-v3-complete-trajectories-v1/tables', 'dd48e622c9cd1f28f8e93cab670349fc937e8e64aa50b98947104151dbda42c2'),
    'weight': ('dense-v3-weight-retrieval-v1/actual', '6891e82a46df11a6d142e79fdf495afa329f40912569337f4b1d6da33d5e8046'),
    'functional': ('dense-v3-functional-inference-v1/actual', '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'),
    'weight_controls': ('dense-v3-predictor-sensitivity-v1/actual', '00009a1f0253d9dfa6de9938f016f20f339f1e0729e6ce31886aca9a96a2db64'),
    'functional_controls': ('dense-v3-functional-sensitivity-v1/actual', '8cb9b93da12a245e214429ffcf61f6922803c72c9c9ea496e572d24ecdc3e231'),
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary input')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input changed')
    return dict(bytes=after.st_size, sha256=sha)


def relative(name):
    p = Path(name)
    need(isinstance(name, str) and not p.is_absolute() and '..' not in p.parts
         and p.as_posix() == name and name not in ('', '.') and '\\' not in name,
         'Noncanonical relative path')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    need(args.output.is_absolute() and not output.exists()
         and not any(p.is_symlink() for p in (output, *output.parents)), 'Use a new ordinary output')
    records, values = {}, {}

    def add(source, destination, expected=None):
        relative(destination)
        actual = identity(source)
        if isinstance(expected, str):
            need(actual['sha256'] == expected, 'Original SHA differs: ' + str(source))
        elif expected is not None:
            need(actual == {k: expected[k] for k in actual}, 'Original binding differs: ' + str(source))
        if destination in records:
            need(records[destination]['binding'] == actual, 'Conflicting copied source')
            return
        records[destination] = dict(origin=str(source), binding=actual)

    for family, (origin, sha) in READOUTS.items():
        base = ROOT / 'reports' / origin
        add(base / 'readout.json', 'inputs/' + family + '/readout.json', sha)
        value = json.loads((base / 'readout.json').read_bytes())
        values[family] = value
        if family == 'functional':
            names = ('tables.json', 'decisions.json', 'results.tex', 'original_panel.json', 'scientific_protocol.json')
            bindings = value['payloads']
        else:
            names, bindings = list(value['outputs']), value['outputs']
        for name in names:
            add(base / relative(name), 'inputs/' + family + '/' + name, bindings[name])

    geometry = [(p, b) for p, b in values['weight']['input_bindings'].items()
                if p.endswith('/approximate/checkpoint_geometry.csv')]
    need(len(geometry) == 1, 'Ambiguous actual geometry source')
    add(Path(geometry[0][0]), 'inputs/weight/checkpoint_geometry.csv', geometry[0][1])
    # Copy exact original numerical dependencies already admitted by the actual weight readout.
    for origin, binding in values['weight']['input_bindings'].items():
        prefix = str(ROOT) + '/'
        if origin.startswith(prefix + 'src/') or origin.startswith(prefix + 'configs/'):
            add(Path(origin), 'source-original/' + origin.removeprefix(prefix), binding)
    source = ROOT / 'reports/dense-v3-weight-retrieval-v1/source/readout.py'
    add(source, 'source/weight_readout.py',
        'b2cd61b867943916b6ed99982864ac228a6e982f996296c3250778a9cde80792')
    # This renderer-only extension authenticates against the existing prepared protocol.
    protocol = ROOT / 'configs/dense_primary_v3_exact_publication_protocol.json'
    add(protocol, 'source-original/configs/dense_primary_v3_exact_publication_protocol.json',
        'e8409f534931b85ad855fbdd3804e3ef8d771b0ee22ee99d9b81bf07d605f83e')
    payload = json.loads(protocol.read_bytes())
    for name in ('corrected_publication', 'primary_v3_publication_bridge',
                 'primary_v3_publication_render', 'primary_v3_exact_publication_render'):
        name = 'src/embed_optim/' + name + '.py'
        add(ROOT / name, 'source-original/' + name, payload['sources'][name])
    # Retain actual numerical verification, not a relabelled synthetic acceptance.
    for name, expected in (
        ('dense-v3-functional-inference-v1/actual/independent_verification.json', '03f9670ab2a8a2334db439c95f1ef6264485cb816addd48457df305f64b45e23'),
        ('dense-v3-functional-sensitivity-v1/actual/independent_verification.json', None),
        ('dense-v3-weight-retrieval-v1/independent-predictions.json', None),
        ('dense-v3-predictor-sensitivity-v1/verification.json', None),
        ('dense-v3-complete-trajectories-v1/verification.json', 'c21ac61b21fc00907011758ca64d92f1e27478942644462d834477e88054b4bc'),
    ):
        add(ROOT / 'reports' / name, 'evidence/' + name, expected)
    # Preparation validates all inputs before copying the first byte.
    output.mkdir(parents=True, exist_ok=False)
    for name, row in records.items():
        source, dest = Path(row['origin']), output / name
        need(identity(source) == row['binding'], 'Input changed before copy')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, dest.open('xb') as out:
            while block := inp.read(1024 * 1024):
                out.write(block)
        need(identity(dest) == row['binding'] == identity(source), 'Copied bytes differ')
    manifest = dict(scope='actual-numerical-result-rendering-inputs-only', records=records,
        upstream_proofs_preserved=True, checkpoint_admission_repeated=False,
        manuscript_installed=False, source_publication=False, scientific_completion=False)
    with (output / 'inputs.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), copied_files=len(records),
                          manifest=identity(output / 'inputs.json'))))


if __name__ == '__main__':
    main()
