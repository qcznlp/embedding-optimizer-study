"""Numerically replay saved functional inference, without model/primary admission.

The original actual-data admission/readback remains in actual/readout.json.
This separate CPU calculation uses only copied source, saved population metadata
and authenticated feature tables. It does not authenticate checkpoints, encode
models, change a contract, or claim formal publication readiness.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ORIGINAL = '/root/embedding-optimizer-story-refactor/'
READOUT_SHA = '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e'
INIT_SHA = 'ddfea072915535be4c693c1263f18e17a47ecc7c25d2f6dbe86ff6d2fad7646e'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary replay input')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
         'Replay input changed')
    return dict(bytes=after.st_size, sha256=sha)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class SavedPopulation:
    """Only recipe/stage metadata for pure helpers; no admission API or contract."""
    def __init__(self, panel):
        self.recipes = {}
        for row in panel:
            recipe = dict(run_id=row['run_id'], optimizer=dict(name=row['optimizer'], lr=row['learning_rate']))
            previous = self.recipes.setdefault(row['run_id'], recipe)
            need(previous == recipe, 'Inconsistent saved recipe metadata')
        need(len(panel) == 60 and len(self.recipes) == 12, 'Incomplete saved population')
        self.inputs = dict(runs=[dict(run_id=run) for run in sorted(self.recipes)])
        self.payload = dict(checkpoint_steps=[782, 1563, 2345, 3126, 3907])

    def expected_identity(self, run_id):
        return dict(recipe=self.recipes[run_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Replay must hide CUDA')
    need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute output')
    need(identity(HERE / 'actual/readout.json')['sha256'] == READOUT_SHA, 'Original actual readout differs')
    readout = json.loads((HERE / 'actual/readout.json').read_bytes())
    bound_sources = {}
    for original, binding in readout['source_bindings'].items():
        if original.startswith(ORIGINAL + 'src/'):
            relative = original.removeprefix(ORIGINAL)
            path = HERE / 'source-original' / relative
            need(identity(path) == binding, 'Copied original source differs: ' + relative)
            bound_sources[str(path)] = binding
    need(len(bound_sources) == 42, 'Copied numerical source closure differs')
    init = HERE / 'source-original/src/embed_optim/__init__.py'
    need(identity(init)['sha256'] == INIT_SHA, 'Package initializer differs')
    bound_sources[str(init)] = identity(init)
    inputs = {}
    for name in ('original_panel.json', 'scientific_protocol.json', 'tables.json', 'decisions.json', 'results.tex',
                 'features/checkpoint_summary.csv', 'features/task_summary.csv',
                 'features/random_removal.csv', 'features/rotation_summary.csv'):
        path = HERE / 'actual' / name
        need(identity(path) == readout['payloads'][name], 'Saved input differs: ' + name)
        inputs[name] = identity(path)
    sys.path.insert(0, str(HERE / 'source-original/src'))
    from embed_optim.dimension_inference import summarize
    from embed_optim.dimension_inference_render import render
    from embed_optim.primary_contract import require_same
    typed_path = HERE / 'source-original/src/embed_optim/primary_v3_exact_bridge.py'
    nodes = [node for node in ast.parse(typed_path.read_bytes()).body if isinstance(node, ast.FunctionDef) and node.name == 'typed_csv']
    need(len(nodes) == 1, 'Original typed parser missing')
    namespace = dict(csv=csv, json=json, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(typed_path), 'exec'), namespace)
    tables = {name: namespace['typed_csv'](HERE / 'actual/features' / (name + '.csv'))
              for name in ('checkpoint_summary', 'task_summary', 'random_removal', 'rotation_summary')}
    panel = json.loads((HERE / 'actual/original_panel.json').read_bytes())
    protocol = json.loads((HERE / 'actual/scientific_protocol.json').read_bytes())
    tasks = sorted({row['task'] for row in tables['task_summary']})
    calculated, decisions = summarize(SavedPopulation(panel), tables, panel, protocol, tasks)
    rendered = render(calculated, decisions)
    require_same(calculated, json.loads((HERE / 'actual/tables.json').read_bytes()))
    require_same(decisions, json.loads((HERE / 'actual/decisions.json').read_bytes()))
    need(rendered.encode() == (HERE / 'actual/results.tex').read_bytes(), 'Generated TeX differs')
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = str(Path(module.__file__).resolve())
            need(path in bound_sources, 'Foreign or unbound project import: ' + name)
            need(identity(Path(path)) == bound_sources[path], 'Loaded source changed')
            imports[name] = dict(path=path, **bound_sources[path])
    for path, binding in bound_sources.items():
        need(identity(Path(path)) == binding, 'Copied source changed after replay')
    for name, binding in inputs.items():
        need(identity(HERE / 'actual' / name) == binding, 'Saved input changed after replay')
    import torch
    need(not torch.cuda.is_initialized(), 'Numerical replay initialized CUDA')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'tables.json', calculated)
    write(args.output / 'decisions.json', decisions)
    with (args.output / 'results.tex').open('x') as stream:
        stream.write(rendered)
    write(args.output / 'verification.json', dict(scope='saved-functional-inference-numerical-replay-only',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), source=identity(Path(__file__)),
        original_readout_sha256=READOUT_SHA, inputs=inputs, copied_source_bindings=bound_sources,
        loaded_project_modules=imports, table_counts={k: len(v) for k, v in calculated.items()},
        all_tables_and_decisions_exact=True, generated_tex_byte_exact=True,
        package_initializer_is_explicit_addition_to_original_archive=True,
        raw_coordinate_computation_repeated=False, model_admission=False,
        source_publication=False, scientific_completion=False))
    print(json.dumps(dict(output=str(args.output), verification=identity(args.output / 'verification.json'),
                          all_tables_and_decisions_exact=True, generated_tex_byte_exact=True)))


if __name__ == '__main__':
    main()
