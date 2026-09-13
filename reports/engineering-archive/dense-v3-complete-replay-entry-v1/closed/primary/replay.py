"""Reconstruct the complete current primary numerical result graph from local roles.

The full authentic model/data admission is an upstream, preserved record, not
performed by the metadata catalog below. All original numerical functions and
all rates/stages/tasks/features/controls are retained. No original producer path
is opened. The actual crossed outcomes and complete-paper release remain separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary local input required')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, key) == getattr(after, key) for key in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input changed')
    return {'bytes': after.st_size, 'sha256': sha}


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def authenticate(root, sha):
    need(identity(root / 'manifest.json')['sha256'] == sha, 'External closure anchor differs')
    manifest = read(root / 'manifest.json')
    need(manifest['scope'] == 'closed-current-primary-numerical-and-document-inputs-v1'
         and manifest['native_primary_assembly_sha256'] ==
         '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc', 'Wrong actual primary lineage')
    names = set()
    for name, expected in manifest['files'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name, 'Nonlocal input role')
        need(identity(root / name) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Copied input changed: ' + name)
        names.add(name)
    observed = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(observed == names | {'manifest.json'}, 'Missing or additional closed input')
    return manifest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SavedRecipeCatalog:
    """Read-only numerical recipe fields; not a PrimaryV3Contract or admission API."""
    def __init__(self, value):
        need(value['scope'] == 'saved-genuine-primary-recipe-metadata-only'
             and value['model_admission_performed_here'] is False, 'Metadata scope differs')
        self.inputs = value['inputs']
        self.payload = value['payload']
        self.sha256 = value['primary_protocol_sha256']
        self.identities = value['expected_identities']
        need(len(self.inputs['runs']) == len(self.identities) == 12, 'Incomplete recipe catalog')

    def expected_identity(self, run_id):
        return self.identities[run_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONPATH', '') == '',
         'Use a clean CPU-only replay namespace')
    root, output = args.bundle, args.output
    need(root.is_absolute() and output.is_absolute() and not output.exists()
         and not output.is_relative_to(root), 'Use separate ordinary absolute roles')
    manifest = authenticate(root, args.manifest_sha256)
    need(not any(name == 'embed_optim' or name.startswith('embed_optim.') for name in sys.modules),
         'A project package was preimported')
    sys.path.insert(0, str(root / 'src'))
    from embed_optim import primary_v3_outcomes as outcomes
    from embed_optim import primary_v3_bridge as bridge
    from embed_optim import primary_v3_exact_bridge as exact_bridge
    from embed_optim import dimension_inference as dimension
    from embed_optim import primary_v3_publication as publication
    from embed_optim import primary_v3_exact_publication as exact_publication
    from embed_optim import primary_v3_manuscript as manuscript
    from embed_optim import bridge_exact_arithmetic as arithmetic
    from embed_optim.primary_contract import digest, require_same
    import torch
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(), 'CPU reconstruction initialized CUDA')
    catalog = SavedRecipeCatalog(read(root / 'inputs/recipe-catalog.json'))
    native = read(root / 'inputs/native-primary/evidence.json')
    expected = read(root / 'inputs/scientific-publication-v3/exact-publication/tables.json')
    complete = read(root / 'inputs/scientific-publication-v3/completed.json')
    native_runs = native['grid']['runs']
    for run in native_runs:
        need(run['run_identity_sha256'] == digest(catalog.expected_identity(run['run_id'])), 'Native recipe link differs')
    geometry = read(root / 'inputs/native-geometry/tables.json')
    selection = native['validation_selection']
    output.mkdir(exist_ok=False)
    print(json.dumps({'phase': 'original-complete-primary-outcomes', 'task_units': 840}), flush=True)
    actual_outcomes = outcomes.outcome_tables(catalog, native['grid']['score_rows'], selection)
    actual_outcomes['system_metrics'] = outcomes.system_rows(catalog, native_runs)
    require_same(actual_outcomes, expected['outcomes'])
    print(json.dumps({'phase': 'original-and-exact-weight-prediction', 'states': 60}), flush=True)
    original_bridge, original_diagnostics = bridge.bridge_tables(catalog,
        geometry['approximate']['checkpoint_geometry'], geometry['approximate']['run_pair_subspace_overlap'],
        actual_outcomes['run_stage_scores'])
    exact, exact_diagnostics = exact_bridge.sensitivity_tables(catalog, original_bridge,
        geometry['exact']['checkpoint_exact_geometry'], geometry['exact']['run_pair_exact_subspace_overlap'])
    require_same(original_bridge, expected['bridge'])
    require_same(exact, expected['exact_bridge'])
    functional_inputs = {name: exact_bridge.typed_csv(root / 'inputs/functional/features' / (name + '.csv'))
                         for name in ('checkpoint_summary', 'task_summary', 'random_removal', 'rotation_summary')}
    protocol = read(root / 'inputs/functional/scientific_protocol.json')
    require_same(original_bridge['bridge_rows'], read(root / 'inputs/functional/original_panel.json'))
    tasks = sorted(catalog.payload['evaluation']['tasks'])
    print(json.dumps({'phase': 'complete-functional-inference', 'states_including_reference': 61}), flush=True)
    functional, decisions = dimension.summarize(catalog, functional_inputs,
        original_bridge['bridge_rows'], protocol, tasks)
    require_same(functional, expected['functional'])
    require_same(decisions, read(root / 'inputs/functional/decisions.json'))
    all_tables = {'outcomes': actual_outcomes, 'geometry': geometry['approximate'],
                  'bridge': original_bridge, 'functional': functional}
    raw = publication.generate(catalog, all_tables, selection, native_runs, decisions,
        read(root / 'inputs/scientific-publication-v3/original-publication/evidence.json'))
    exact_contents = exact_publication.generate(catalog, raw, geometry['exact'], exact,
        read(root / 'inputs/scientific-publication-v3/exact-publication/evidence.json'))
    reproduced = {}
    for family, contents in (('original-publication', raw), ('exact-publication', exact_contents)):
        for name, content in contents.items():
            target = output / family / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
            reference = complete['outputs'][family + '/' + name]
            need(identity(target) == {k: reference[k] for k in ('bytes', 'sha256')},
                 'Complete original scientific generation differs: ' + family + '/' + name)
            reproduced[family + '/' + name] = identity(target)
    print(json.dumps({'phase': 'all-post-result-controls', 'comparisons': 108}), flush=True)
    weight_source = load('_portable_original_weight_controls', root / 'source/weight_controls.py')
    functional_source = load('_portable_original_functional_controls', root / 'source/functional_controls.py')
    weight_controls = weight_source.calculate({'rows': exact['bridge_rows'], 'arithmetic': arithmetic,
        'features': [r['feature'] for tables in (original_bridge, exact) for r in tables['feature_prediction_summary']],
        'tables': {'original': original_bridge, 'exact': exact}})
    functional_controls = functional_source.calculate(functional, weight_source, arithmetic)
    require_same(weight_controls, read(root / 'inputs/weight_controls.json'))
    require_same(functional_controls, read(root / 'inputs/functional_controls.json'))
    for name, result in (('weight-controls', weight_controls), ('functional-controls', functional_controls)):
        write(output / (name + '.json'), result)
    control_source = load('_portable_original_control_rendering', root / 'source/control_rendering.py')
    from embed_optim.corrected_publication import FEATURE_LABELS, _tex_escape
    from embed_optim.primary_v3_exact_publication_render import LABELS
    labels = {**FEATURE_LABELS, **{key: 'Exact: ' + value for key, value in LABELS.items()}}
    labels['full_spectrum_nonzero_saved_segment_entropy_rank_fraction'] = 'Full-spectrum segment entropy'
    wc = control_source.checked_controls(weight_controls, labels, 84)
    fc = control_source.checked_controls(functional_controls, control_source.FUNCTIONAL_LABELS, 24)
    controls = control_source.controls_tex(wc, fc, labels, _tex_escape).encode()
    need(hashlib.sha256(controls).hexdigest() == complete['outputs']['recipe-sensitivity.tex']['sha256'],
         'Fresh complete exploratory-control LaTeX differs')
    with (output / 'recipe-sensitivity.tex').open('xb') as stream:
        stream.write(controls)
    constants = manuscript.render_constants(catalog)
    need(constants == (root / 'paper-inputs/results.tex').read_bytes(), 'Original generated constants differ')
    with (output / 'results.tex').open('xb') as stream:
        stream.write(constants)
    print(json.dumps({'phase': 'all-sixty-state-empirical-figures'}), flush=True)
    maps = load('_portable_original_empirical_maps', root / 'maps/render_trajectory_maps.py')
    before_argv = sys.argv
    try:
        sys.argv = [str(root / 'maps/render_trajectory_maps.py'), '--output', str(output / 'figures')]
        maps.main()
    finally:
        sys.argv = before_argv
    map_readout = read(root / 'inputs/map-readout.json')
    for name, binding in map_readout['outputs'].items():
        need(identity(output / 'figures' / name) == {k: binding[k] for k in ('bytes', 'sha256')},
             'Original all-state figure or point bytes differ: ' + name)
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = Path(module.__file__).resolve()
            need(path.is_relative_to(root / 'src/embed_optim'), 'Foreign project package imported')
            relative = path.relative_to(root).as_posix()
            need(relative in manifest['source_modules'], 'Unbound project module imported')
            loaded[name] = {**identity(path), 'local_role': relative}
    need(not torch.cuda.is_initialized(), 'Reconstruction used CUDA')
    require_same(authenticate(root, args.manifest_sha256), manifest)
    write(output / 'complete.json', {'scope': 'actual-closed-current-primary-numerical-graph-reconstruction',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': args.manifest_sha256,
        'all_ten_primary_outcome_tables_reconstructed': True,
        'all_nine_functional_tables_and_decisions_reconstructed': True,
        'original_nine_and_exact_five_weight_predictors_reconstructed': True,
        'all_84_weight_and_24_functional_controls_reconstructed': True,
        'all_17_original_publication_outputs_byte_exact': reproduced,
        'all_60_state_map_and_trajectory_outputs_byte_exact': map_readout['outputs'],
        'loaded_project_modules': loaded, 'fresh_native_model_admission_claimed': False,
        'original_native_admission_preserved_as_upstream': True,
        'geometry_svd_or_raw_coordinate_computation_repeated': False,
        'reencoding_or_gpu_access': False, 'physical_second_host': False,
        'factorial_evidence_reconstructed_here': False,
        'complete_document_compiled_here': False, 'source_release': False, 'full_goal_complete': False})
    print(json.dumps({'current_primary_graph_reconstructed': True, 'completion': identity(output / 'complete.json'),
                      'loaded_project_modules': len(loaded), 'full_goal_complete': False}), flush=True)


if __name__ == '__main__':
    main()
