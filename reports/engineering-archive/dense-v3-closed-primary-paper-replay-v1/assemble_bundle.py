"""Build a closed actual-primary numerical/paper source bundle, not a release.

The genuine native assembly is an authenticated upstream input. Its historical
failures and unknown exits are preserved. This copies source and data; it never
restarts a producer or rewrites a source/publication contract.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual'
ASSEMBLY = ARCHIVE / 'scientific-publication-v3'
DOC = Path('/tmp/dense-v3-document-integration.Xz1qvTME')
ASSEMBLY_SHA = '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc'
AUTH_SHA = 'ca7a0503b9324d51ce62a9c19cd1ec380e06311a0ff498b84ea30e683670e0e4'


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary input required')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input raced')
    return {'bytes': after.st_size, 'sha256': sha}


def bound(path, expected):
    actual = identity(path)
    need(actual['sha256'] == expected if isinstance(expected, str)
         else actual == {k: expected[k] for k in actual}, 'Upstream content changed: ' + str(path))
    return actual


def read(path, expected=None):
    before = identity(path) if expected is None else bound(path, expected)
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant')
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    bound(path, before)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
         and os.environ.get('PYTHONPATH') == str(STORY / 'src'), 'Explicit CPU scientific namespace required')
    need(args.output.is_absolute() and not args.output.exists(), 'Preserve old bundles; use a new output')
    archive = read(ARCHIVE.parent / 'archive-manifest.json',
                   '82563d6a98625c96cb41bbfcc7c2239a01e0f4f0a6261f130268032837c7c630')
    receipt = read(ASSEMBLY / 'completed.json', ASSEMBLY_SHA)
    need(receipt['current_primary_scientific_result_assembly_complete'] is True
         and receipt['original_gather_passed'] is False
         and receipt['original_single_selection_guard_passed'] is False, 'Genuine upstream scope differs')
    auth = read(DOC / 'authorization.json', AUTH_SHA)
    bound(DOC / 'author.py', 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d')
    bound(DOC / 'document_component.py', 'dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b')
    for path, value in auth['bound_inputs'].items():
        bound(path, value)
    native = read(ARCHIVE / 'native-primary/evidence.json',
                  '4915423f782a5ce952bc6c253f85568d691e435908542b93d2bf0154b4d26317')
    geometry_native = read(ARCHIVE / 'native-geometry/evidence.json',
                          '25796c12a69eb56db4e16c36a4bdc4d9fc7b16683de6014fe1cc72c99a3bf306')
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_contract import digest, require_same
    # Read actual recipe/source identity; this is not a new model-payload audit.
    primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
    expected = {r['run_id']: primary.expected_identity(r['run_id']) for r in primary.inputs['runs']}
    prior = {r['run_id']: r['expected'] for r in geometry_native['original_primary_admission']['rows']}
    require_same(expected, prior)
    need(len(expected) == 12 and len(native['grid']['runs']) == 12, 'Incomplete native recipe population')
    for run in native['grid']['runs']:
        need(run['run_identity_sha256'] == digest(expected[run['run_id']]), 'Whole-run recipe identity differs')
    # Import only the original pure numerical/display components. Their imports
    # are copied verbatim; no source tree or contract is altered to relocate.
    from embed_optim import (primary_v3_outcomes, primary_v3_bridge, primary_v3_exact_bridge,
        dimension_inference, primary_v3_publication, primary_v3_exact_publication,
        primary_v3_manuscript)
    load('_closed_source_inventory_document', DOC / 'document_component.py')
    copied_sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = Path(module.__file__).resolve()
            need(path.is_relative_to(STORY / 'src/embed_optim') and path.suffix == '.py', 'Foreign package import')
            if str(path) in auth['bound_inputs']:
                bound(path, auth['bound_inputs'][str(path)])
            copied_sources[path.relative_to(STORY).as_posix()] = path
    need('src/embed_optim/__init__.py' in copied_sources, 'Package initializer missing')
    args.output.mkdir(exist_ok=False)
    files = {}

    def copy(name, original, expected_binding=None):
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and name not in files, 'Nonlocal/duplicate bundle role')
        before = identity(original) if expected_binding is None else bound(original, expected_binding)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output, Path(original).open('rb') as source:
            shutil.copyfileobj(source, output)
        bound(target, before)
        bound(original, before)
        files[name] = {**before, 'origin': str(original)}

    def native_copy(name, original):
        relative = Path(original).relative_to(ARCHIVE.parent).as_posix()
        copy(name, original, archive['files'][relative])

    for name, original in copied_sources.items():
        copy(name, original)
    copy('source/document_component.py', DOC / 'document_component.py', auth['component'])
    for name in ('replay.py', 'run_isolated.py'):
        copy(name, HERE / name)
    for name in ('native-primary/evidence.json', 'native-geometry/tables.json',
                 'native-functional/decisions.json', 'native-functional/tables.json'):
        native_copy('inputs/' + name, ARCHIVE / name)
    for name in ('completed.json', 'outcomes.json', 'original-publication/evidence.json',
                 'exact-publication/evidence.json', 'exact-publication/tables.json'):
        native_copy('inputs/scientific-publication-v3/' + name, ASSEMBLY / name)
    # All current scientific output hashes are expected outputs, not replacements
    # for numerical reproduction or the complete original native admission chain.
    copy('provenance/document-authorization.json', DOC / 'authorization.json', AUTH_SHA)
    functional = STORY / 'reports/dense-v3-functional-inference-v1'
    functional_receipt = read(functional / 'actual/readout.json',
                              '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e')
    for name in ('original_panel.json', 'scientific_protocol.json', 'tables.json', 'decisions.json',
                 'features/checkpoint_summary.csv', 'features/task_summary.csv',
                 'features/random_removal.csv', 'features/rotation_summary.csv'):
        copy('inputs/functional/' + name, functional / 'actual' / name, functional_receipt['payloads'][name])
    copy('provenance/functional-readout.json', functional / 'actual/readout.json',
         '608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e')
    for kind, report, source, sha in (
        ('weight', 'dense-v3-predictor-sensitivity-v1', 'sensitivity.py',
         '6a87b441917c95d1fb0355c340beb8715acee89e66c102b51ecdfce6265d9141'),
        ('functional', 'dense-v3-functional-sensitivity-v1', 'functional_sensitivity.py',
         '0e98fbf4c768ef10bcae2cf6c660e37c583c88863055229ccdd65c0109661013')):
        root = STORY / 'reports' / report
        original = read(root / 'actual/readout.json',
                        '00009a1f0253d9dfa6de9938f016f20f339f1e0729e6ce31886aca9a96a2db64'
                        if kind == 'weight' else '8cb9b93da12a245e214429ffcf61f6922803c72c9c9ea496e572d24ecdc3e231')
        copy('source/' + kind + '_controls.py', root / 'source' / source, sha)
        copy('inputs/' + kind + '_controls.json', root / 'actual/result.json', original['outputs']['result.json'])
    fragments = STORY / 'reports/dense-v3-manuscript-results-v1'
    copy('source/control_rendering.py', fragments / 'render.py',
         '2c9c0a8b0580b35e25b8debf008e0e2e01ffbcc6eca3a2c01d838b9e1c3dcdea')
    maps = ARCHIVE / 'trajectory-maps-v2'
    map_receipt = read(maps / 'completed.json',
                       'c94d3c0bc986308b64ddf898d516c338942832635c0f16021d81f26d0e4cab4f')
    native_copy('maps/render_trajectory_maps.py', ARCHIVE / 'render_trajectory_maps.py')
    for name in ('completed.json', 'exact-publication/tables.json'):
        native_copy('maps/scientific-publication-v3/' + name, ASSEMBLY / name)
    copy('inputs/map-readout.json', maps / 'completed.json', identity(maps / 'completed.json'))
    for name, original in auth['paper_inputs'].items():
        copy('paper-inputs/' + name, Path(original['path']), original)
    # This metadata object intentionally has no PrimaryV3Contract admission API.
    catalog = {'scope': 'saved-genuine-primary-recipe-metadata-only', 'primary_protocol_sha256': primary.sha256,
        'payload': primary.payload, 'inputs': primary.inputs, 'expected_identities': expected,
        'upstream_native_assembly_sha256': ASSEMBLY_SHA, 'model_admission_performed_here': False}
    write(args.output / 'inputs/recipe-catalog.json', catalog)
    files['inputs/recipe-catalog.json'] = {**identity(args.output / 'inputs/recipe-catalog.json'),
        'origin': 'generated from genuine PrimaryV3Contract and matched to complete native identities'}
    for path, value in auth['bound_inputs'].items():
        bound(path, value)
    result = {'scope': 'closed-current-primary-numerical-and-document-inputs-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'builder': identity(Path(__file__)),
        'owner_authority': '你有权做一切事情，目标是尽快完成任务',
        'files': files, 'source_modules': sorted(copied_sources),
        'native_primary_assembly_sha256': ASSEMBLY_SHA,
        'document_authorization_sha256': AUTH_SHA, 'model_encoding_or_svd_repeated': False,
        'genuine_native_admission_is_upstream_not_relabelled': True,
        'original_unknown_exit_codes_preserved': True,
        'old_source_or_guard_modified': False, 'current_factorial_complete': False,
        'full_paper_portability_complete': False, 'source_release': False, 'full_goal_complete': False}
    write(args.output / 'manifest.json', result)
    print(json.dumps({'bundle': str(args.output), 'manifest': identity(args.output / 'manifest.json'),
        'source_modules': len(copied_sources), 'files': len(files),
        'bytes': sum(row['bytes'] for row in files.values()), 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
