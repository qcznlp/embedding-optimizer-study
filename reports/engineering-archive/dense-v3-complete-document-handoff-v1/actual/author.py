"""Source-bound current paper assembly, queued behind genuine complete outcomes.

This explicitly named consumer reuses the completed native primary assembly and
the completed original factorial collectors. It does not patch old admission
guards, rerun GPU experiments, install the authoritative paper, or declare release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual'
ASSEMBLY = ARCHIVE / 'scientific-publication-v3'
ASSEMBLY_SHA = '1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc'
MAPS = ARCHIVE / 'trajectory-maps-v2'
MAPS_SHA = 'c94d3c0bc986308b64ddf898d516c338942832635c0f16021d81f26d0e4cab4f'
TEMPLATE = ARCHIVE / 'paper-preview-v2'
TEMPLATE_SHA = '36a6b3509d1ad71a0c9609832e37869bde97379daf34bf69832af0f68f7f95fe'
SUMMARY = Path('/tmp/dense-v3-factorial-summary.BQ08HjeP')
SUMMARY_SHA = '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1'
SUMMARY_AUTH = '47485b8a16af1b0be419e67c4cf45ede08cdacbd01059e42357e16c97b9eb92c'
RENDERER_SHA = '83b7a92e43711ce55647aef8084bd507ae2ea8bec01d14b15754a45d15ec18d0'
SCOPE = 'owner_authorized_current_complete_dense_paper_consumer_v1'
OWNER = '你有权做一切事情，目标是尽快完成任务'
OUTPUT = HERE / 'actual-complete-paper'
TABLE_COUNTS = {'beir_seed_task_scores': 168, 'factorial_cell_summary': 4,
    'estimand_seed_task_contrasts': 126, 'estimand_summary': 3,
    'probe_checkpoint_metrics': 60, 'probe_task_metrics': 840}


def need(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path).absolute()
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Require an ordinary bound file: ' + str(path))
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
         ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Bound file changed while reading')
    return {'bytes': after.st_size, 'sha256': sha}


def bound(path, expected):
    actual = identity(path)
    need(actual['sha256'] == expected if isinstance(expected, str)
         else actual == {k: expected[k] for k in actual}, 'Bound content differs: ' + str(path))
    return actual


def read(path, expected=None):
    before = identity(path) if expected is None else bound(path, expected)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    bound(path, before)
    return result


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def copy_file(source, target, expected):
    bound(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream, Path(source).open('rb') as original:
        shutil.copyfileobj(original, stream)
    bound(source, expected)
    bound(target, expected)


def load(name, path, sha):
    bound(path, sha)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    bound(path, sha)
    return value


def component(sha):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Current paper consumer must hide CUDA')
    need(os.environ.get('PYTHONPATH') == str(STORY / 'src'), 'Use the explicit current scientific source namespace')
    return load('_current_complete_document', HERE / 'document_component.py', sha)


def factorial_source():
    for name, sha in {
        'dense_no_packing_state_operator_factorial_protocol.json':
            '5773943a3ae9b581021a0f7b85b162c74d5c497eeca1386578ff9d2c3bcafe76',
        'dense_no_packing_state_operator_factorial_implementation_protocol.json':
            '7573a28781735782ec1ec527cfd5d70b01a97ba14e081df77a91104a3a066d97',
        'dense_no_packing_state_operator_factorial_publication_protocol.json':
            '405166720a021e6173e66d2b549bccda3e0e67a35b69ffe50f4f0db043f5ba0d',
        'dense_no_packing_state_operator_claim_wording_amendment.json':
            '929fe6445598b84b546670eb5eefd408c5221cacad6cc8e93a984651dbbe5de7',
    }.items():
        bound(STORY / 'configs' / name, sha)
    result = load('_current_factorial_summary_source', SUMMARY / 'summarize.py', SUMMARY_SHA)
    result.authenticate(SimpleNamespace(source_sha256=SUMMARY_SHA, authorization_sha256=SUMMARY_AUTH))
    bound(STORY / 'src/embed_optim/state_operator_factorial_publication.py', RENDERER_SHA)
    from embed_optim import state_operator_factorial_publication as renderer
    need(Path(renderer.__file__).resolve() == STORY / 'src/embed_optim/state_operator_factorial_publication.py',
         'Historical factorial renderer imported from another namespace')
    return result, renderer


def primary_evidence(doc):
    """Reuse and join completed genuine native reads; no full SVD or encoder rerun."""
    bound(ARCHIVE / 'native_primary.py', '98d2cfaf76d212792906f4701f5c6014a4da69fec40fe41eedd096a717fbb6e7')
    sys.path.insert(0, str(ARCHIVE))
    assembly = load('_current_accepted_primary_assembly', ARCHIVE / 'assemble_publication.py',
                    'fe41ff895bb92d3a71523ea96fc4b8654ccc507d0f34c3f57a01c30a8aaaa674')
    receipt = read(ASSEMBLY / 'completed.json', ASSEMBLY_SHA)
    need(receipt['current_primary_scientific_result_assembly_complete'] is True
         and receipt['all_108_exploratory_comparisons_retained'] is True
         and receipt['original_gather_passed'] is False
         and receipt['original_single_selection_guard_passed'] is False,
         'Incomplete current primary assembly or historical admission relabelled')
    bindings = {str(ASSEMBLY / 'completed.json'): identity(ASSEMBLY / 'completed.json')}
    for name, expected in receipt['outputs'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Nonlocal primary output')
        bindings[str(ASSEMBLY / name)] = bound(ASSEMBLY / name, expected)
    values, native_bindings = assembly.native_inputs()
    bindings.update(native_bindings)
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_v3_publication_contract import PublicationContract
    from embed_optim.primary_v3_exact_publication_contract import ExactPublicationContract
    primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
    publication = PublicationContract.load(STORY / 'configs/dense_primary_v3_publication_protocol.json', primary)
    exact = ExactPublicationContract.load(STORY / 'configs/dense_primary_v3_exact_publication_protocol.json', primary)
    joins = assembly.join_native(primary, values)
    need(joins == receipt['full_native_primary_geometry_functional_joins'], 'Native joined population changed')
    constants = doc.original.render_constants(primary)
    publication.recheck()
    exact.recheck()
    # These are prior complete native proofs, not new model-payload reads.
    return constants, bindings, {'scope': 'completed_native_primary_result_reuse',
        'assembly': identity(ASSEMBLY / 'completed.json'), 'joins': joins,
        'primary_protocol_sha256': primary.sha256, 'publication_protocol_sha256': publication.sha256,
        'exact_publication_protocol_sha256': exact.sha256,
        'original_native_admission_reused': True, 'new_model_payload_reads': False,
        'new_svd_or_coordinate_computations': False, 'old_admission_guards_changed': False}


def fixed_paper_inputs(doc, constants_path):
    archive = read(ARCHIVE.parent / 'archive-manifest.json',
                   '82563d6a98625c96cb41bbfcc7c2239a01e0f4f0a6261f130268032837c7c630')
    bound(TEMPLATE / 'main.tex', TEMPLATE_SHA)
    maps = read(MAPS / 'completed.json', MAPS_SHA)
    need(maps['native_assembly']['sha256'] == ASSEMBLY_SHA
         and maps['displayed_states_per_figure'] == 60 and maps['rates_per_optimizer'] == 4
         and maps['points_not_fitted_projections'] is True, 'Incomplete real all-rate display')
    bound(ASSEMBLY / 'exact-publication/tables.json', maps['native_tables'])
    for name, expected in maps['outputs'].items():
        bound(MAPS / name, expected)
    result = {name: TEMPLATE / name for name in ('main.tex', 'references.bib',
        'vendor/acl.sty', 'vendor/acl_natbib.bst', 'figures/optimizer-weight-dimension-map.pdf')}
    for name in ('optimizer-primary.tex', 'dimension-utilization.tex'):
        result['generated/' + name] = ASSEMBLY / 'exact-publication' / name
    result['results.tex'] = constants_path
    result['generated/recipe-sensitivity.tex'] = ASSEMBLY / 'recipe-sensitivity.tex'
    for name in doc.EXTERNAL_FIGURES[1:]:
        result[name] = MAPS / Path(name).name
    need(set(result) == set(doc.BUILD_INPUTS) - {'generated/state-operator-factorial.tex'},
         'Incomplete fixed current manuscript sources')
    for path in result.values():
        if path != constants_path:
            bound(path, archive['files'][path.relative_to(ARCHIVE.parent).as_posix()])
    return {name: {'path': str(path), **identity(path)} for name, path in result.items()}


def imported_scientific_sources():
    result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and (name.startswith('embed_optim') or name.startswith('_current_')):
            resolved = Path(path).resolve()
            need(resolved.is_relative_to(STORY) or resolved.is_relative_to(HERE)
                 or resolved.is_relative_to(SUMMARY), 'Unexpected scientific module origin')
            result[str(resolved)] = identity(resolved)
    for path in (ARCHIVE / 'native_primary.py', ARCHIVE / 'native_analysis.py',
                 SUMMARY / 'collect.py', SUMMARY / 'authorization.json'):
        result[str(path)] = identity(path)
    return result


def authenticate(args):
    bound(__file__, args.source_sha256)
    auth = read(HERE / 'authorization.json', args.authorization_sha256)
    need(auth['scope'] == SCOPE and auth['owner_message'] == OWNER
         and auth['source'] == identity(__file__) and auth['output_root'] == str(OUTPUT)
         and auth['summary_source_sha256'] == SUMMARY_SHA
         and auth['summary_authorization_sha256'] == SUMMARY_AUTH,
         'Current consumer authority or upstream identity differs')
    doc = component(auth['component']['sha256'])
    for path, expected in auth['bound_inputs'].items():
        bound(path, expected)
    for value in auth['paper_inputs'].values():
        bound(value['path'], value)
    factorial_source()
    return auth, doc


def prepare(args):
    bound(__file__, args.source_sha256)
    need(not (HERE / 'authorization.json').exists() and not OUTPUT.exists(), 'Preserve previous consumer attempts')
    doc = component(args.component_sha256)
    tests = read(HERE / 'tests.json', args.tests_sha256)
    need(tests['component'] == identity(HERE / 'document_component.py')
         and tests['author'] == identity(__file__) and tests['source'] == identity(HERE / 'test_document.py')
         and tests['failures'] == tests['errors'] == tests['skipped'] == 0
         and tests['tests_run'] >= 12, 'Missing current complete-document controls')
    constants, bindings, proof = primary_evidence(doc)
    factorial_source()
    prepared = HERE / 'prepared'
    prepared.mkdir(exist_ok=False)
    with (prepared / 'constants.tex').open('xb') as stream:
        stream.write(constants)
    write(prepared / 'primary-native-reuse.json', proof)
    paper_inputs = fixed_paper_inputs(doc, prepared / 'constants.tex')
    bindings.update(imported_scientific_sources())
    for path in (prepared / 'primary-native-reuse.json', HERE / 'tests.json', HERE / 'test_document.py',
                 MAPS / 'completed.json', MAPS / 'all-sixty-state-points.json',
                 ARCHIVE.parent / 'archive-manifest.json', ARCHIVE / 'render_trajectory_maps.py'):
        bindings[str(path)] = identity(path)
    auth = {'scope': SCOPE, 'prepared_at_utc': now(), 'owner_message': OWNER,
        'source': identity(__file__), 'component': identity(HERE / 'document_component.py'),
        'bound_inputs': bindings, 'paper_inputs': paper_inputs,
        'summary_source_sha256': SUMMARY_SHA, 'summary_authorization_sha256': SUMMARY_AUTH,
        'required_factorial_tables': TABLE_COUNTS, 'output_root': str(OUTPUT),
        'primary_native_assembly_complete': True, 'wait_for_actual_factorial_outcomes': True,
        'old_guards_modified': False, 'gpu_access': False, 'manuscript_installation': False,
        'committed_source_release': False, 'full_goal_complete': False}
    write(HERE / 'authorization.json', auth)
    print(json.dumps({'authorization': identity(HERE / 'authorization.json'),
        'primary_native_assembly_complete': True, 'current_inputs_bound': len(bindings),
        'factorial_outcomes_complete': (SUMMARY / 'run/completed.json').exists(),
        'full_goal_complete': False}), flush=True)


def complete_factorial():
    """Admit only the actual native complete chain; rerender unchanged statistics."""
    summary, renderer = factorial_source()
    need(not (SUMMARY / 'run/failed.json').exists(), 'Actual inference failed')
    completed = read(SUMMARY / 'run/completed.json')
    need(completed['source_sha256'] == SUMMARY_SHA and completed['authorization_sha256'] == SUMMARY_AUTH,
         'Wrong completed inference source')
    root = summary.OUTPUT
    receipt = read(root / 'readout.json', completed['readout'])
    need(receipt['scope'] == summary.SCOPE and receipt['source'] == identity(SUMMARY / 'summarize.py')
         and receipt['authorization_sha256'] == SUMMARY_AUTH
         and receipt['original_statistical_source_sha256'] == summary.ORIGINAL_SHA
         and receipt['actual_collector_exits'] == {'beir': 0, 'probe': 0}
         and receipt['table_counts'] == TABLE_COUNTS and receipt['independent_arithmetic_verified'] is True,
         'Incomplete actual factorial inference')
    need(receipt['inference'] == {'samples': 100000, 'seed': 20260904,
                                 'intervals': 'three marginal linear percentile 95%'}, 'Changed inference population')
    bindings = {str(SUMMARY / 'run/completed.json'): identity(SUMMARY / 'run/completed.json'),
                str(root / 'readout.json'): identity(root / 'readout.json')}
    expected_names = {'tables.json', 'independent_verification.json', *(name + '.csv' for name in TABLE_COUNTS)}
    need(set(receipt['outputs']) == expected_names, 'Missing or additional summary outputs')
    for name, value in receipt['outputs'].items():
        bindings[str(root / name)] = bound(root / name, value)
    collectors = {}
    for kind, expected_scope, count in (
        ('beir', 'actual-complete-genuine-v3-factorial-beir-readback', 168),
        ('probe', 'actual-complete-genuine-v3-factorial-probe-readback', 60)):
        path = SUMMARY / 'run' / (kind + '.json')
        value = read(path, receipt['actual_native_collectors'][kind])
        need(value['scope'] == expected_scope and value['actual_exit_zero_workers'] == count
             and value['historical_factorial_guard_admission'] is False
             and value['model_encoding_repeated'] is False
             and value['collector_source']['sha256'] == summary.COLLECT_SHA,
             'Missing original native collector or relabelled historical admission')
        need(read(SUMMARY / 'run' / (kind + '.exited.json'))['exit_code'] == 0,
             'Actual native collector did not exit zero')
        pools = summary.c.require_complete_pair(kind)
        need(value['pools'] == pools, 'Native collector and completed pool identities differ')
        if kind == 'beir':
            need(value['fresh_original_task_reader'] is True, 'Missing complete final-settings native BEIR read')
        else:
            need(value['raw_arrays_and_metrics_reconstructed'] is True, 'Missing raw probe reconstruction')
        for entry in pools.values():
            bindings[entry['path']] = bound(entry['path'], entry['binding'])
        for item in (path, SUMMARY / 'run' / (kind + '.started.json'), SUMMARY / 'run' / (kind + '.exited.json')):
            bindings[str(item)] = identity(item)
        collectors[kind] = value
    _, tasks, training = summary.context()
    tables = read(root / 'tables.json', receipt['outputs']['tables.json'])
    rebuilt = summary.infer(collectors['beir'], collectors['probe'], tasks, training)
    from embed_optim.primary_contract import require_same
    require_same(tables, rebuilt)
    independent = summary.independent(rebuilt)
    require_same(independent, read(root / 'independent_verification.json', receipt['outputs']['independent_verification.json']))
    indexed = {row['estimand']: row for row in rebuilt['estimand_summary']}
    need(len(indexed) == 3 and set(indexed) == set(renderer.ESTIMANDS), 'Missing factorial estimand')
    latex = renderer._render_latex(indexed).encode('utf-8')
    for path, value in bindings.items():
        bound(path, value)
    return latex, bindings, {'scope': 'current_complete_factorial_paper_readout',
        'original_native_collectors_reused': True, 'original_numerics_reconstructed_exactly': True,
        'independent_arithmetic_repeated': independent, 'table_counts': TABLE_COUNTS,
        'all_three_estimands': [indexed[name] for name in renderer.ESTIMANDS],
        'interpretation': renderer._interpretation(indexed), 'renderer': identity(Path(renderer.__file__)),
        'historical_guard_admission_claimed': False, 'full_goal_complete': False}


def coordinate(args):
    auth, doc = authenticate(args)
    run = HERE / 'run'
    run.mkdir(exist_ok=False)
    # Read only our own process identity for an exact future observer.
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    write(run / 'started.json', {'pid': os.getpid(), 'start_ticks': int(stat[19]),
        'started_at_utc': now(), 'source_sha256': args.source_sha256,
        'authorization_sha256': args.authorization_sha256, 'gpu_access': False})
    try:
        while not (SUMMARY / 'run/completed.json').exists():
            need(not (SUMMARY / 'run/failed.json').exists(), 'Actual outcome inference failed; retain evidence')
            time.sleep(30)
        auth, doc = authenticate(args)
        factorial_tex, factor_bindings, factor_proof = complete_factorial()
        OUTPUT.mkdir(exist_ok=False)
        for name, item in auth['paper_inputs'].items():
            copy_file(item['path'], OUTPUT / 'paper' / name, item)
        target = OUTPUT / 'paper/generated/state-operator-factorial.tex'
        with target.open('xb') as stream:
            stream.write(factorial_tex)
        expected = {name: (OUTPUT / 'paper' / name).read_bytes() for name in doc.RESULT_FILES}
        constants = (OUTPUT / 'paper/results.tex').read_bytes()
        write(OUTPUT / 'factorial-readout.json', factor_proof)
        copy_file(HERE / 'prepared/primary-native-reuse.json', OUTPUT / 'primary-native-reuse.json',
                  auth['bound_inputs'][str(HERE / 'prepared/primary-native-reuse.json')])
        write(OUTPUT / 'scientific-input-bindings.json', {'primary': auth['bound_inputs'], 'factorial': factor_bindings})
        result = doc.compile_fresh(OUTPUT, expected, constants)
        authenticate(args)
        for path, value in factor_bindings.items():
            bound(path, value)
        write(OUTPUT / 'completed.json', {'scope': SCOPE, 'completed_at_utc': now(),
            'source': identity(__file__), 'authorization': identity(HERE / 'authorization.json'),
            'document': identity(OUTPUT / 'document.json'), 'pdf': identity(OUTPUT / 'paper/build/main.pdf'),
            'original_complete_primary_native_assembly_reused': True,
            'actual_complete_factorial_native_inference_reconstructed': True,
            'compiled_current_document_verified': result['compiled_pdf_verified'],
            'main_end_page': result['layout']['main_end_page'],
            'abstract_words': result['source_inspection']['actual_abstract']['words_conservative'],
            'visual_review_complete': False, 'portable_full_paper_reconstruction_complete': False,
            'authoritative_manuscript_installed': False, 'committed_source_release': False,
            'gpu_resume_verified_here': False, 'full_goal_complete': False})
        write(run / 'completed.json', {'completed_at_utc': now(), 'receipt': identity(OUTPUT / 'completed.json')})
        print(json.dumps({'document_complete': True, 'output': str(OUTPUT), 'full_goal_complete': False}), flush=True)
    except BaseException as error:
        write(run / 'failed.json', {'failed_at_utc': now(), 'exception_type': type(error).__name__,
            'message': str(error), 'old_guards_modified': False, 'full_goal_complete': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'coordinate', 'inspect'))
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--component-sha256')
    parser.add_argument('--tests-sha256')
    parser.add_argument('--authorization-sha256')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'coordinate':
        coordinate(args)
    else:
        authenticate(args)
        print(json.dumps({'source_bound': True, 'outcomes_complete': (SUMMARY / 'run/completed.json').exists(),
                          'full_goal_complete': False}), flush=True)


if __name__ == '__main__':
    main()
