"""Join genuine current native chains and run the unchanged scientific renderers.

This named consumer incorporates the accepted exact data-history amendment. It
never impersonates a passed original gather, stubs a PrimaryV3Contract method,
installs a manuscript, or claims that pending factorial/release work is complete.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

from native_primary import identity, need, write_new, event

STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
HERE = Path(__file__).resolve().parent
HELPER_SHA = '98d2cfaf76d212792906f4701f5c6014a4da69fec40fe41eedd096a717fbb6e7'
ANALYSIS_SHA = '918006092ef37a1ca48c80237c91071d1444f16eb58f71e78af1bae1a02608a3'
NATIVE = {
    'primary': ('dense_v3_current_complete_primary_view_history_readout_v1', '4915423f782a5ce952bc6c253f85568d691e435908542b93d2bf0154b4d26317'),
    'geometry': ('current_primary_geometry_native_chain_v1', '25796c12a69eb56db4e16c36a4bdc4d9fc7b16683de6014fe1cc72c99a3bf306'),
    'functional': ('current_primary_functional_native_chain_v1', 'c6a191f5601a083771783c6d72a5b44eab17b8a7161ba773384d963789073345'),
}
FRAGMENTS = STORY / 'reports/dense-v3-manuscript-results-v1'
FRAGMENTS_SHA = '4c8a94599e7710db3d25c45d41df5d3fb7dd92c784c209af581ddd8a4e10d29b'


def read(path, bound=None):
    before = identity(path)
    need(bound is None or (before['sha256'] == bound if isinstance(bound, str)
                           else before == {k: bound[k] for k in before}), 'External input binding differs')
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    need(identity(path) == before, 'Bound input raced')
    return value


def native_inputs():
    need(identity(HERE / 'native_primary.py')['sha256'] == HELPER_SHA
         and identity(HERE / 'native_analysis.py')['sha256'] == ANALYSIS_SHA, 'Native entry sources changed')
    values, bound = {}, {}
    for kind, (scope, sha) in NATIVE.items():
        root = HERE / ('native-' + kind)
        value = read(root / 'evidence.json', sha)
        completion = read(root / 'completed.json')
        need(not (root / 'failed.json').exists(), 'Native consumer failed')
        need(value['scope'] == scope and value['scientific_completion'] is False
             and completion['evidence'] == identity(root / 'evidence.json'), 'Incomplete native result')
        values[kind] = value
        for path in sorted(root.rglob('*')):
            if path.is_file():
                bound[str(path)] = identity(path)
    need(values['primary']['current_native_primary_admission_complete'] is True
         and values['primary']['original_single_selection_guard_passed'] is False,
         'Native primary admission missing or historical guard relabelled')
    return values, bound


def compare_previous_table(primary, name, current, previous):
    from embed_optim.primary_contract import require_same
    from embed_optim.primary_v3_outcomes import validate_scores
    if name == 'all_task_scores':
        # The trajectory exporter uses numeric recipe order; the original
        # outcome helper uses lexical run ID order. Compare the same complete
        # keyed population, not row positions. No score or other table changes.
        validate_scores(primary, current)
        validate_scores(primary, previous)
        previous = sorted(previous, key=lambda row: (row['run_id'], row['stage'], row['task']))
    require_same(current, previous)


def join_native(primary, values):
    from embed_optim.primary_contract import digest, require_same
    native = values['primary']
    runs = native['grid']['runs']
    indexed = {row['run_id']: row for row in runs}
    need(len(indexed) == len(runs) == 12, 'Incomplete native primary population')
    need(set(indexed) == {row['run_id'] for row in primary.inputs['runs']}, 'Primary run identities differ')
    geometry = values['geometry']['original_primary_admission']
    need(len(geometry['rows']) == 12
         and {row['run_id'] for row in geometry['rows']} == set(indexed),
         'Incomplete or duplicated geometry whole-run admission')
    bindings = {b['path']: b for b in native['original_completions']}
    for row in geometry['rows']:
        run = indexed[row['run_id']]
        require_same(row['expected'], primary.expected_identity(row['run_id']))
        require_same(row['binding'], bindings[row['binding']['path']])
        require_same({k: row['proof'][k] for k in run if k != 'run_id'},
                     {k: v for k, v in run.items() if k != 'run_id'})
    functional = values['functional']['admitted']
    need(set(functional['complete_runs']) == set(indexed), 'Functional whole-run population differs')
    for run_id, admitted in functional['complete_runs'].items():
        run = indexed[run_id]
        require_same({k: admitted[k] for k in run if k != 'run_id'},
                     {k: v for k, v in run.items() if k != 'run_id'})
    jobs = {(row['run_id'], row['step']): row for row in values['functional']['checkpoint_joins']}
    need(len(jobs) == len(values['functional']['checkpoint_joins']) == 60,
         'Incomplete or duplicated native functional checkpoint joins')
    for run_id, run in indexed.items():
        need(run['run_identity_sha256'] == digest(primary.expected_identity(run_id)), 'Wrong fresh run identity')
        for stage, checkpoint in enumerate(run['checkpoints'], 1):
            row = jobs[(run_id, checkpoint['step'])]
            model = next(item for item in checkpoint['files'] if item['path'] == 'model.safetensors')
            require_same(row['model_safetensors'], {k: model[k] for k in ('bytes', 'sha256')})
            require_same(row['checkpoint_seal'], checkpoint['checkpoint_seal'])
            need(row['run_identity_sha256'] == run['run_identity_sha256'] and row['stage'] == stage,
                 'Weight/functional/retrieval state identity differs')
    return {'whole_runs_joined': 12, 'saved_model_and_seal_joins': 60,
            'current_native_primary_admission_complete': True,
            'old_single_selection_guard_passed': False, 'scientific_completion': False}


def exploratory(primary_tables, functional_tables):
    """Retain already independently checked post-result controls, labelled as such."""
    bindings, result = {}, {}
    for kind, report, sha in (
        ('weight', 'dense-v3-predictor-sensitivity-v1', '00009a1f0253d9dfa6de9938f016f20f339f1e0729e6ce31886aca9a96a2db64'),
        ('functional', 'dense-v3-functional-sensitivity-v1', '8cb9b93da12a245e214429ffcf61f6922803c72c9c9ea496e572d24ecdc3e231')):
        root = STORY / 'reports' / report / 'actual'
        value = read(root / 'readout.json', sha)
        need(value['exploratory_post_result'] is True and value['scientific_completion'] is False,
             'Post-result analysis relabelled')
        bindings[str(root / 'readout.json')] = identity(root / 'readout.json')
        for path, bound in value['input_bindings'].items():
            need(identity(path) == bound, 'Exploratory parent differs')
            bindings[path] = bound
        for name, bound in value['outputs'].items():
            need(identity(root / name) == {k: bound[k] for k in ('bytes', 'sha256')}, 'Exploratory output differs')
            bindings[str(root / name)] = identity(root / name)
        result[kind] = read(root / 'result.json')
    from embed_optim.primary_contract import require_same
    require_same(primary_tables, read(STORY / 'reports/dense-v3-weight-retrieval-v1/actual/tables.json'))
    require_same(functional_tables, read(STORY / 'reports/dense-v3-functional-inference-v1/actual/tables.json'))
    # The existing renderer verifies every exact MSE/fold flag, with no model
    # admission claim. Only its pure helper is used in this project namespace.
    source = FRAGMENTS / 'render.py'
    need(identity(source)['sha256'] == '2c9c0a8b0580b35e25b8debf008e0e2e01ffbcc6eca3a2c01d838b9e1c3dcdea', 'Control renderer changed')
    spec = importlib.util.spec_from_file_location('original_complete_control_rendering', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.authenticate(FRAGMENTS / 'bundle')
    from embed_optim.corrected_publication import FEATURE_LABELS, _tex_escape
    from embed_optim.primary_v3_exact_publication_render import LABELS
    labels = {**FEATURE_LABELS, **{f: 'Exact: ' + label for f, label in LABELS.items()}}
    labels['full_spectrum_nonzero_saved_segment_entropy_rank_fraction'] = 'Full-spectrum segment entropy'
    weight = module.checked_controls(result['weight'], labels, 84)
    functional = module.checked_controls(result['functional'], module.FUNCTIONAL_LABELS, 24)
    rendered = module.controls_tex(weight, functional, labels, _tex_escape).encode()
    need(rendered == (FRAGMENTS / 'actual/recipe-sensitivity.tex').read_bytes(), 'Control LaTeX changed')
    return rendered, {'bindings': bindings, 'all_108_comparisons_retained': True,
        'weight_all_four': weight['survivors'], 'functional_all_four': functional['survivors'],
        'exploratory_post_result': True, 'inference_reused_from_independently_verified_result': True,
        'statistical_inference_repeated': False, 'scientific_completion': False}


def assemble(output):
    values, bindings = native_inputs()
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_v3_publication_contract import PublicationContract
    from embed_optim.primary_v3_exact_publication_contract import ExactPublicationContract
    from embed_optim import primary_v3_outcomes as outcomes
    from embed_optim import primary_v3_bridge as bridge
    from embed_optim import primary_v3_exact_bridge as exact_bridge
    from embed_optim import primary_v3_publication as publication
    from embed_optim import primary_v3_exact_publication as exact_publication
    from embed_optim.primary_contract import canonical, digest, require_same
    primary = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
    pub_contract = PublicationContract.load(STORY / 'configs/dense_primary_v3_publication_protocol.json', primary)
    exact_contract = ExactPublicationContract.load(STORY / 'configs/dense_primary_v3_exact_publication_protocol.json', primary)
    joins = join_native(primary, values)
    native = values['primary']
    selection, grid = native['validation_selection'], native['grid']
    outcome_tables = outcomes.outcome_tables(primary, grid['score_rows'], selection)
    outcome_tables['system_metrics'] = outcomes.system_rows(primary, grid['runs'])
    geometry = read(HERE / 'native-geometry/tables.json')
    functional = read(HERE / 'native-functional/tables.json')
    decisions = read(HERE / 'native-functional/decisions.json')
    approx = geometry['approximate']
    bridged, diagnostics = bridge.bridge_tables(primary, approx['checkpoint_geometry'],
        approx['run_pair_subspace_overlap'], outcome_tables['run_stage_scores'])
    exact, exact_diagnostics = exact_bridge.sensitivity_tables(primary, bridged,
        geometry['exact']['checkpoint_exact_geometry'], geometry['exact']['run_pair_exact_subspace_overlap'])
    # Fresh original inference must reproduce every earlier primary score and
    # every original/exact predictor; no output is adopted from a edited digest.
    previous = STORY / 'reports/dense-v3-complete-trajectories-v1/tables'
    for name in ('all_task_scores', 'primary_task_effects', 'primary_summary',
                 'secondary_task_effects', 'secondary_summary', 'run_stage_scores',
                 'optimizer_stage_scores', 'run_observed_auc'):
        compare_previous_table(primary, name, outcome_tables[name], exact_bridge.typed_csv(previous / (name + '.csv')))
    require_same(bridged['bridge_rows'], read(HERE / 'native-functional/original_panel.json'))
    controls_tex, controls = exploratory({'original': bridged, 'exact': exact}, functional)
    event('all_original_scientific_tables_reproduced', outcomes=10, weight_predictors=14, functional_tables=9)
    outcome_evidence = {'grid': grid, 'validation_selection': selection,
                        'current_native_admission': native}
    bridge_evidence = {'outcome_evidence': outcome_evidence,
        'current_native_geometry': values['geometry'], 'numerical_diagnostics': diagnostics}
    functional_evidence = {'original_bridge_evidence': bridge_evidence,
        'current_native_functional': values['functional'], 'decisions': decisions}
    evidence = {'scope': 'current_native_primary_publication_consumer_v1',
        'functional_evidence': functional_evidence, 'joins': joins,
        'source_bindings': bindings, 'exploratory_controls': controls,
        'original_single_selection_guard_passed': False, 'committed_source_release': False,
        'manuscript_installed': False, 'scientific_completion': False}
    tables = {'outcomes': outcome_tables, 'geometry': approx, 'bridge': bridged, 'functional': functional}
    contents = publication.generate(primary, tables, selection, grid['runs'], decisions, evidence)
    complete_evidence = {'scope': 'current_native_primary_exact_publication_consumer_v1',
        'original_publication_evidence': evidence, 'exact_bridge_diagnostics': exact_diagnostics,
        'source_bindings': bindings, 'manuscript_installed': False, 'scientific_completion': False}
    exact_contents = exact_publication.generate(primary, contents, geometry['exact'], exact, complete_evidence)
    fragments = read(FRAGMENTS / 'actual/verification.json', FRAGMENTS_SHA)
    matched = []
    for name in ('original-optimizer-primary.tex', 'optimizer-primary.tex', 'exact-sensitivity.tex',
                 'functional-inference.tex', 'dimension-utilization.tex'):
        need(identity(FRAGMENTS / 'actual' / name) == fragments['outputs'][name], 'Accepted fragment changed')
        need(exact_contents[name] == (FRAGMENTS / 'actual' / name).read_bytes(), 'Native scientific result differs from numerical preview: ' + name)
        matched.append(name)
    summary = json.loads(contents['primary-summary.json'])
    previous_summary = read(FRAGMENTS / 'actual/primary-summary.json', fragments['outputs']['primary-summary.json'])
    require_same({key: summary[key] for key in previous_summary}, previous_summary)
    for path, bound in bindings.items():
        need(identity(path) == bound, 'Current native source/evidence changed before output')
    pub_contract.recheck()
    exact_contract.recheck()
    publication.reject_simulation(evidence)
    publication.reject_simulation(complete_evidence)
    plan = {'scope': evidence['scope'], 'publication_protocol_sha256': pub_contract.sha256,
        'primary_protocol_sha256': primary.sha256, 'evidence_sha256': digest(evidence),
        'current_native_primary_admission_complete': True,
        'original_single_selection_guard_passed': False, 'scientific_completion': False,
        'manuscript_installed': False, 'final_portable_publication_verified': False}
    exact_plan = {**plan, 'scope': complete_evidence['scope'],
        'exact_publication_protocol_sha256': exact_contract.sha256,
        'evidence_sha256': digest(complete_evidence)}
    publication.save(output / 'original-publication', plan, contents)
    exact_publication.save(output / 'exact-publication', exact_plan, exact_contents)
    with (output / 'recipe-sensitivity.tex').open('xb') as stream:
        stream.write(controls_tex)
    write_new(output / 'exploratory-controls.json', controls)
    # Expose the full original tables, not only rounded display values.
    write_new(output / 'outcomes.json', outcome_tables)
    write_new(output / 'bridge-diagnostics.json', {'original': diagnostics, 'exact': exact_diagnostics})
    write_new(output / 'completed.json', {'scope': 'current_native_primary_scientific_result_assembly_v1',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'source': identity(__file__),
        'native_inputs': {kind: {'scope': scope, 'evidence_sha256': sha} for kind, (scope, sha) in NATIVE.items()},
        'full_native_primary_geometry_functional_joins': joins,
        'primary_table_counts': {k: len(v) for k, v in outcome_tables.items()},
        'original_publication_outputs': len(contents), 'exact_publication_outputs': len(exact_contents),
        'native_tex_matches_previous_numeric_fragments': matched,
        'all_108_exploratory_comparisons_retained': True,
        'current_primary_scientific_result_assembly_complete': True,
        'original_gather_passed': False, 'original_single_selection_guard_passed': False,
        'model_svds_and_coordinate_computations_reused_with_native_provenance': True,
        'manuscript_installed': False, 'factorial_outcomes_complete': False,
        'committed_source_release': False, 'scientific_completion': False,
        'outputs': {str(p.relative_to(output)): identity(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    event('current_primary_scientific_result_assembly_complete', **identity(output / 'completed.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only current scientific consumer')
    need(identity(__file__)['sha256'] == args.source_sha, 'Consumer source changed')
    need(args.output.is_absolute() and not args.output.exists()
         and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'Use a new ordinary output')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / 'started.json', {'pid': os.getpid(), 'source': identity(__file__),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'scientific_completion': False})
    try:
        assemble(args.output)
        need(identity(__file__)['sha256'] == args.source_sha, 'Consumer source raced')
    except Exception as error:
        write_new(args.output / 'failed.json', {'exception_type': type(error).__name__,
            'message': str(error), 'old_guards_modified': False, 'scientific_completion': False})
        raise
