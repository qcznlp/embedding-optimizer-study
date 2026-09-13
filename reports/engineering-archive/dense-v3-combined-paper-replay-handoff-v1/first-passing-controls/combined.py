"""CPU-only complete-paper portability after the actual native paper finishes.

Original numerical functions, validators, independent arithmetic and current
document component are unchanged. Native model admission is upstream evidence;
the metadata accessor below has no model/experiment admission API.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
DOC = Path('/tmp/dense-v3-document-integration.Xz1qvTME')
SUMMARY = Path('/tmp/dense-v3-factorial-summary.BQ08HjeP')
INFERENCE = Path('/root/embedding-optimizer-v3-experiment/analyses/dense-v3-factorial-inference-v1')
TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
PRIMARY_ARCHIVE = STORY / 'reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1'
PRIMARY = PRIMARY_ARCHIVE / 'closed-primary'
PRIMARY_SHA = '88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d'
DOC_SHA = 'cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d'
DOC_AUTH = 'ca7a0503b9324d51ce62a9c19cd1ec380e06311a0ff498b84ea30e683670e0e4'
SUMMARY_SHA = '4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1'
SUMMARY_AUTH = '47485b8a16af1b0be419e67c4cf45ede08cdacbd01059e42357e16c97b9eb92c'
ORIGINAL_SHA = 'd6d4d8481404e38b0e7ab9291680c39a99909fc3fb87959bc42be49f4361cc27'
TRAIN_AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
COMPONENT_SHA = 'dbed8e2d4d418ce89d5d56d75da4e433e9b9b0888194a9c44ca053dd03d2684b'
ESTIMANDS = ('weight_state_effect', 'operator_effect', 'state_operator_interaction')
COUNTS = {'beir_seed_task_scores': 168, 'factorial_cell_summary': 4,
          'estimand_seed_task_contrasts': 126, 'estimand_summary': 3,
          'probe_checkpoint_metrics': 60, 'probe_task_metrics': 840}
SCOPE = 'actual-combined-dense-paper-numerical-portability-v1'
PINS = {DOC / 'author.py': DOC_SHA, DOC / 'authorization.json': DOC_AUTH,
        DOC / 'document_component.py': COMPONENT_SHA,
        SUMMARY / 'summarize.py': SUMMARY_SHA, SUMMARY / 'authorization.json': SUMMARY_AUTH,
        STORY / 'src/embed_optim/state_operator_factorial_summary.py': ORIGINAL_SHA,
        TRAIN / 'authorization.json': TRAIN_AUTH,
        PRIMARY / 'manifest.json': PRIMARY_SHA,
        PRIMARY_ARCHIVE / 'actual-primary/reconstructed/complete.json':
        'cc980ed44ed7a3681a35d8d40f84d8d14bf0c42a17996fae75c0e060a773e41a'}


def need(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Ordinary bound file required')
    before = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, key) == getattr(after, key) for key in
             ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'Input raced')
    return {'bytes': after.st_size, 'sha256': sha}


def bound(path, expected):
    actual = identity(path)
    need(actual['sha256'] == expected if isinstance(expected, str)
         else actual == {k: expected[k] for k in actual}, 'Bound content differs: ' + str(path))
    return actual


def read(path, expected=None):
    before = identity(path) if expected is None else bound(path, expected)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant')
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    bound(path, before)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def copy(source, target, expected):
    bound(source, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as out, Path(source).open('rb') as original:
        shutil.copyfileobj(original, out)
    bound(source, expected)
    return bound(target, expected)


def local_name(name):
    path = Path(name)
    need(bool(name) and not path.is_absolute() and '..' not in path.parts
         and path.as_posix() == name, 'Nonlocal bundle role')
    return name


def inventory(root, sha, scope):
    manifest = read(root / 'manifest.json', sha)
    need(manifest['scope'] == scope, 'Wrong closure scope')
    for name, value in manifest['files'].items():
        bound(root / local_name(name), value)
    paths = list(root.rglob('*'))
    need(not any(p.is_symlink() for p in paths), 'Symlink in bundle')
    observed = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    need(observed == set(manifest['files']) | {'manifest.json'}, 'Closed inventory differs')
    return manifest


def parent_inputs():
    for path, sha in PINS.items():
        bound(path, sha)
    auth = read(DOC / 'authorization.json', DOC_AUTH)
    need(auth['scope'] == 'owner_authorized_current_complete_dense_paper_consumer_v1'
         and auth['wait_for_actual_factorial_outcomes'] is True, 'Wrong native document authority')
    for path, expected in auth['bound_inputs'].items():
        bound(path, expected)
    inventory(PRIMARY, PRIMARY_SHA, 'closed-current-primary-numerical-and-document-inputs-v1')
    return auth


def admit_completed_document():
    """Authenticate actual completed original readers, never create native success."""
    auth = parent_inputs()
    need(not (DOC / 'run/failed.json').exists() and not (SUMMARY / 'run/failed.json').exists(), 'Upstream failed')
    terminal = read(DOC / 'run/completed.json')
    root = DOC / 'actual-complete-paper'
    completed = read(root / 'completed.json', terminal['receipt'])
    need(completed['scope'] == auth['scope'] and completed['source']['sha256'] == DOC_SHA
         and completed['authorization']['sha256'] == DOC_AUTH
         and completed['actual_complete_factorial_native_inference_reconstructed'] is True
         and completed['original_complete_primary_native_assembly_reused'] is True
         and completed['compiled_current_document_verified'] is True, 'Incomplete real native paper')
    document = read(root / 'document.json', completed['document'])
    bound(root / 'paper/build/main.pdf', completed['pdf'])
    need(document['compiled_pdf_verified'] is True and document['layout']['main_end_page'] <= 8
         and document['source_inspection']['actual_abstract']['words_conservative'] <= 200,
         'Native strict paper gates not passed')
    for name, expected in document['artifacts'].items():
        bound(root / local_name(name), expected)
    scientific = read(root / 'scientific-input-bindings.json')
    need(scientific['primary'] == auth['bound_inputs'], 'Native primary bindings differ')
    for path, expected in scientific['factorial'].items():
        bound(path, expected)
    summary_end = read(SUMMARY / 'run/completed.json')
    need(summary_end['source_sha256'] == SUMMARY_SHA
         and summary_end['authorization_sha256'] == SUMMARY_AUTH, 'Wrong native inference terminal')
    inferred = read(INFERENCE / 'readout.json', summary_end['readout'])
    need(inferred['actual_collector_exits'] == {'beir': 0, 'probe': 0}
         and inferred['table_counts'] == COUNTS and inferred['independent_arithmetic_verified'] is True
         and inferred['original_statistical_source_sha256'] == ORIGINAL_SHA
         and inferred['inference'] == {'samples': 100000, 'seed': 20260904,
                                      'intervals': 'three marginal linear percentile 95%'},
         'Incomplete original inference')
    need(set(inferred['outputs']) == {'tables.json', 'independent_verification.json',
                                    *(name + '.csv' for name in COUNTS)}, 'Wrong summary inventory')
    for name, expected in inferred['outputs'].items():
        bound(INFERENCE / name, expected)
    for kind, count in (('beir', 168), ('probe', 60)):
        value = read(SUMMARY / 'run' / (kind + '.json'), inferred['actual_native_collectors'][kind])
        need(value['scope'] == 'actual-complete-genuine-v3-factorial-' + kind + '-readback'
             and value['actual_exit_zero_workers'] == count
             and value['historical_factorial_guard_admission'] is False
             and value['model_encoding_repeated'] is False
             and read(SUMMARY / 'run' / (kind + '.exited.json'))['exit_code'] == 0,
             'Incomplete actual native collector')
    return root, document, inferred, scientific


def build_bundle(target):
    root, document, inferred, scientific = admit_completed_document()
    target.mkdir(exist_ok=False)
    files = {}
    def add(name, source, expected=None):
        value = identity(source) if expected is None else expected
        files[local_name(name)] = {**copy(source, target / name, value), 'origin': str(source)}
    primary = read(PRIMARY / 'manifest.json', PRIMARY_SHA)
    for name, value in primary['files'].items():
        add('primary/' + name, PRIMARY / name, value)
    add('primary/manifest.json', PRIMARY / 'manifest.json', PRIMARY_SHA)
    add('replay_complete.py', Path(__file__))
    add('source/original_summary.py', STORY / 'src/embed_optim/state_operator_factorial_summary.py', ORIGINAL_SHA)
    add('source/native_summary.py', SUMMARY / 'summarize.py', SUMMARY_SHA)
    add('factorial/training-authority.json', TRAIN / 'authorization.json', TRAIN_AUTH)
    add('factorial/summary-authority.json', SUMMARY / 'authorization.json', SUMMARY_AUTH)
    add('factorial/readout.json', INFERENCE / 'readout.json')
    for name, value in inferred['outputs'].items():
        add('factorial/' + name, INFERENCE / name, value)
    for name in ('completed.json', 'beir.json', 'probe.json', 'beir.started.json',
                 'beir.exited.json', 'probe.started.json', 'probe.exited.json'):
        add('factorial/native/' + name, SUMMARY / 'run' / name)
    for name in ('completed.json', 'document.json', 'scientific-input-bindings.json',
                 'factorial-readout.json', 'primary-native-reuse.json'):
        add('native-document/' + name, root / name)
    for name, expected in document['source_inspection']['inputs'].items():
        add('expected-paper/' + name, root / 'paper' / name, expected)
    for index, (path, value) in enumerate(sorted(scientific['factorial'].items())):
        add(f'native-factorial-bindings/{index:03d}.json', Path(path), value)
    text = subprocess.run(['pdftotext', '-layout', str(root / 'paper/build/main.pdf'), '-'],
                          capture_output=True, check=True, timeout=30).stdout
    target_text = target / 'native-document/pdf-text.txt'
    with target_text.open('xb') as stream:
        stream.write(text)
    files['native-document/pdf-text.txt'] = {**identity(target_text), 'origin': 'pdftotext -layout of actual strict native PDF'}
    # Check upstream again after every copy, without repeating the native computations.
    admit_completed_document()
    manifest = {'scope': SCOPE, 'created_at_utc': now(), 'files': files,
                'native_primary_manifest_sha256': PRIMARY_SHA,
                'native_document_completion': identity(root / 'completed.json'),
                'native_factorial_readout': identity(INFERENCE / 'readout.json'),
                'full_goal_complete': False, 'fresh_native_model_admission': False,
                'source_release': False, 'physical_second_host': False}
    write(target / 'manifest.json', manifest)
    return identity(target / 'manifest.json')


class MetadataAccess:
    """Only saved queue metadata and real assertions, not native admission APIs."""
    need = staticmethod(need)
    bound = staticmethod(bound)

    def __init__(self, queues):
        self.saved_queues = queues

    def queues(self):
        return self.saved_queues


def factorial_functions(bundle, training):
    import numpy as np
    original = bundle / 'source/original_summary.py'
    source = bundle / 'source/native_summary.py'
    bound(original, ORIGINAL_SHA)
    bound(source, SUMMARY_SHA)
    names = {'numerical_functions', 'validate_scores', 'validate_probe', 'infer', 'independent'}
    nodes = [n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names and len(nodes) == 5, 'Original numerical adapter functions missing')
    namespace = dict(ast=ast, np=np, Any=Any, defaultdict=defaultdict, statistics=statistics,
                     math=math, Fraction=Fraction, ORIGINAL=original, ORIGINAL_SHA=ORIGINAL_SHA,
                     ESTIMANDS=ESTIMANDS, e=MetadataAccess(training['queues']))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def replay_numerical_and_paper(bundle, output, sha):
    import csv
    import numpy as np
    manifest = inventory(bundle, sha, SCOPE)
    primary = bundle / 'primary'
    # The existing full primary replay is now a component of the actual combined
    # paper reconstruction, not a newly missing or changed primary experiment.
    command = [sys.executable, '-B', str(primary / 'run_isolated.py'), '--manifest-sha256', PRIMARY_SHA,
               '--output', str(output / 'primary')]
    with (output / 'primary-replay.log').open('x') as log:
        result = subprocess.run(command, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    write(output / 'primary-exited.json', {'exit_code': result.returncode, 'command': command})
    need(result.returncode == 0, 'Combined primary numerical component failed')
    rebuilt = output / 'primary/reconstructed'
    io = read(output / 'primary/io-boundary.json')
    need(io['failure'] is None and not io['producer_reads_refused'] and io['network_refused'] == 0,
         'Primary component attempted a producer fallback')
    sys.path.insert(0, str(primary / 'src'))
    from embed_optim.primary_contract import require_same
    from embed_optim import state_operator_factorial_publication as renderer
    need(Path(renderer.__file__).resolve() == primary / 'src/embed_optim/state_operator_factorial_publication.py',
         'Renderer imported from another namespace')
    training = read(bundle / 'factorial/training-authority.json', TRAIN_AUTH)
    authority = read(bundle / 'factorial/summary-authority.json', SUMMARY_AUTH)
    tasks = authority['tasks']
    need(set(training['queues']) == {'a', 'b'} and all(len(x) == 6 for x in training['queues'].values()), 'Wrong queue population')
    expected = {(s, o, z) for s in ('adamw_state', 'muon_state') for o in ('adamw', 'muon')
                for z in (314159, 271828, 161803)}
    runs = [r for cell in training['queues'].values() for r in cell]
    observed = {(training['requests'][r]['state'], training['requests'][r]['operator'],
                 training['requests'][r]['seed']) for r in runs}
    need(len(set(runs)) == 12 and observed == expected and len(set(tasks)) == 14, 'Wrong complete factorial population')
    require_same(tasks, read(primary / 'inputs/recipe-catalog.json')['payload']['evaluation']['tasks'])
    functions = factorial_functions(bundle, training)
    tables = functions['infer'](read(bundle / 'factorial/native/beir.json'),
                               read(bundle / 'factorial/native/probe.json'), tasks, training)
    require_same(tables, read(bundle / 'factorial/tables.json'))
    need({k: len(v) for k, v in tables.items()} == COUNTS, 'Complete table counts differ')
    verification = functions['independent'](tables)
    require_same(verification, read(bundle / 'factorial/independent_verification.json'))
    write(output / 'factorial/tables.json', tables)
    write(output / 'factorial/independent_verification.json', verification)
    for name, rows in tables.items():
        path = output / 'factorial' / (name + '.csv')
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        bound(path, identity(bundle / 'factorial' / path.name))
    latex = renderer._render_latex({r['estimand']: r for r in tables['estimand_summary']}).encode('utf-8')
    expected_factorial = bundle / 'expected-paper/generated/state-operator-factorial.tex'
    need(latex == expected_factorial.read_bytes(), 'Original factorial rendering differs')
    spec = importlib.util.spec_from_file_location('_combined_original_document', primary / 'source/document_component.py')
    doc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doc)
    paper_root = output / 'document'
    native = read(bundle / 'native-document/document.json')
    generated = {'generated/optimizer-primary.tex': rebuilt / 'exact-publication/optimizer-primary.tex',
                 'generated/dimension-utilization.tex': rebuilt / 'exact-publication/dimension-utilization.tex',
                 'generated/recipe-sensitivity.tex': rebuilt / 'recipe-sensitivity.tex',
                 'results.tex': rebuilt / 'results.tex',
                 'figures/weight-to-retrieval-map.pdf': rebuilt / 'figures/weight-to-retrieval-map.pdf',
                 'figures/full-rate-retrieval-trajectories.pdf': rebuilt / 'figures/full-rate-retrieval-trajectories.pdf'}
    for name in doc.BUILD_INPUTS:
        target = paper_root / 'paper' / name
        expected_path = bundle / 'expected-paper' / name
        if name == 'generated/state-operator-factorial.tex':
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(latex)
        else:
            copy(generated.get(name, expected_path), target, identity(expected_path))
        bound(target, native['source_inspection']['inputs'][name])
    includes = {name: (paper_root / 'paper' / name).read_bytes() for name in doc.RESULT_FILES}
    result = doc.compile_fresh(paper_root, includes, (paper_root / 'paper/results.tex').read_bytes())
    for key in ('source_inspection', 'layout', 'font_count', 'pdf_pages'):
        require_same(result[key], native[key])
    pdf = paper_root / 'paper/build/main.pdf'
    text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, check=True, timeout=30).stdout
    need(text == (bundle / 'native-document/pdf-text.txt').read_bytes(), 'Fresh complete PDF text differs')
    with (paper_root / 'pdf-text.txt').open('xb') as stream:
        stream.write(text)
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = Path(module.__file__).resolve()
            need(path.is_relative_to(primary / 'src'), 'Original project fallback import')
            loaded[name] = {'role': path.relative_to(bundle).as_posix(), **identity(path)}
    inventory(bundle, sha, SCOPE)
    write(output / 'complete.json', {'scope': SCOPE, 'completed_at_utc': now(),
        'input_manifest_sha256': sha, 'primary_completion': identity(rebuilt / 'complete.json'),
        'factorial_counts': COUNTS, 'factorial_independent_verification': verification,
        'complete_paper_source_and_extracted_text_exact': True, 'strict_document': identity(paper_root / 'document.json'),
        'pdf': identity(pdf), 'loaded_project_modules': loaded, 'all_original_scientific_rules_unchanged': True,
        'native_model_admission_is_upstream_provenance': True, 'visual_review_complete': False,
        'physical_second_host': False, 'authoritative_paper_installed': False,
        'gpu_resume_verified_here': False, 'source_release': False, 'full_goal_complete': False})


def isolated_replay(args):
    bundle, output = args.bundle, args.output
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONPATH', '') == '', 'Clean CPU-only process required')
    need(bundle.is_absolute() and output.is_absolute() and not output.exists()
         and not output.is_relative_to(bundle) and not any(p.is_symlink() for p in (output, *output.parents)),
         'Use a new separate ordinary absolute attempt')
    inventory(bundle, args.manifest_sha256, SCOPE)
    output.mkdir(exist_ok=False)
    runtime = output / 'runtime'
    for name, key in (('matplotlib', 'MPLCONFIGDIR'), ('cache', 'XDG_CACHE_HOME'), ('tmp', 'TMPDIR')):
        path = runtime / name
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    sys.path[:] = [p for p in sys.path if not p.startswith(('/root/embedding-optimizer', '/tmp/dense-'))]
    observations = {'producer_reads_refused': [], 'network_refused': 0}
    def audit(event, values):
        if event == 'socket.connect':
            observations['network_refused'] += 1
            raise PermissionError('No network in complete numerical replay')
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).absolute().resolve(strict=False)
            if not path.is_relative_to(bundle) and not path.is_relative_to(output) and str(path).startswith(
                    ('/root/embedding-optimizer', '/tmp/dense-')):
                observations['producer_reads_refused'].append(str(path))
                raise PermissionError('No original producer fallback')
    sys.addaudithook(audit)
    failure = None
    try:
        replay_numerical_and_paper(bundle, output, args.manifest_sha256)
        need(not observations['producer_reads_refused'] and observations['network_refused'] == 0, 'Observed refused fallback')
    except BaseException as error:
        failure = {'exception_type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        write(output / 'io-boundary.json', {**observations, 'failure': failure, 'observed_at_utc': now(),
              'python_io_boundary_only': True, 'os_sandbox_claimed': False, 'full_goal_complete': False})


def authenticate(args):
    bound(__file__, args.source_sha256)
    auth = read(HERE / 'authorization.json', args.authorization_sha256)
    need(auth['scope'] == SCOPE and auth['source'] == identity(__file__)
         and auth['output_root'] == str(HERE / 'actual') and auth['owner_message'] == '你有权做一切事情，目标是尽快完成任务',
         'Wrong portability handoff')
    for path, expected in auth['inputs'].items():
        bound(path, expected)
    return auth


def prepare(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'No GPU access')
    bound(__file__, args.source_sha256)
    parent_inputs()
    tests = read(HERE / 'tests.json', args.tests_sha256)
    need(tests['source'] == identity(__file__) and tests['failures'] == tests['errors'] == tests['skips'] == 0,
         'Operational controls did not pass')
    inputs = {str(path): identity(path) for path in PINS}
    inputs[str(HERE / 'test_combined.py')] = identity(HERE / 'test_combined.py')
    inputs[str(HERE / 'tests.json')] = identity(HERE / 'tests.json')
    write(HERE / 'authorization.json', {'scope': SCOPE, 'prepared_at_utc': now(), 'source': identity(__file__),
          'owner_message': '你有权做一切事情，目标是尽快完成任务', 'inputs': inputs,
          'output_root': str(HERE / 'actual'), 'wait_for_original_native_document': True,
          'new_gpu_work': False, 'authoritative_paper_installation': False, 'source_release': False,
          'full_goal_complete': False})
    print(json.dumps({'authorization': identity(HERE / 'authorization.json'), 'full_goal_complete': False}), flush=True)


def coordinate(args):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'No GPU access')
    authenticate(args)
    run = HERE / 'run'
    run.mkdir(exist_ok=False)
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    write(run / 'started.json', {'pid': os.getpid(), 'start_ticks': int(stat[19]), 'started_at_utc': now(),
          'source_sha256': args.source_sha256, 'authorization_sha256': args.authorization_sha256, 'gpu_access': False})
    try:
        while not (DOC / 'run/completed.json').exists():
            need(not (DOC / 'run/failed.json').exists() and not (SUMMARY / 'run/failed.json').exists(), 'Upstream failed; retain evidence')
            time.sleep(30)
        authenticate(args)
        bundle = HERE / 'closed'
        manifest = build_bundle(bundle)
        write(run / 'bundle.json', {'manifest': manifest, 'bundle': str(bundle), 'created_at_utc': now()})
        command = ['/usr/bin/python', '-B', str(bundle / 'replay_complete.py'), 'replay', '--bundle', str(bundle),
                   '--manifest-sha256', manifest['sha256'], '--output', str(HERE / 'actual')]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with (run / 'replay.log').open('x') as log:
            child = subprocess.Popen(command, env=env, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            child_stat = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
            write(run / 'replay.started.json', {'pid': child.pid, 'start_ticks': int(child_stat[19]),
                  'command': command, 'started_at_utc': now(), 'gpu_access': False})
            code = child.wait()
        write(run / 'replay.exited.json', {'exit_code': code, 'completed_at_utc': now()})
        need(code == 0, 'Real combined numerical/PDF replay failed; retain attempt')
        io = read(HERE / 'actual/io-boundary.json')
        need(io['failure'] is None and not io['producer_reads_refused'] and io['network_refused'] == 0, 'Replay I/O boundary failed')
        authenticate(args)
        write(run / 'completed.json', {'completion': identity(HERE / 'actual/complete.json'),
              'input_manifest': manifest, 'completed_at_utc': now(), 'full_goal_complete': False})
        print(json.dumps({'combined_replay_complete': True, 'full_goal_complete': False}), flush=True)
    except BaseException as error:
        write(run / 'failed.json', {'exception_type': type(error).__name__, 'message': str(error),
              'failed_at_utc': now(), 'full_goal_complete': False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'coordinate', 'replay', 'inspect'))
    p.add_argument('--source-sha256')
    p.add_argument('--authorization-sha256')
    p.add_argument('--tests-sha256')
    p.add_argument('--manifest-sha256')
    p.add_argument('--bundle', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'coordinate':
        coordinate(args)
    elif args.action == 'replay':
        isolated_replay(args)
    else:
        authenticate(args)
        print(json.dumps({'upstream_complete': (DOC / 'run/completed.json').exists(), 'full_goal_complete': False}))


if __name__ == '__main__':
    main()
