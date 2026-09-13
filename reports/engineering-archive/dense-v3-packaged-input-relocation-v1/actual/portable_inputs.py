"""Read genuine sealed inputs using the unchanged extracted distribution reader.

CPU-only relocation evidence. No GPU work, native-guard replacement, publication,
or claim that the unchanged full distribution gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

HERE = Path(__file__).resolve().parent
STORY = Path('/root/embedding-optimizer-story-refactor')
PRIMARY = Path('/root/embedding-optimizer-primary-v3')
EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
OLD_DATA = Path('/root/embedding-optimizer-study/data')
EVIDENCE = 'reports/engineering-archive/dense-v3-factorial-inputs-v1'
WHEEL = STORY / ('reports/engineering-archive/dense-v3-distribution-recovery-surface-v1/'
                 'expanded/embedding_optimizer_study-0.1.0-py3-none-any.whl')
WHEEL_SHA = '51a65e250a75b5e54f459c12cd930b7bf8cd87dc2c0928889091928be4e37f6e'
AUDITS = {
    'data-first.json': '94ea1b27631cf9270f445e6406e79c0c470bd346564a4057cee1f742923d103f',
    'states-first.json': '9958ecc4f23d9165838632c6bd7e24f1e64e3753eb5632206f2228411179e932',
    'verification.json': '9e92896fd14855ea1635f5571e824fc38c9071297e7fa3b4b848a55db08aa526',
}
CODE = {
    'embed_optim/factorial_v3_inputs.py': '372db4abf1e7136f118a42916e07146d20dc09d5c5de868d3c09f0951aa237f7',
    'embed_optim/primary_contract.py': 'e49f55d5d01408de001b8e1d56244fbcaf189de45c0209caffa0b4e34f828e19',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Ordinary file and nonsymlinked parents required')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k)
             for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
         'Input changed during its content read')
    return {'bytes': after.st_size, 'sha256': digest}


def relative(value):
    path = PurePosixPath(value)
    need(value and not path.is_absolute() and '..' not in path.parts
         and '\\' not in value and path.as_posix() == value and value != '.',
         'Noncanonical or escaping relative path')
    return Path(value)


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def roots():
    wheel = HERE / 'wheel'
    roles = {
        'repository': wheel / 'embedding_optimizer_study-0.1.0.data/data/share/embedding-optimizer-study',
        'primary_source': HERE / 'payload/primary-source',
        'experiment': HERE / 'payload/experiment',
        'data_store': HERE / 'payload/data-store',
        'evidence': HERE / 'payload/evidence',
    }
    return wheel, roles


def mapped(recorded, roles):
    original = Path(recorded)
    for old, role in ((STORY / EVIDENCE, 'evidence'), (STORY, 'repository'),
                      (PRIMARY, 'primary_source'), (EXPERIMENT, 'experiment'),
                      (OLD_DATA, 'data_store')):
        if original.is_relative_to(old):
            return role, roles[role] / relative(original.relative_to(old).as_posix())
    raise ValueError('Recorded file is outside the explicit study input roles')


def child():
    wheel, roles = roots()
    output = HERE / 'native-readback.json'
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    need(Path.cwd() == HERE / 'empty-cwd', 'Fresh working directory required')
    for member, expected in CODE.items():
        need(identity(wheel / member)['sha256'] == expected, 'Original packaged reader changed')
    forbidden = (STORY, PRIMARY, EXPERIMENT, OLD_DATA.parent)
    opened = set()
    denied = []
    controls = []

    def guard(event, args):
        if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'}:
            denied.append({'event': event, 'kind': 'network_or_spawn'})
            raise PermissionError('Reader cannot use network or spawn a process')
        if event in {'os.remove', 'os.rmdir', 'os.rename', 'os.symlink', 'os.link', 'os.mkdir', 'os.chmod', 'os.truncate'}:
            denied.append({'event': event, 'kind': 'filesystem_mutation'})
            raise PermissionError('Reader cannot mutate input files or directories')
        if event not in {'open', 'os.listdir', 'os.scandir'} or not args or isinstance(args[0], int):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        if any(path == old or path.is_relative_to(old) for old in forbidden):
            denied.append({'event': event, 'kind': 'old_producer_location', 'path': str(path)})
            raise PermissionError('Reader cannot access an original producer directory')
        if event == 'open':
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if write and path != output:
                denied.append({'event': event, 'kind': 'unexpected_write', 'path': str(path)})
                raise PermissionError('Only the new readback receipt may be written')
            if path.is_relative_to(HERE):
                opened.add(path.relative_to(HERE).as_posix())

    sys.addaudithook(guard)
    # These controls are not scientific data or parent-admission substitutes.
    for path, mode in ((STORY / 'configs/formal_runtime.json', 'rb'),
                       (roles['data_store'] / 'must-not-be-created', 'wb')):
        try:
            with path.open(mode):
                pass
        except PermissionError:
            controls.append({'path': str(path), 'mode': mode, 'refused_before_open': True})
        else:
            raise ValueError('Reader isolation control did not refuse the operation')
    control_denials = list(denied)
    denied.clear()
    sys.path.insert(0, str(wheel))
    from embed_optim import factorial_v3_inputs as native
    need(native.AUDITS == AUDITS, 'The genuine original audit anchors changed')
    locations = native.Locations(**roles)
    inputs = native.load_inputs(locations)
    need(inputs['scientific_admission'] is False and inputs['execution_authorized'] is False,
         'Read-only input scope changed')
    need(inputs['branch']['rows'] == 50000 and inputs['calibration']['rows'] == 32,
         'Wrong genuine input populations')
    need(not denied, 'Original producer, network, spawn or mutation access attempted')
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name == 'embed_optim' or name.startswith('embed_optim.'):
            path = Path(module.__file__).absolute()
            need(path.is_relative_to(wheel), 'Project import escaped the exact wheel')
            modules[name] = {'path': str(path), **identity(path)}
    result = {
        'completed_at_utc': now(), 'status': 'complete_native_input_readback',
        'scope': 'unchanged-wheel genuine input relocation; not full distribution or source-release admission',
        'wheel_sha256': WHEEL_SHA,
        'source_sha256': identity(HERE / 'portable_inputs.py')['sha256'],
        'accepted_original_audits': inputs['accepted_audit_digests'],
        'role_locations': {k: str(v) for k, v in roles.items()},
        'branch_rows': inputs['branch']['rows'], 'calibration_rows': inputs['calibration']['rows'],
        'sources': {k: {field: row[field] for field in
                    ('run_id', 'checkpoint', 'checkpoint_step', 'native_checkpoint', 'immutable_remote_commit')}
                    for k, row in inputs['sources'].items()},
        'readback_digest_path_dependent': native.digest(inputs),
        'loaded_project_modules': modules, 'opened_local_files': sorted(opened),
        'native_denied_operations': denied, 'separate_isolation_controls': controls,
        'separate_control_denials': control_denials,
        'gpu_access': False, 'network_access': False, 'source_or_input_mutation': False,
        'original_distribution_audit_passed': False, 'physical_second_host': False,
        'isolation_boundary': 'Python open/directory-enumeration/network/spawn/mutation audit hooks; not OS sandbox or syscall trace',
        'full_goal_complete': False,
    }
    dump(output, result)
    print(json.dumps({'event': 'native_input_readback_complete', 'branch_rows': result['branch_rows'],
                      'calibration_rows': result['calibration_rows'], 'source_states': len(result['sources']),
                      'project_modules': len(modules), 'denied_native_operations': len(denied)}), flush=True)


def prepare():
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    wheel, roles = roots()
    wheel_id = identity(WHEEL)
    need(wheel_id['sha256'] == WHEEL_SHA, 'The actual expanded distribution changed')
    wheel.mkdir()
    with zipfile.ZipFile(WHEEL) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'Duplicate archive member')
        for info in archive.infolist():
            relative(info.filename.rstrip('/'))
            need(not stat.S_ISLNK(info.external_attr >> 16), 'Symlink archive member')
        archive.extractall(wheel)
    for member, expected in CODE.items():
        need(identity(wheel / member)['sha256'] == expected, 'Original packaged source differs')
    for role, root in roles.items():
        if role != 'repository':
            root.mkdir(parents=True)
    files = {}

    def add(source, expected):
        source = Path(source)
        role, target = mapped(source, roles)
        expected = {k: expected[k] for k in ('bytes', 'sha256')}
        old = files.get(str(source))
        row = {'source': str(source), 'target': str(target), 'role': role, **expected}
        need(old is None or old == row, 'Conflicting input binding')
        files[str(source)] = row

    for name, expected_sha in AUDITS.items():
        source = STORY / EVIDENCE / name
        ident = identity(source)
        need(ident['sha256'] == expected_sha, 'Genuine original audit changed')
        add(source, ident)
    data = json.loads((STORY / EVIDENCE / 'data-first.json').read_text())
    states = json.loads((STORY / EVIDENCE / 'states-first.json').read_text())
    verified = json.loads((STORY / EVIDENCE / 'verification.json').read_text())
    for source, binding in [*data['inputs'].items(), *verified['files'].items()]:
        add(source, binding)
    for state in states['states']:
        checkpoint = Path(state['checkpoint'])
        for file in state['native_checkpoint']['files']:
            add(checkpoint / relative(file['path']), file)
        for key in ('original_durability_receipt', 'original_remote_audit_receipt'):
            add(state[key]['path'], state[key])
    inventory = {
        'created_at_utc': now(), 'wheel': {'path': str(WHEEL), **wheel_id},
        'source_sha256': identity(HERE / 'portable_inputs.py')['sha256'],
        'files': [files[k] for k in sorted(files)],
        'logical_input_bytes': sum(row['bytes'] for row in files.values()),
        'copy_boundary': 'genuine native input payloads only; packaged repository assets must already exist and match the wheel',
    }
    dump(HERE / 'copy-inventory.json', inventory)
    print(json.dumps({'event': 'inventory_complete', 'files': len(files),
                      'logical_input_bytes': inventory['logical_input_bytes']}), flush=True)
    for i, row in enumerate(inventory['files'], 1):
        source, target = Path(row['source']), Path(row['target'])
        expected = {k: row[k] for k in ('bytes', 'sha256')}
        need(identity(source) == expected, 'Original native input binding differs')
        if row['role'] == 'repository':
            need(identity(target) == expected, 'Actual wheel is missing/differs on a required repository asset')
        else:
            need(not target.exists(), 'Refuse overwriting an existing copied input')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            need(identity(target) == expected, 'Copied payload content differs')
        if i % 20 == 0 or i == len(files):
            print(json.dumps({'event': 'inputs_copied_and_verified', 'files': i}), flush=True)
    (HERE / 'empty-cwd').mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = ['/usr/bin/python', '-B', '-I', str(HERE / 'portable_inputs.py'), '--child']
    with (HERE / 'native-reader.log').open('x') as log:
        process = subprocess.Popen(command, cwd=HERE / 'empty-cwd', env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   close_fds=True)
        dump(HERE / 'native-reader-started.json', {'started_at_utc': now(), 'pid': process.pid,
             'command': command, 'source_sha256': inventory['source_sha256'], 'gpu_access': False})
        code = process.wait()
    dump(HERE / 'native-reader-exited.json', {'exited_at_utc': now(), 'exit_code': code})
    need(code == 0, 'Actual native reader failed; preserve log and stop')
    # Recheck original inputs and the copied payloads after the genuine readback.
    for row in inventory['files']:
        expected = {k: row[k] for k in ('bytes', 'sha256')}
        need(identity(row['source']) == expected and identity(row['target']) == expected,
             'Source or copied input changed after the readback')
    need(identity(WHEEL) == wheel_id, 'Original distribution changed')
    dump(HERE / 'completed.json', {'completed_at_utc': now(), 'native_reader_exit_code': code,
         'input_files': len(files), 'logical_input_bytes': inventory['logical_input_bytes'],
         'inventory': identity(HERE / 'copy-inventory.json'),
         'native_readback': identity(HERE / 'native-readback.json'),
         'source_sha256': inventory['source_sha256'], 'all_original_and_copied_inputs_rechecked': True,
         'original_distribution_audit_passed': False, 'source_release': False,
         'gpu_access': False, 'full_goal_complete': False})
    print(json.dumps({'event': 'complete', 'native_reader_exit_code': code,
                      'input_files': len(files), 'logical_input_bytes': inventory['logical_input_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    try:
        child() if args.child else prepare()
    except BaseException as error:
        if not args.child:
            failed = HERE / 'failed.json'
            if not failed.exists():
                dump(failed, {'failed_at_utc': now(), 'type': type(error).__name__,
                              'message': str(error), 'gpu_access': False,
                              'full_goal_complete': False})
        raise
