"""Wheel-local restoration readback and real CPU backward integration only."""
import json
import os
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
wheel=here/'wheel'
denied=(
    '/root/embedding-optimizer-story-refactor', '/root/embedding-optimizer-primary-v3',
    '/root/embedding-optimizer-v3-experiment', '/root/embedding-optimizer-study',
    '/tmp/dense-v3-warm-reducer-recovery.rP4NVV4i',
    '/tmp/dense-v3-paired-backward-fixed.JXOJBril',
    '/tmp/dense-v3-resume-device-recovery.uQ0ynb0k',
)
def forbidden(path):
    return any(path==root or path.startswith(root+'/') for root in denied)
attempts=[]
controls=True
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.path.abspath(os.fsdecode(args[0]))
        if forbidden(path):
            attempts.append({'path':path,'negative_control':controls})
            raise PermissionError('Producer-root access refused')
    if event in ('socket.connect','socket.bind','socket.getaddrinfo'):
        raise PermissionError('Cold restoration check forbids network')
sys.path=[str(wheel),str(here/'wheel-tests'),*[p for p in sys.path if not forbidden(os.path.abspath(p))]]
sys.addaudithook(audit)
for root in denied:
    try:
        open(root+'/forbidden-negative-control','rb')
    except PermissionError:
        pass
    else:
        raise AssertionError('Missing original-root refusal')
controls=False
from embed_optim_restore.reference import load_reference
references=[]
for case in ('adamw','muon'):
    for rank in range(4):
        ref=load_reference(case,rank)
        references.append({k:ref[k] for k in ('case','rank','component_sha256','cold_sha256','warm_sha256')})
assert 'torch' not in sys.modules
from embed_optim_restore import entry
import pytest
tests=here/'wheel-tests/test_restore_entry.py'
code=pytest.main(['-q','--rootdir='+str(here/'wheel-tests'),'-c','/dev/null',
    str(tests)+'::test_real_native_cpu_backward_repeats_once_then_delegates_clipping',
    str(tests)+'::test_gradient_gate_failure_never_calls_clipping_and_restores_handlers',
    str(tests)+'::test_early_exit_without_backward_does_not_report_pass',
    str(tests)+'::test_no_live_group_refused_before_installing_callbacks',
    str(tests)+'::test_symlink_checkpoint_refused_before_accessing_trainer',
    '--junitxml='+str(here/'wheel-tests.xml')])
assert code==0
import torch
assert not torch.cuda.is_initialized()
assert not any(n=='embed_optim' or n.startswith('embed_optim.') for n in sys.modules)
modules={n:str(Path(m.__file__).absolute()) for n,m in sys.modules.items()
         if (n=='embed_optim_restore' or n.startswith('embed_optim_restore.')) and getattr(m,'__file__',None)}
assert modules and all(Path(p).is_relative_to(wheel) for p in modules.values())
assert not any(not v['negative_control'] for v in attempts)
with (here/'cold-result.json').open('x') as stream:
    json.dump({'references':references,'modules':modules,'negative_controls':attempts,
        'actual_forbidden_reads':0,'actual_cpu_integration_tests':5,
        'GPU_initialized':False,'original_training_package_imported':False,
        'physical_second_host':False,'new_GPU_resume':False},stream,indent=2,sort_keys=True)
