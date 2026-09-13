"""Full coordinator/feature-body simulation. All process/model dependencies are stubs."""
from __future__ import annotations
import argparse
import ast
import builtins
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import flow_candidate as flow
import record_layout as layout

WORK=Path(__file__).parents[1]
OLD=Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions')
flow.bound(OLD/'dispatch.py',{'sha256':flow.DISPATCH_SHA})
flow.bound(OLD/'inputs.json',{'sha256':'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067'})
INPUTS=json.loads((OLD/'inputs.json').read_bytes())
CELLS=tuple(j['plan']['state']['cell'] for j in INPUTS['jobs'])
layout.validate_cells(CELLS)
TABLE_COUNTS={k:61 for k in ('checkpoint_summary','task_summary','random_removal','rotation_summary')}
OBSERVATIONS=[]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_new(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')


def same(a,b):
    if a!=b:raise ValueError('Synthetic dependency check differs')


def csv_bytes(rows):
    stream=io.StringIO(newline=''); w=csv.DictWriter(stream,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows);return stream.getvalue().encode()


class Fixture:
    def __init__(self,root,fail=None):
        self.root=root;self.fail=fail;self.events=[];self.encoded=[];self.feature_calls=[];self.children=[]
        self.entry=SimpleNamespace(STORY=root,OUTPUT=root/'output',SCOPE='synthetic_full_flow_fixture_only',
                                  VALIDATION=root/'validation/validation.py',verify_state_inputs=lambda *_:None)
        validation=root/'validation/run';(validation/'jobs').mkdir(parents=True)
        write_new(validation/'completed.json',{'all_twelve_validations_verified':True,'scientific_completion':False})
        for run in layout.RUNS:write_new(validation/'jobs'/f'{run}.scored.json',{'synthetic_fixture':True})
        jobs=[{'plan':{'state':copy.deepcopy(j['plan']['state'])},'checkpoint':root/'unread-models'/j['plan']['state']['cell']}
              for j in INPUTS['jobs']]
        self.context=SimpleNamespace(jobs=jobs,admitted={'complete_runs':list(layout.RUNS),'synthetic_fixture_only':True},
            identities=[{'source':'synthetic-task'}],dataset=(),digest=digest,
            contract=SimpleNamespace(scientific={'synthetic_only':True}),
            geometry=SimpleNamespace(identity=flow.identity,write_new=write_new,now=lambda:'synthetic-fixture-time'))
        self.entry.authenticate=lambda *_:(self.context,{'synthetic_fixture_only':True})
        owner=self

        class Lease:
            def __enter__(self):owner.events.append('lease-enter');return (31,32)
            def __exit__(self,*_):owner.events.append('lease-close')
        self.context.parent=SimpleNamespace(handoff=lambda:owner.events.append('handoff'),
            leases=lambda tokens:Lease(), environment=lambda tokens:{'CUDA_VISIBLE_DEVICES':','.join(tokens or [])})

        def saved(path):
            manifest=json.loads((path/'manifest.json').read_bytes())
            return {'manifest':{'path':str(path/'manifest.json'),**flow.identity(path/'manifest.json')},
                    'output':manifest['output'],'array_metadata':{'synthetic_fixture_only':True},
                    'model_encoding_repeated':False,'scientific_completion':False}

        def save(path,plan,*_):
            path.mkdir(parents=True,exist_ok=False)
            with (path/'vectors.npz').open('xb') as f:f.write(b'NOT_AN_NPZ: synthetic model stub\n')
            write_new(path/'manifest.json',{'cell':plan['state']['cell'],'plan':plan,
                'output':{'path':'vectors.npz',**flow.identity(path/'vectors.npz')},'synthetic_fixture_only':True})
            return saved(path)

        def inspect(path,plan,*_,expected_manifest_sha256):
            flow.bound(path/'manifest.json',{'sha256':expected_manifest_sha256})
            manifest=json.loads((path/'manifest.json').read_bytes());same(manifest['plan'],plan)
            if owner.fail=='copied-native-readback' and path==owner.entry.OUTPUT/'vectors/states/pretrained':
                raise ValueError('Synthetic copied native readback failure')
            return {'synthetic_cell':plan['state']['cell']},saved(path)

        def encode(path,*_):
            cell=next(j['plan']['state']['cell'] for j in jobs if j['checkpoint']==path)
            owner.encoded.append(cell)
            return {'synthetic_cell':cell},{'synthetic_fixture_only':True}

        self.context.vectors=SimpleNamespace(encode_state=encode,save_vectors=save,inspect_vectors=inspect)
        def matrix(path,admitted,sha):
            flow.bound(path/'manifest.json',{'sha256':sha});m=json.loads((path/'manifest.json').read_bytes())
            same(list(m['states']),list(CELLS));return m

        def iterate(path,manifest,population,*_):
            owner.events.append('iter-vectors')
            for index,job in enumerate(population):
                if owner.fail=='incomplete-finalizer' and index==60:return
                cell=job['plan']['state']['cell']; item=manifest['states'][cell]
                arrays,checked=inspect(path/'states'/cell,job['plan'],expected_manifest_sha256=item['sha256'])
                yield job,arrays,checked
        self.context.exports=SimpleNamespace(_manifest=matrix,iter_vectors=iterate,recheck_contract=lambda _:None)

        def compute(meta,arrays,_):
            cell=arrays['synthetic_cell'];owner.feature_calls.append(cell)
            if owner.fail=='feature-kernel' and cell==CELLS[2]:raise ValueError('Synthetic feature kernel failure')
            return {'tables':{key:[{'cell':cell,'synthetic_value':1}] for key in TABLE_COUNTS},'synthetic_fixture_only':True}

        def feature_save(path,plan,result):
            path.mkdir(parents=True,exist_ok=False);write_new(path/'manifest.json',{'plan':plan,'result':result})

        def feature_inspect(path,plan,result):
            same(json.loads((path/'manifest.json').read_bytes()),{'plan':plan,'result':result})
            if owner.fail=='feature-readback':raise ValueError('Synthetic full feature readback failure')
        self.context.kernels=SimpleNamespace(compute_state=compute)
        self.context.features=SimpleNamespace(feature_plan=lambda *a:{'synthetic_feature_plan':True},
            check_result=lambda *a:None,state_plan=lambda plan,job,item:{'cell':job['plan']['state']['cell'],'vector':item},
            _feature_manifest=lambda path,*a:json.loads((path/'manifest.json').read_bytes()))
        self.context.feature_io=SimpleNamespace(save_state=feature_save,inspect_state=feature_inspect)
        self.args=SimpleNamespace(source_sha='a'*64,authorization_sha='b'*64,run_root=root/'run')
        self.origin={'source_sha256':'c'*64,'authorization_sha256':'d'*64,
                     'records':str(root/'prior/jobs'),'vectors':str(root/'prior/vectors/pretrained')}
        prior=Path(self.origin['records']);prior.mkdir(parents=True)
        priorvectors=Path(self.origin['vectors']);old_saved=save(priorvectors,jobs[0]['plan'])
        handle={'pid':-2,'ppid':-3,'start_ticks':-2,'command':['SYNTHETIC_OLD_WORKER'],
                'cell':'pretrained','authorization_sha256':'d'*64}
        write_new(prior/'pretrained.started.json',{**handle,'source_sha256':'c'*64})
        write_new(prior/'pretrained.exited.json',{**handle,'exit_code':0})
        write_new(prior/'pretrained.encoded.json',{'cell':'pretrained','source_sha256':'c'*64,
            'authorization_sha256':'d'*64,'plan_sha256':digest(jobs[0]['plan']),'saved':old_saved})
        write_new(prior/'pretrained.verified.json',{'cell':'pretrained','actual_exit_code':0,'native_readback_passed':True,
            'worker_receipt':{'path':str(prior/'pretrained.encoded.json'),**flow.identity(prior/'pretrained.encoded.json')},'saved':old_saved})
        self.origin['files']={n:flow.identity(prior/('pretrained.'+n)) for n in
                             ('started.json','exited.json','encoded.json','verified.json')}
        self.origin['files'].update({n:flow.identity(priorvectors/n) for n in ('manifest.json','vectors.npz')})
        self.origin_before=copy.deepcopy(self.origin['files'])

        class Child:
            def __init__(self,command,**kwargs):
                self.command=command;self.pid=-10-len(owner.children);owner.children.append(command)
                self.features='--features' in command
                if self.features:
                    same(kwargs['close_fds'],True);same(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                else:same(kwargs['pass_fds'],(31,32))
            def wait(self):
                if not self.features and owner.fail=='worker-exit':return 7
                if self.features and owner.fail=='feature-exit':return 8
                arguments=owner.ns['parse_args'](self.command[3:])
                try:owner.ns['feature_worker' if self.features else 'worker'](arguments)
                except Exception as e:
                    owner.events.append('synthetic-child-failure:'+str(e));return 9
                return 0

        modules={'embed_optim.primary_contract':SimpleNamespace(require_same=same),
                 'embed_optim.primary_v3_dimension_contract':SimpleNamespace(TABLE_COUNTS=TABLE_COUNTS),
                 'embed_optim.primary_v3_outcomes':SimpleNamespace(csv_bytes=csv_bytes)}
        def fixture_import(name,*args,**kwargs):
            if name in modules:return modules[name]
            raise ValueError('Unexpected import in isolated original body: '+name)
        runtime={'synthetic_fixture_only':True,'__file__':str(root/'NEVER_EXECUTE_SYNTHETIC_WORKER.py'),
            '__builtins__':{**vars(builtins),'__import__':fixture_import},'Path':Path,'json':json,
            'argparse':argparse,'os':SimpleNamespace(getpid=lambda:-100,nice=lambda value:owner.events.append('nice:'+str(value))),
            'time':SimpleNamespace(monotonic=time.monotonic,sleep=lambda _:owner.events.append('fixture-wait')),
            'traceback':traceback,'RUN':root/'run','TOKENS':tuple(str(i) for i in range(8)),
            'SETTINGS':{'cpu_feature_nice':10,'validation_resource_priority':'all_twelve_full_validations_before_first_encoding'},
            'FUNCTIONAL_SHA':'e'*64,'INPUTS_SHA':'f'*64,
            'authenticate':lambda *a,**kw:(owner.entry,owner.context,{}),
            'process_identity':lambda pid,command=None:{'pid':pid,'ppid':-100,'start_ticks':pid,
                'command':command or ['SYNTHETIC_COORDINATOR']},
            'subprocess':SimpleNamespace(synthetic_fixture_only=True,Popen=Child,DEVNULL=None,STDOUT=None),
            'reuse_pretrained_callback':lambda args,entry,context,job:flow.reuse_pretrained(args,entry,context,job,owner.origin)}
        self.ns,self.binding=flow.construct_fixture_flow(OLD/'dispatch.py',runtime)

    def run(self):
        with patch('sys.stdout',new=io.StringIO()):
            return self.ns['coordinate'](self.args)

    def observe(self,exact=None):
        prefix=['/usr/bin/python','-B',self.ns['__file__'],'--source-sha',self.args.source_sha,
                '--authorization-sha',self.args.authorization_sha]
        return flow.observe_records(self.root/'run',CELLS,{j['plan']['state']['cell']:digest(j['plan']) for j in self.context.jobs},
            source_sha=self.args.source_sha,authorization_sha=self.args.authorization_sha,command_prefix=prefix,
            coordinator_pid=-100,observe_exact=exact or (lambda _:(_ for _ in ()).throw(AssertionError('Unexpected process read'))))


class Tests(unittest.TestCase):
    def fixture(self,fail=None):
        return Fixture(Path(tempfile.mkdtemp(prefix='case-',dir=FIXTURES)),fail)

    def test_complete_sixty_new_plus_reused_full_flow(self):
        f=self.fixture();out=f.run();s=f.observe()
        self.assertEqual(out['feature_states'],61);self.assertEqual(f.encoded,list(CELLS[1:]))
        self.assertEqual((len(f.children),len(f.feature_calls)),(61,122))
        self.assertEqual(f.events.count('lease-enter'),60);self.assertEqual(f.events.count('lease-close'),60)
        self.assertEqual([f.feature_calls.count(c) for c in CELLS],[2]*61)
        self.assertEqual((len(s['new_encoded']),len(s['new_verified']),len(s['features_verified'])),(60,60,61))
        self.assertTrue(s['reused_pretrained']);self.assertEqual(s['active_tokens'],[])
        self.assertFalse((f.root/'run/jobs/pretrained.exited.json').exists())
        self.assertEqual(len(f.binding['unchanged_function_bodies']),11)
        self.assertEqual(f.binding['coordinator_edits'],2)
        for n in ('manifest.json','vectors.npz'):
            self.assertEqual(flow.identity(f.entry.OUTPUT/'vectors/states/pretrained'/n),f.origin_before[n])
        OBSERVATIONS.append({'fixture_root':str(f.root),'new_stub_encodings':60,'reused_pretrained':1,
             'feature_stub_calculations':122,'observed_new_workers':60,'observed_features':61,'binding':f.binding,
             'native_numerical_acceptance':False,'actual_model_execution':False})

    def test_worker_failure_stops_and_preserves_exit(self):
        f=self.fixture('worker-exit')
        with self.assertRaisesRegex(ValueError,'no retry'):f.run()
        self.assertEqual(len(f.children),1);self.assertEqual(f.encoded,[])
        self.assertEqual(json.loads((f.root/'run/jobs'/f'{CELLS[1]}.exited.json').read_bytes())['exit_code'],7)
        self.assertTrue((f.root/'run/failed.json').exists());self.assertEqual(f.feature_calls,[])
        self.assertEqual(f.events.count('lease-enter'),f.events.count('lease-close'))

    def test_incomplete_finalizer_stops_before_features(self):
        f=self.fixture('incomplete-finalizer')
        with self.assertRaisesRegex(ValueError,'omitted states'):f.run()
        self.assertEqual(len(f.children),60);self.assertEqual(f.feature_calls,[])
        self.assertFalse((f.root/'run/vectors.completed.json').exists())

    def test_feature_worker_failure_preserves_vectors(self):
        f=self.fixture('feature-exit')
        with self.assertRaisesRegex(ValueError,'no automatic retry'):f.run()
        self.assertTrue((f.root/'run/vectors.completed.json').exists())
        self.assertFalse((f.root/'run/completed.json').exists())
        self.assertEqual(json.loads((f.root/'run/features.exited.json').read_bytes())['exit_code'],8)

    def test_feature_kernel_and_readback_failures_are_not_completion(self):
        for failure in ('feature-kernel','feature-readback'):
            with self.subTest(failure=failure):
                f=self.fixture(failure)
                with self.assertRaisesRegex(ValueError,'no automatic retry'):f.run()
                self.assertFalse((f.root/'run/features.completed.json').exists())
                self.assertFalse((f.root/'run/completed.json').exists())

    def test_reuse_rejects_changed_origin(self):
        f=self.fixture();f.origin['files']['vectors.npz']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Bound origin'):f.run()
        self.assertEqual(f.children,[]);self.assertFalse(f.entry.OUTPUT.joinpath('vectors/states/pretrained').exists())

    def test_reuse_checks_copied_native_read(self):
        f=self.fixture('copied-native-readback')
        with self.assertRaisesRegex(ValueError,'copied native'):f.run()
        self.assertEqual(f.children,[]);self.assertTrue((f.root/'run/failed.json').exists())
        self.assertFalse((f.root/'run/pretrained.reused.json').exists())

    def test_reuse_never_invents_new_worker_exit(self):
        f=self.fixture();f.run();record=json.loads((f.root/'run/pretrained.reused.json').read_bytes())
        self.assertFalse(record['new_worker_created']);self.assertFalse(record['model_reencoded'])
        self.assertNotIn('actual_exit_code',record);self.assertEqual(record['origin'],f.origin)

    def test_fresh_namespace_and_state_order_required(self):
        f=self.fixture();f.context.jobs.reverse()
        with self.assertRaisesRegex(ValueError,'ordered 61-state'):f.run()
        self.assertEqual(f.children,[])
        g=self.fixture();g.run()
        with self.assertRaisesRegex(ValueError,'existing campaign'):g.run()

    def test_validation_failure_prevents_reuse_and_workers(self):
        f=self.fixture();write_new(f.root/'validation/run/failed.json',{'synthetic_failure':True})
        with self.assertRaisesRegex(ValueError,'validation reported failure'):f.run()
        self.assertEqual(f.children,[]);self.assertFalse((f.root/'run/pretrained.reused.json').exists())

    def test_observer_rejects_changed_nested_worker_bindings(self):
        for key,value in (('source_sha256','wrong'),('authorization_sha256','wrong'),('ppid',-999)):
            with self.subTest(key=key):
                f=self.fixture();f.run();p=f.root/'run/jobs'/f'{CELLS[-1]}.started.json'
                record=json.loads(p.read_bytes());record[key]=value;p.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'worker identity'):f.observe()

    def test_observer_rejects_terminal_identity_change(self):
        f=self.fixture();f.run();p=f.root/'run/jobs'/f'{CELLS[-1]}.exited.json'
        record=json.loads(p.read_bytes());record['start_ticks']-=1;p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Terminal identity'):f.observe()

    def test_observer_rejects_changed_verification_anchor(self):
        f=self.fixture();f.run();p=f.root/'run/jobs'/f'{CELLS[-1]}.verified.json'
        record=json.loads(p.read_bytes());record['worker_receipt']['sha256']='0'*64;p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Verified chain'):f.observe()

    def test_observer_missing_handle_is_not_terminal(self):
        f=self.fixture('worker-exit')
        with self.assertRaises(ValueError):f.run()
        path=f.root/'run/jobs'/f'{CELLS[1]}.exited.json';path.rename(f.root/'held-original-exit.json')
        seen=[]
        def exact(record):seen.append(record['pid']);return {'state':'missing_reconcile_with_terminal_records'}
        result=f.observe(exact)
        self.assertEqual(len(seen),1);self.assertFalse(result['workers'][0]['terminal'])
        self.assertNotIn('exit_code',result['workers'][0])

    def test_production_runtime_and_extra_source_refused(self):
        with self.assertRaisesRegex(ValueError,'offline process fixtures'):
            flow.construct_fixture_flow(OLD/'dispatch.py',{})
        self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('embed_optim',sys.modules)


def protected():
    auth=json.loads((OLD/'authorization.json').read_bytes())
    paths={Path(p):b for p,b in auth['sources'].items()}
    for rel,sha in (('dispatch.py',flow.DISPATCH_SHA),('inputs.json','be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067'),
        ('authorization.json','d72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336'),
        ('run/failed.json','c84f93bc431612be2b885d7a467828d08a18d5f259829c4b925a4f23ea7def2e')):
        paths[OLD/rel]={'sha256':sha}
    for role in ('started','exited','encoded','verified'):
        p=OLD/'run/jobs'/f'pretrained.{role}.json';paths[p]=flow.identity(p)
    baseline=OLD.parent.parent/'analyses/dense-primary-v3-functional-dimensions/vectors/states/pretrained'
    for n,sha in (('manifest.json','90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac'),
                  ('vectors.npz','e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f')):
        paths[baseline/n]={'sha256':sha}
    for rel in ('observe.py','test_dispatch.py','tests-first.json'):paths[OLD/rel]=flow.identity(OLD/rel)
    output={}
    for p,b in paths.items():
        if p.name=='gpu.py':raise ValueError('Protected helper is outside this task')
        output[str(p)]=flow.bound(p,b)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    args=parser.parse_args();assert args.attempt.isalnum()
    FIXTURES=WORK/('fixtures-'+args.attempt);FIXTURES.mkdir(exist_ok=False)
    before=protected()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    after=protected();same(before,after)
    value={'scope':'full-functional-recovery-flow-offline-integration-tests','observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'completed_full_fixture_flows':OBSERVATIONS,'source':flow.identity(Path(flow.__file__)),
        'test_source':flow.identity(Path(__file__)),'protected_files':after,
        'actual_model_or_gpu_execution':False,'native_numeric_acceptance':False,'real_process_or_lease_inspection':False,
        'execution_authority_created':False,'original_sources_modified':False,'production_launcher':False,
        'scientific_completion':False}
    write_new(WORK/('tests-'+args.attempt+'.json'),value)
    raise SystemExit(0 if result.wasSuccessful() else 1)
