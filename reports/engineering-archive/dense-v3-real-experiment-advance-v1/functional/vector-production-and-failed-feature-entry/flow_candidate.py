"""Offline full-flow integration candidate; no production authorization or launcher."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

import record_layout as layout

DISPATCH_SHA = '3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5'
LAYOUT_SHA = 'f16dcc94d5eadc1f2a8da1fe50840b639c49919138b409fa8715fc81dbaaaa89'
FUNCTIONS = ('require', 'validation_priority', 'priority', 'job_for', 'environment',
             'worker_command', 'worker', 'encode_one', 'finalize_vectors', 'feature_worker',
             'coordinate', 'parse_args')


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Nonordinary evidence')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    need(all(getattr(before,k) == getattr(after,k) for k in
             ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'Evidence changed while reading')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def bound(path, expected):
    actual = identity(path)
    need(all(actual[k] == v for k,v in expected.items()), 'Bound origin changed')
    return actual


def read_record(path):
    before=identity(path)
    value=json.loads(Path(path).read_bytes())
    need(identity(path)==before,'Record changed during JSON read')
    return value


def reuse_pretrained(args, entry, context, job, origin):
    """Copy only accepted bytes; native readback remains an injected required dependency.

    This function does not authorize execution. The caller must authenticate the
    origin bindings and its new recovery authority before invoking it in production.
    No new encoded/started/exited/verified worker record is invented for pretrained.
    """
    need(job['plan']['state']['cell'] == 'pretrained', 'Only pretrained is reusable')
    records, vectors = Path(origin['records']), Path(origin['vectors'])
    need(set(origin['files']) == {'started.json','exited.json','encoded.json','verified.json',
                                 'manifest.json','vectors.npz'}, 'Incomplete origin bindings')
    paths = {n: vectors/n if n in ('manifest.json','vectors.npz') else records/('pretrained.'+n)
             for n in origin['files']}
    for name,path in paths.items(): bound(path,origin['files'][name])
    original = {n:json.loads(paths[n].read_bytes()) for n in
                ('started.json','exited.json','encoded.json','verified.json')}
    start, ended, encoded, verified = (original[n] for n in
                ('started.json','exited.json','encoded.json','verified.json'))
    need(all(x['cell']=='pretrained' for x in original.values()), 'Origin cell differs')
    need(all(start[k]==ended[k] for k in ('pid','ppid','start_ticks','command','authorization_sha256'))
         and ended['exit_code'] == verified['actual_exit_code'] == 0
         and verified['native_readback_passed'] is True, 'Origin terminal/readback proof differs')
    need(start['source_sha256'] == encoded['source_sha256'] == origin['source_sha256']
         and start['authorization_sha256'] == encoded['authorization_sha256'] == origin['authorization_sha256']
         and encoded['plan_sha256'] == context.digest(job['plan'])
         and verified['saved'] == encoded['saved'], 'Origin source/authority/plan differs')
    need(verified['worker_receipt'] == {'path':str(paths['encoded.json']),**origin['files']['encoded.json']},
         'Origin worker anchor differs')
    saved = encoded['saved']
    need(saved['manifest'] == {'path':str(vectors/'manifest.json'),**origin['files']['manifest.json']}
         and saved['output'] == {'path':'vectors.npz',**origin['files']['vectors.npz']}, 'Origin vector anchor differs')
    context_arrays, original_read = context.vectors.inspect_vectors(vectors, job['plan'], context.identities,
        job['checkpoint'], expected_manifest_sha256=saved['manifest']['sha256'])
    need(original_read == saved, 'Original native vector readback differs')
    del context_arrays
    target = entry.OUTPUT/'vectors/states/pretrained'
    need(not target.exists() and not target.is_symlink(), 'Preserve existing copied pretrained state')
    target.mkdir(parents=True,exist_ok=False)
    for name in ('manifest.json','vectors.npz'):
        with paths[name].open('rb') as src, (target/name).open('xb') as dst:
            shutil.copyfileobj(src,dst)
        bound(target/name,origin['files'][name])
    arrays, current = context.vectors.inspect_vectors(target,job['plan'],context.identities,
        job['checkpoint'],expected_manifest_sha256=saved['manifest']['sha256'])
    expected = copy.deepcopy(saved)
    expected['manifest']['path'] = str(target/'manifest.json')
    need(current == expected, 'Copied native read differs beyond its explicit location')
    del arrays
    for name,path in paths.items(): bound(path,origin['files'][name])
    context.geometry.write_new(args.run_root/'pretrained.reused.json', {
        'cell':'pretrained','scope':'pretrained-origin-preserving-reuse',
        'authorization_sha256':args.authorization_sha,'source_sha256':args.source_sha,
        'plan_sha256':context.digest(job['plan']),'origin':origin,
        'original_worker_exit_code':0,'new_worker_created':False,'model_reencoded':False,
        'native_original_and_copied_readback':True,'saved':current,'scientific_completion':False})
    return {'path':'states/pretrained/manifest.json',**identity(target/'manifest.json')}


def construct_fixture_flow(original, runtime):
    """Execute original operational bodies only in an explicit simulation namespace.

    No imports/authentication/process implementation is borrowed implicitly.
    There is no production entry point or authority writer in this candidate.
    """
    need(runtime.get('synthetic_fixture_only') is True
         and getattr(runtime.get('subprocess'),'synthetic_fixture_only',False) is True,
         'Only explicit offline process fixtures are admitted')
    bound(original,{'sha256':DISPATCH_SHA})
    bound(Path(layout.__file__),{'sha256':LAYOUT_SHA})
    tree = ast.parse(Path(original).read_text())
    originals = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS}
    need(set(originals)==set(FUNCTIONS),'Original operational function inventory differs')
    nodes = copy.deepcopy(originals)
    coordinate = nodes['coordinate']
    mkdir = ast.parse("(RUN / 'jobs').mkdir(parents=True, exist_ok=False)").body[0]
    added = ast.parse("prepare_record_parents(RUN / 'jobs', [j['plan']['state']['cell'] for j in context.jobs])").body[0]
    hits = [i for i,n in enumerate(coordinate.body) if ast.dump(n)==ast.dump(mkdir)]
    need(len(hits)==1,'Expected exclusive jobs-root creation differs')
    coordinate.body.insert(hits[0]+1,added)
    target = ast.parse("states[job['plan']['state']['cell']] = encode_one(args, entry, context, job)").body[0]
    replacement = ast.parse("states[job['plan']['state']['cell']] = reuse_pretrained_callback(args, entry, context, job) if job['plan']['state']['cell'] == 'pretrained' else encode_one(args, entry, context, job)").body[0]
    replacements = 0
    for n in ast.walk(coordinate):
        if isinstance(n,ast.For):
            for i,item in enumerate(n.body):
                if ast.dump(item)==ast.dump(target): n.body[i]=replacement; replacements+=1
    need(replacements==1,'Expected one all-state encoding loop differs')
    for name in FUNCTIONS:
        if name != 'coordinate': need(ast.dump(nodes[name])==ast.dump(originals[name]),'Original body changed')
    namespace = dict(runtime)
    namespace['prepare_record_parents'] = layout.prepare_record_parents
    module = ast.fix_missing_locations(ast.Module(body=list(nodes.values()),type_ignores=[]))
    exec(compile(module,'<offline-full-functional-flow>','exec'),namespace)
    return namespace, {'unchanged_function_bodies': [n for n in FUNCTIONS if n!='coordinate'],
                       'coordinator_edits':2,'original_dispatch_sha256':DISPATCH_SHA,
                       'synthetic_fixture_only':True,'production_launcher':False}


def observe_records(run_root, cells, plans, *, source_sha, authorization_sha,
                    command_prefix, coordinator_pid, observe_exact, output_root, expected_reuse_origin):
    """Validate all nested record chains; no direct /proc, GPU, writes or signals.

    A future admitted observer must first bind its coordinator and sources, then
    inject the unchanged exact-process reader. Missing handles are NOT terminal.
    """
    layout.validate_cells(cells)
    need(tuple(plans)==tuple(cells),'Observation plan order differs')
    records={}
    for role in layout.JSON_ROLES:
        records[role]={}
        for p in layout.existing_records(run_root/'jobs',cells,role):
            row=read_record(p)
            need(p==run_root/'jobs'/f"{row['cell']}.{role}.json",'Record cell changed after layout read')
            records[role][row['cell']]=(p,row)
    result = {'new_encoded':[],'new_verified':[],'features_verified':[],'workers':[],
              'reused_pretrained':False,'active_tokens':[]}
    for cell in cells:
        s = records['started'].get(cell); e = records['exited'].get(cell)
        a = records['admission'].get(cell); encoded = records['encoded'].get(cell)
        verified = records['verified'].get(cell); feature = records['features-verified'].get(cell)
        if any((s,e,encoded,verified)):
            need(a is not None and s is not None,'Orphan worker chain')
            start = s[1]; admission = a[1]; token = start['gpu_token']; argv = start['command']
            expected = command_prefix+['--worker',cell,'--plan-sha',plans[cell], '--gpu-token',token]
            need(cell!='pretrained' and token in tuple(str(i) for i in range(8))
                 and start['ppid']==coordinator_pid and start['source_sha256']==source_sha
                 and start['authorization_sha256']==authorization_sha
                 and argv[:-4]==expected and argv[-4]=='--lease-fd' and argv[-2]=='--lease-fd'
                 and len({int(argv[-3]),int(argv[-1])})==2,'Wrong exact new worker identity')
            need(admission['plan_sha256']==plans[cell] and admission['source_sha256']==source_sha
                 and admission['authorization_sha256']==authorization_sha and admission['gpu_token']==token,
                 'Worker admission differs')
            status={'cell':cell,'gpu_token':token,'terminal':e is not None}
            if e is not None:
                need(all(e[1][k]==start[k] for k in ('pid','ppid','start_ticks','command','authorization_sha256')),
                     'Terminal identity differs')
                status['exit_code']=e[1]['exit_code']
            else:
                status['observed']=observe_exact(start)
                if status['observed']['state'] not in ('missing_reconcile_with_terminal_records','Z'):
                    result['active_tokens'].append(token)
            result['workers'].append(status)
        if encoded:
            need(encoded[1]['plan_sha256']==plans[cell] and encoded[1]['source_sha256']==source_sha
                 and encoded[1]['authorization_sha256']==authorization_sha,'Encoded source/plan differs')
            result['new_encoded'].append(cell)
        if verified:
            need(encoded is not None and e is not None and e[1]['exit_code']==0
                 and verified[1]['actual_exit_code']==0 and verified[1]['native_readback_passed'] is True
                 and verified[1]['saved']==encoded[1]['saved']
                 and verified[1]['worker_receipt']=={'path':str(encoded[0]),**identity(encoded[0])},
                 'Verified chain differs')
            result['new_verified'].append(cell)
        if feature:
            need(feature[1]['fresh_raw_vectors_fully_recomputed'] is True,'Feature replay flag differs')
            need(cell=='pretrained' or verified is not None,'Feature lacks a verified vector chain')
            item=feature[1]['manifest']
            need(item['path']==f'states/{cell}/manifest.json','Feature manifest path differs')
            bound(output_root/'features'/item['path'],{k:item[k] for k in ('bytes','sha256')})
            result['features_verified'].append(cell)
    reused = run_root/'pretrained.reused.json'
    if reused.exists():
        record=read_record(reused)
        need(record['cell']=='pretrained' and record['source_sha256']==source_sha
             and record['authorization_sha256']==authorization_sha and record['plan_sha256']==plans['pretrained']
             and record['new_worker_created'] is False and record['model_reencoded'] is False
             and record['native_original_and_copied_readback'] is True,'Pretrained reuse boundary differs')
        need(record['origin']==expected_reuse_origin,'Pretrained origin differs from external recovery binding')
        origin=expected_reuse_origin
        for name,item in origin['files'].items():
            original=(Path(origin['vectors'])/name if name in ('manifest.json','vectors.npz')
                      else Path(origin['records'])/('pretrained.'+name))
            bound(original,item)
        target=output_root/'vectors/states/pretrained'
        for name in ('manifest.json','vectors.npz'):bound(target/name,origin['files'][name])
        need(record['saved']['manifest']=={'path':str(target/'manifest.json'),**origin['files']['manifest.json']},
             'Copied pretrained location or anchor differs')
        result['reused_pretrained']=True
    need('pretrained' not in result['features_verified'] or result['reused_pretrained'],
         'Pretrained feature lacks authenticated reuse evidence')
    need(len(result['active_tokens'])<=1,'Concurrent functional GPU workers')
    return result


if __name__ == '__main__':
    raise SystemExit('Offline integration candidate only; no production recovery is authorized')
