"""Read-only new-recovery observer; authenticates exact handles before /proc."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import recovery as dispatch
import recovery_support as support


def observe(args, exact=None):
    here, run = dispatch.HERE, dispatch.RUN
    auth, proposal = support.authority(here, args.source_sha, args.authorization_sha, dispatch.SETTINGS)
    started = support.read(run / 'coordinator.started.json', args.started_sha)
    prefix = ['/usr/bin/python', '-B', str(here / 'recovery.py'), '--source-sha', args.source_sha,
              '--authorization-sha', args.authorization_sha]
    support.need(started['source_sha256'] == args.source_sha
                 and started['authorization_sha256'] == args.authorization_sha
                 and started['command'] == prefix + ['--coordinate']
                 and type(started['pid']) is int and started['pid'] > 0
                 and type(started['start_ticks']) is int and started['start_ticks'] > 0
                 and started['settings'] == dispatch.SETTINGS, 'Wrong externally bound coordinator')
    exact = support.original_process_reader() if exact is None else exact
    _, flow = support.modules(here)
    states = flow.observe_records(run, proposal['state_order'],
        {cell: proposal['plans'][cell] for cell in proposal['state_order']},
        source_sha=args.source_sha, authorization_sha=args.authorization_sha,
        command_prefix=prefix, coordinator_pid=started['pid'], observe_exact=exact,
        output_root=support.OUTPUT, expected_reuse_origin=proposal['pretrained_origin'])
    feature = None
    if (run / 'features.started.json').exists():
        s = support.read(run / 'features.started.json')
        support.need(s['command'] == prefix + ['--features'] and s['ppid'] == started['pid']
                     and s['authorization_sha256'] == args.authorization_sha
                     and s['cuda_hidden'] is True and s['pid'] > 0 and s['start_ticks'] > 0,
                     'Wrong exact feature worker')
        if (run / 'features.exited.json').exists():
            e = support.read(run / 'features.exited.json')
            support.need(all(s[k] == e[k] for k in
                             ('pid', 'ppid', 'start_ticks', 'command', 'authorization_sha256')),
                         'Feature terminal identity differs')
            feature = {'terminal': True, 'exit_code': e['exit_code']}
        else:
            feature = {'terminal': False, 'observed': exact(s)}
    else:
        support.need(not (run / 'features.exited.json').exists(), 'Orphan feature exit')
    terminal = {}
    for name in ('vectors.completed', 'features.completed', 'completed', 'failed'):
        path = run / (name + '.json')
        if path.exists():
            value = support.read(path)
            support.need(value['authorization_sha256'] == args.authorization_sha
                         and value['scientific_completion'] is False, 'Wrong terminal authority')
            if name == 'failed':
                support.need(value['source_sha256'] == args.source_sha
                             and value['automatic_retry_authorized'] is False, 'Wrong failure boundary')
            elif name in ('vectors.completed', 'features.completed'):
                expected = support.OUTPUT / ('vectors' if name.startswith('vectors') else 'features') / 'manifest.json'
                support.need(value['manifest']['path'] == str(expected), 'Wrong terminal manifest location')
                support.verify_record(value['manifest'])
            terminal[name] = value
    support.need(not ('completed' in terminal and 'failed' in terminal), 'Conflicting terminal receipts')
    if 'vectors.completed' in terminal:
        support.need(len(states['new_verified']) == 60 and states['reused_pretrained']
                     and terminal['vectors.completed']['all_states_encoded_and_native_readback_verified'] == 61,
                     'Incomplete vector terminal chain')
    if 'features.completed' in terminal:
        support.need('vectors.completed' in terminal and len(states['features_verified']) == 61
                     and terminal['features.completed']['states'] == 61, 'Incomplete feature terminal chain')
    if 'completed' in terminal:
        support.need(feature == {'terminal': True, 'exit_code': 0}
                     and terminal['completed']['vectors'] == terminal.get('vectors.completed')
                     and terminal['completed']['features'] == terminal.get('features.completed'),
                     'Coordinator completion lacks a successful full child chain')
    coordinator = ({'terminal_record': 'completed' if 'completed' in terminal else 'failed'}
                   if 'completed' in terminal or 'failed' in terminal else exact(started))
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'coordinator': coordinator, 'coordinator_started': support.record(run / 'coordinator.started.json'),
            'states': states, 'feature_worker': feature, 'terminal_records': terminal,
            'writes_signals_or_broad_process_inspection': False,
            'fresh_numerical_replay': False, 'scientific_completion': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--authorization-sha', required=True)
    parser.add_argument('--started-sha', required=True)
    print(json.dumps(observe(parser.parse_args()), sort_keys=True), flush=True)
