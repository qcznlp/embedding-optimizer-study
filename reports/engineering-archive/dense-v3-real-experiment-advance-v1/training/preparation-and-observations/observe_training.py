"""Read only the two externally anchored new training coordinators/direct children."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')
SOURCE = 'd11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53'
AUTH = '00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f'
ANCHORS = {'a': (776959, 319670514, '60cffc21518c69dd6d73f32ee1005caf4302248270545ab24a7c17b95c9922a5'),
           'b': (777271, 319673599, '94d9cdedd6fbca9f879cdf362fedf882467e7d41b2f1b52bd1c9ce15c92822cd')}
sys.path.insert(0, str(ROOT))
import factorial_dispatch as entry


def snapshot():
    auth, _ = entry.authenticate(SimpleNamespace(source_sha=SOURCE, authorization_sha=AUTH))
    old = entry.EXPERIMENT / 'launch/functional-dimensions/observe.py'
    entry.need(entry.identity(old)['sha256'] == 'c314e4d540305bb67e0bcdcb6ad921d22926e4cbd40f59e00e84415be18cd078',
               'Pure exact-process reader changed')
    spec = importlib.util.spec_from_file_location('owned_rank_read_only_process', old)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    prefix = ['/usr/bin/python', '-B', str(ROOT / 'factorial_dispatch.py'), '--source-sha', SOURCE,
              '--authorization-sha', AUTH]
    pools = {}
    for pool, (pid, start, sha) in ANCHORS.items():
        root = ROOT / 'run' / f'pool-{pool}'
        begun = entry.read(root / 'coordinator.started.json', sha)
        entry.need((begun['pid'], begun['start_ticks'], begun['command']) ==
                   (pid, start, prefix + ['--coordinate', '--pool', pool])
                   and begun['source_sha256'] == SOURCE and begun['authorization_sha256'] == AUTH
                   and begun['queue'] == auth['queues'][pool], 'Wrong exact coordinator')
        terminals = [name for name in ('completed', 'failed') if (root / f'coordinator.{name}.json').exists()]
        entry.need(len(terminals) <= 1, 'Conflicting pool terminal records')
        rows = []
        for run_id in auth['queues'][pool]:
            job = root / run_id
            if not job.exists():
                rows.append({'run_id': run_id, 'status': 'not_started'}); continue
            admission = entry.read(job / 'admission.json')
            entry.need(admission['coordinator_pid'] == pid and admission['source_sha256'] == SOURCE
                       and admission['authorization_sha256'] == AUTH
                       and admission['request_sha256'] == entry.digest(auth['requests'][run_id])
                       and admission['gpu_pool'] == entry.POOLS[pool], 'Wrong new run admission')
            ended = entry.read(job / 'ranks.exited.json') if (job / 'ranks.exited.json').exists() else None
            if ended is not None:
                entry.need(ended['source_sha256'] == SOURCE and ended['authorization_sha256'] == AUTH,
                           'Wrong rank terminal authority')
            ranks = []
            for rank in range(4):
                path = job / f'rank-{rank}.started.json'
                if not path.exists(): continue
                s = entry.read(path); argv = s['command']
                expected = prefix + ['--pool', pool, '--worker', run_id, '--rank', str(rank)]
                entry.need(s['ppid'] == pid and s['coordinator_pid'] == pid and s['rank'] == rank
                           and s['source_sha256'] == SOURCE and s['authorization_sha256'] == AUTH
                           and argv[:-16] == expected and argv[-16::2] == ['--lease-fd'] * 8
                           and len({int(x) for x in argv[-15::2]}) == 8, 'Wrong exact owned rank')
                ranks.append({'rank': rank, 'terminal': ended is not None,
                    **({'exit_code': ended['actual_rank_exits'][str(rank)]} if ended is not None
                       else {'observed': reader.process(s)})})
            progress = []
            log = job / 'rank-0.log'
            if log.exists():
                # Native progress text is a dated step observation, not checkpoint acceptance.
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    progress = re.findall(r'\|\s*(\d+)/391\s*\[', stream.read().decode(errors='replace'))
            run_root = Path(auth['requests'][run_id]['run_root'])
            seals = [step for step in (79, 157, 235, 313, 391)
                     if (run_root / f'checkpoint-{step}/factorial_trainer_component.json').is_file()]
            row = {'run_id': run_id, 'ranks': ranks,
                   'native_progress_bar_step': int(progress[-1]) if progress else None,
                   'checkpoint_seals_present_not_freshly_read': seals,
                   'status': 'failed' if (job / 'failed.json').exists() else 'running_or_finalizing'}
            if (job / 'completed.json').exists():
                complete = entry.read(job / 'completed.json')
                entry.need(complete['actual_rank_exits'] == [0] * 4
                           and complete['actual_fresh_reader_exit'] == 0
                           and complete['source_sha256'] == SOURCE and complete['authorization_sha256'] == AUTH
                           and complete['full_horizon_391_steps_verified'] is True
                           and entry.identity(job / 'fresh-native-readback.json') == complete['native_readback_binding'],
                           'Incomplete whole-run terminal chain')
                row.update(status='full_training_and_fresh_readback_complete', complete=complete)
            rows.append(row)
        pools[pool] = {'coordinator': {'terminal_record': terminals[0]} if terminals else reader.process(begun),
                       'gpu_tokens': entry.POOLS[pool], 'runs': rows}
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'pools': pools,
            'source_sha256': SOURCE, 'authorization_sha256': AUTH,
            'completed_branches': sum(r['status'] == 'full_training_and_fresh_readback_complete'
                                      for pool in pools.values() for r in pool['runs']),
            'read_only_owned_handles': True, 'scientific_completion': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = snapshot()
    with args.output.open('x') as stream: stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True), flush=True)
