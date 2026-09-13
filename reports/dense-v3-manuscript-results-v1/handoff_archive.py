"""Preserve actual cold replay and the eight-run live handoff."""
import json
import shutil
from pathlib import Path
from archive import HERE, WORK, identity

TRAIN = Path('/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1')


def main():
    pairs = []
    for name in ('verification.json', 'execution.json'):
        pairs.append((WORK / 'cold-complete' / name, HERE / 'cold-replay' / name))
    for name in ('training', 'evaluation', 'probe', 'summary'):
        pairs.append((WORK / (name + '-1820.json'), HERE / 'observations' / (name + '-1820.json')))
    observation = json.loads((WORK / 'training-1820.json').read_bytes())
    if observation['completed_branches'] != 8:
        raise ValueError('Wrong dated completion population')
    native = []
    for pool in ('a', 'b'):
        for row in observation['pools'][pool]['runs']:
            if row.get('complete') and row['run_id'].startswith('factorial-v3-muon_state-') and row['run_id'].endswith('271828'):
                native.append(row['complete'])
    if len(native) != 2 or any(r['actual_rank_exits'] != [0, 0, 0, 0]
                              or r['actual_fresh_reader_exit'] != 0 for r in native):
        raise ValueError('Missing actual fourth-pair completion')
    records = {}
    for source, target in pairs:
        before = identity(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
        if identity(source) != before or identity(target) != before:
            raise ValueError('Copy differs')
        records[str(target.relative_to(HERE))] = dict(origin=str(source), binding=before)
    value = dict(scope='actual-fragment-replay-and-eight-branch-handoff',
        copies=records, fourth_pair_original_completions=native,
        observation_at_utc=observation['observed_at_utc'], completed_continuations=8,
        completed_native_checkpoints=40, remote_checkpoints_verified_at_snapshot=30,
        next_pair_registered=True, scientific_completion=False)
    with (HERE / 'handoff.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(copies=len(records), receipt=identity(HERE / 'handoff.json'))))


if __name__ == '__main__':
    main()
