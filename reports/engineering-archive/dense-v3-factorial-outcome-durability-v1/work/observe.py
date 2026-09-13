"""Read only this exact source-bound backup waiter and its own receipts."""
import argparse
import json
from pathlib import Path
import outcomes as o

SOURCE = 'a27fbbd07c3b6bbe1b8d2675be7c4ad58ec13c6ca74d8ed66c97d993aa7c4183'
AUTH = '4b48d60de683ea3af68808987a29499f32369dabdb5cf35356ba5e5e269c0fba'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    o.need(args.output.is_absolute() and not args.output.exists(), 'Use a new absolute observation')
    t = o.transport()
    o.authenticate(t, SOURCE, AUTH)
    root = o.HERE / 'run'
    started = t.read(root / 'started.json')
    o.need(started['source_sha256'] == SOURCE and started['authorization_sha256'] == AUTH, 'Wrong exact waiter')
    completion = t.read(root / 'completed.json') if (root / 'completed.json').exists() else None
    value = {'observed_at_utc': t.stamp(),
        'coordinator': o.process(started['pid'], started['start_ticks'], [str(o.HERE / 'outcomes.py'), 'coordinate', SOURCE, AUTH]),
        'original_summary_complete': (o.SUMMARY / 'run/completed.json').exists(),
        'manifest_created': (o.HERE / 'artifact_manifest.json').exists(),
        'upload_started': (o.HERE / 'upload-started.json').exists(),
        'upload_returned': (o.HERE / 'upload-returned.json').exists(),
        'remote_verification_complete': (o.HERE / 'remote-verified.json').exists(),
        'actual_anonymous_download_complete': (o.HERE / 'download-verified.json').exists(),
        'failed': (root / 'failed.json').exists(), 'completion': completion,
        'gpu_access': False, 'read_only_exact_registered_handle': True, 'full_goal_complete': False}
    t.write_new(args.output, value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
