"""Read two exact W&B identities; disclose only whitelisted metadata/history."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
FIELDS = ('run_id', 'model_family', 'optimizer', 'model_name', 'model_revision', 'dataset_path',
          'output_root', 'seed', 'epochs', 'global_batch_size', 'micro_batch_size', 'temperature',
          'max_length', 'warmup_ratio', 'max_grad_norm', 'dataloader_workers', 'gradient_checkpointing',
          'flash_attention', 'dense_can_flatten_inputs', 'wandb_project', 'wandb_entity',
          'checkpoint_fractions', 'numerical_policy', 'learning_rate', 'num_train_epochs',
          'gradient_accumulation_steps', 'per_device_train_batch_size', 'run_name', 'report_to',
          'logging_first_step', 'logging_steps', 'warmup_steps', 'bf16', 'fp16', 'tf32', 'data_seed')
KEYS = ['train/global_step', 'train/loss', 'train/grad_norm', 'train/learning_rate']


def main():
    import wandb
    api = wandb.Api(timeout=30)
    records = []
    for run_id in ('study-v5-dense-verified-v3-adamw-3e-5-seed42-adamw',
                   'factorial-v3-adamw_state-adamw-seed314159'):
        remote = api.run('stevezenguom/embedding-optimizer-study/' + run_id)
        history = list(remote.scan_history(keys=KEYS, page_size=1000))
        config = {key: remote.config[key] for key in FIELDS if key in remote.config}
        record = {'id': remote.id, 'name': remote.name, 'state': remote.state,
                  'group': remote.group, 'tags': list(remote.tags or []), 'url': remote.url,
                  'config': config, 'config_keys_not_values': sorted(remote.config),
                  'summary': {key: remote.summary.get(key) for key in ('train/global_step', 'train/epoch')},
                  'history': history, 'read_only': True}
        with (ROOT / (run_id + '.remote-probe.json')).open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        records.append({key: record[key] for key in ('id', 'name', 'state', 'group', 'tags', 'summary', 'config')})
        records[-1].update(history_rows=len(history), history_end=history[-1])
    print(json.dumps({'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': records,
                      'read_only': True, 'scientific_completion': False}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not print exception/request representations that could disclose authentication.
        print(json.dumps({'read_only_probe_failed': True, 'exception_type': type(error).__name__}))
        raise SystemExit(1)
