"""Load both genuine corrected source states and observe four fresh optimizer resets."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from audit_data import STORY, PRIMARY, file_identity, read_json, require


EXPERIMENT = Path('/root/embedding-optimizer-v3-experiment')
SOURCE = Path('/tmp/dense-v3-factorial-trainer-integration.dmkUEso6/source-third')
SOURCE_MANIFEST = SOURCE.parent / 'source-third.json'
SOURCE_SHA = '634747159f3ee871d9d0e9ca7bd4eb04567f561695184f2b70a4be12e938952c'
FUNCTIONAL_INPUTS = EXPERIMENT / 'launch/functional-dimensions/inputs.json'
FUNCTIONAL_SHA = 'be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067'
DATA_SHA = '94ea1b27631cf9270f445e6406e79c0c470bd346564a4057cee1f742923d103f'


def audit(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('HF_HUB_OFFLINE') == '1',
            'Require CPU-only offline loading; no GPU or remote calls')
    require(not args.output.exists(), 'Preserve previous source-state audit attempts')
    os.nice(10)
    require(file_identity(SOURCE_MANIFEST)['sha256'] == SOURCE_SHA, 'Actual Trainer source assembly changed')
    source_manifest = read_json(SOURCE_MANIFEST)
    for relative, record in source_manifest['files'].items():
        require(file_identity(SOURCE / relative) == {k: record[k] for k in ('bytes', 'sha256')},
                'A verified Trainer source changed')
    require(file_identity(args.data_receipt)['sha256'] == DATA_SHA, 'Full actual data-relation receipt changed')
    data = read_json(args.data_receipt)
    require(data['all_selected_field_values_equal_to_revised_primary'] is True and
            data['scientific_admission'] is False, 'Require actual data-only acceptance')
    # The full data audit has just verified every input before/after its complete
    # numeric-field read. Retain that exact receipt; do not repeat the 500K scan.
    require(file_identity(FUNCTIONAL_INPUTS)['sha256'] == FUNCTIONAL_SHA, 'Complete primary admission input changed')
    functional = read_json(FUNCTIONAL_INPUTS)
    require(functional['scientific_completion'] is False and functional['native_guards_modified'] is False,
            'Wrong actual primary completion scope')
    import torch
    from accelerate import Accelerator
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer

    from embed_optim import config, dense_numerical_contract, factorial_v3_optimizer, optimizers
    from embed_optim.primary_contract import digest
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.primary_v3_io import compare_remote

    modules = (config, dense_numerical_contract, factorial_v3_optimizer, optimizers)
    source_files = {}
    for module in modules:
        path = Path(inspect.getsourcefile(module)).resolve()
        require(path.parent == SOURCE / 'src/embed_optim', 'Imported numerical code from another tree')
        source_files[str(path)] = file_identity(path)
    contract = PrimaryV3Contract.load(STORY / 'configs/dense_primary_v3_protocol.json', STORY, PRIMARY)
    require(contract.sha256 == '4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b',
            'The genuine primary protocol differs')
    design = read_json(STORY / 'configs/dense_no_packing_state_operator_factorial_protocol.json')
    require(design['source_states']['checkpoint_step'] == 2345, 'Frozen source stage differs')
    source_plans = []
    for declared in design['source_states']['states']:
        run_id = {'adamw_state': 'verified-v3-adamw-3e-5', 'muon_state': 'verified-v3-muon-3e-4'}[declared['label']]
        expected = contract.expected_identity(run_id)
        require(expected['recipe']['optimizer']['name'] == declared['optimizer'] and
                expected['recipe']['optimizer']['lr'] == declared['learning_rate'],
                'The revised source changes the predeclared optimizer/rate')
        matching = [job for job in functional['jobs']
                    if job['plan']['state']['cell'] == f'{run_id}/checkpoint-2345']
        require(len(matching) == 1, 'Missing unique actual fixed source state')
        source_plans.append((declared, run_id, matching[0]))
    accelerator = Accelerator(cpu=True)
    states = []
    for declared, run_id, job in source_plans:
        checkpoint = Path(job['checkpoint'])
        require(checkpoint == EXPERIMENT / contract.payload['output_root'] / 'dense' / run_id / 'checkpoint-2345',
                'The source path is not the genuine corrected checkpoint')
        checked = contract.checkpoint(checkpoint, run_id, 2345)
        require({k: checked[k] for k in job['plan']['model']['checkpoint']} == job['plan']['model']['checkpoint'],
                'Fresh native checkpoint read disagrees with actual complete-primary admission')
        run_proof = functional['admitted']['complete_runs'][run_id]
        require(run_proof['whole_run_artifacts_verified'] is True and
                digest(run_proof) == job['plan']['model']['complete_run_sha256'] and
                checked['run_identity_sha256'] == job['plan']['model']['run_identity_sha256'],
                'The original complete-run proof differs')
        receipt = EXPERIMENT / 'launch/backup-receipts' / f'{run_id}-step-2345.json'
        audit_receipt = receipt.with_name(f'{run_id}-step-2345.audit.json')
        remote = read_json(receipt)
        prior_audit = read_json(audit_receipt)
        require(remote['checkpoint'] == checked and prior_audit['durability_verified'] is True and
                prior_audit['commit_oid'] == remote['commit_oid'] and
                remote['repo_id'] == contract.payload['checkpoint_repository'],
                'Original immutable checkpoint durability proof differs')
        compare_remote(checkpoint, checked['files'], remote['remote_inventory'])
        model = SentenceTransformer(
            str(checkpoint), device='cpu', local_files_only=True,
            model_kwargs={'dtype': torch.float32, 'attn_implementation': 'sdpa'},
        )
        dense_numerical_contract.require_backward(model)
        require(model.max_seq_length == 8192 and model[0].can_flatten_inputs is False,
                'Loaded context/execution flags differ')
        parameters = dict(model.named_parameters())
        require({n: list(p.shape) for n, p in parameters.items()} == factorial_v3_optimizer.denseon_shapes(),
                'The genuine model differs from the complete default DenseOn topology')
        def verify_weights():
            with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as store:
                require({'0.model.' + name for name in store.keys()} == set(parameters), 'Saved tensor namespace differs')
                for name in store.keys():
                    value = parameters['0.model.' + name]
                    require(value.dtype == torch.float32 and torch.equal(value.detach(), store.get_tensor(name)),
                            'Actual loaded source weights differ from their original saved values')
        verify_weights()
        resets, layouts = {}, []
        for algorithm in ('hybrid_adamw', 'muon'):
            # Only a constructor/reset diagnostic. Real hidden rates must still
            # come from the specified eight-history-step GPU calibration.
            settings = config.OptimizerConfig(name=algorithm, lr=3e-4)
            optimizer = factorial_v3_optimizer.FactorialOptimizer(model, settings)
            scheduler = factorial_v3_optimizer.create_scheduler(optimizer)
            wrapped = accelerator.prepare_optimizer(optimizer, device_placement=True)
            observed = optimizer.state_dict()
            require(wrapped.optimizer is optimizer and observed['state'] == {} and
                    optimizer.completed_steps == 0 and all(p.grad is None for p in model.parameters()),
                    'The reset inherited or changed optimizer/gradient state')
            layout = optimizer.named_parameter_layout
            require([len(group['members']) for group in layout] == [88, 1, 45], 'Default routed topology differs')
            layouts.append(layout)
            resets[algorithm] = {
                'optimizer_steps': 0, 'parameter_states': 0, 'groups': optimizer.partition_summary,
                'scheduler': factorial_v3_optimizer.inspect_scheduler(scheduler.state_dict(), observed, settings, step=0),
                'diagnostic_constructor_hidden_lr_not_calibration': 3e-4,
            }
            accelerator.free_memory(wrapped)
            del optimizer, scheduler, wrapped
        require(layouts[0] == layouts[1], 'Two reset operators do not address identical named tensors')
        verify_weights()
        require(file_identity(checkpoint / 'model.safetensors') == next(
            {k: row[k] for k in ('bytes', 'sha256')} for row in checked['files'] if row['path'] == 'model.safetensors'),
            'Original saved weights changed during loading')
        states.append({
            'label': declared['label'], 'historical_selection_rule_record': declared,
            'genuine_v3_run_id': run_id, 'checkpoint': str(checkpoint), 'native_checkpoint': checked,
            'original_whole_run_proof': run_proof,
            'original_durability_receipt': {'path': str(receipt), **file_identity(receipt)},
            'original_remote_audit_receipt': {'path': str(audit_receipt), **file_identity(audit_receipt)},
            'immutable_remote_commit': remote['commit_oid'],
            'fresh_network_audit': False, 'stored_remote_inventory_matches_current_local_payload': True,
            'model_tensors': len(parameters), 'model_parameters': sum(p.numel() for p in parameters.values()),
            'loaded_weights_bitwise_equal_before_and_after': True, 'named_layout': layouts[0],
            'fresh_resets': resets, 'actual_forward_or_backward_executed': False,
            'loading_only_attention_backend': 'sdpa', 'formal_gpu_execution_verified': False,
        })
        print(json.dumps({'verified_source_state': declared['label'], 'run_id': run_id,
                          'model_tensors': len(parameters), 'fresh_reset_operators': 2}), flush=True)
        del model, parameters
        gc.collect()
    result = {
        'scope': 'genuine-corrected-factorial-source-state-loading-and-reset-only',
        'scientific_admission': False, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'states': states, 'source_manifest': {'path': str(SOURCE_MANIFEST), **file_identity(SOURCE_MANIFEST)},
        'source_files': source_files, 'data_relation_receipt': {'path': str(args.data_receipt), **file_identity(args.data_receipt)},
        'complete_primary_input': {'path': str(FUNCTIONAL_INPUTS), **file_identity(FUNCTIONAL_INPUTS)},
        'audit_source': {'path': str(Path(__file__).resolve()), **file_identity(__file__)},
        'calibration_performed': False, 'formal_branch_started': False,
        'boundary': 'Fresh genuine state loading/default routing and empty placement only. No GPU, forward/backward, update, calibration, legacy checkpoint adoption, source guard waiver, or scientific admission.',
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), **file_identity(args.output), 'states': 2, 'fresh_resets': 4}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
