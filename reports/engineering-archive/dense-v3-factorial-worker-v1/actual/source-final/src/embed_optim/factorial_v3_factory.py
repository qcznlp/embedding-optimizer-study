"""Build the fixed crossed-continuation recipe and actual model/Trainer inputs.

This internal factory does not acquire GPUs, start a process group, train, resume,
publish, or waive the separately required main-completion/resource/source gates.
The existing 68-file run component stays byte-identical. The factory's separate
creation record must be bound by the future admitted worker and its consumer.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import factorial_v3_run_contract as contract
from . import train as primary_train
from .config import DENSE_NUMERICAL_POLICY, OptimizerConfig, RunConfig
from .factorial_v3_bound_trainer import RunBoundFactorialTrainer, collective_phase, require_collator
from .factorial_v3_calibration import _verify_loaded_weights
from .factorial_v3_trainer import FactorialTrainingArguments
from .primary_contract import digest, file_identity, read_json, require_same, verify_file

SCOPE = "dense-v3-factorial-factory-v1"
BRANCH = "short-branch-50k-seed20260826"
# These are non-computational metadata, or a setting checked on the live model
# instead: Transformers removes the legacy checkpointing key during loading.
CONFIG_METADATA = ("_name_or_path", "transformers_version", "gradient_checkpointing")


def recipe(identity, branch_root, output_root, *, project, entity):
    """Pure resolved settings; a structural identity alone is not calibration admission."""
    identity = contract.require_identity(identity)
    branch_root, output_root = Path(branch_root), Path(output_root)
    if not branch_root.is_absolute() or not output_root.is_absolute():
        raise ValueError("Require explicit absolute branch/output roots")
    if any(not isinstance(v, str) or not v.strip() for v in (project, entity)):
        raise ValueError("Require explicit logging project and entity, never credentials")
    config = RunConfig(
        run_id=identity["run_id"],
        model_family="dense",
        optimizer=OptimizerConfig.from_dict(identity["optimizer"]),
        model_name=identity["source"]["checkpoint"],
        model_revision=None,
        dataset_path=str(branch_root),
        output_root=str(output_root),
        seed=identity["seed"],
        epochs=1.0,
        global_batch_size=128,
        micro_batch_size=8,
        temperature=0.02,
        max_length=8192,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        dataloader_workers=8,
        gradient_checkpointing=True,
        flash_attention=True,
        dense_can_flatten_inputs=False,
        wandb_project=project,
        wandb_entity=entity,
        checkpoint_fractions=tuple(contract.CHECKPOINT_POLICY["fractions"]),
        numerical_policy=DENSE_NUMERICAL_POLICY,
    )
    # Do not call the primary argument builder: its length-grouped sampler and
    # dropping policy belong to the 500K primary, not this 50K continuation.
    arguments = {
        "output_dir": str(config.output_dir),
        "run_name": identity["run_id"],
        "project": project,
        "report_to": ["wandb"],
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "learning_rate": config.optimizer.lr,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0.1,
        "max_grad_norm": 1.0,
        "save_strategy": "no",
        "save_only_model": False,
        "save_on_each_node": False,
        "save_total_limit": None,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "logging_first_step": True,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "use_cpu": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "full_determinism": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 4,
        "dataloader_drop_last": False,
        "train_sampling_strategy": "random",
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    return config, arguments


def model_configuration(model, checkpoint, *, diagnostic_cpu=False):
    """Compare saved model/tokenizer/pooling semantics, not just tensor shapes."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling
    from transformers import ModernBertConfig, ModernBertModel

    root = Path(checkpoint)
    if (
        type(model) is not SentenceTransformer
        or list(model._modules) != ["0", "1"]
        or type(model[0]) is not Transformer
        or type(model[1]) is not Pooling
        or type(model[0].model) is not ModernBertModel
    ):
        raise ValueError("Factory loaded a different model/module topology")
    expected = ModernBertConfig.from_dict(read_json(root / "config.json")).to_dict()
    observed = model[0].model.config.to_dict()
    for name in CONFIG_METADATA:
        expected.pop(name, None)
        observed.pop(name, None)
    require_same(observed, expected)
    require_same(model[1].get_config_dict(), read_json(root / "1_Pooling/config.json"))
    require_same(model[0].get_config_dict(), read_json(root / "sentence_bert_config.json"))
    settings = read_json(root / "config_sentence_transformers.json")
    for name in ("prompts", "default_prompt_name", "similarity_fn_name"):
        require_same(getattr(model, name), settings[name])
    tokenizer = model.tokenizer
    require_same(
        json.loads(tokenizer.backend_tokenizer.to_str()), read_json(root / "tokenizer.json")
    )
    source_tokens = read_json(root / "tokenizer_config.json")
    for name in ("padding_side", "truncation_side", "model_max_length"):
        if name in source_tokens:
            require_same(getattr(tokenizer, name), source_tokens[name])
    expected_backend = "sdpa" if diagnostic_cpu else "flash_attention_2"
    if model[0].model.config._attn_implementation != expected_backend:
        raise ValueError("Loaded attention backend differs from this declared path")
    if (
        model.truncate_dim is not None
        or model.get_sentence_embedding_dimension() != 768
        or model.max_seq_length != 8192
        or model[0].can_flatten_inputs is not False
        or not model[0].model.is_gradient_checkpointing
        or any(not module.training for module in model.modules())
        or any(
            module.p != 0.0 for module in model.modules() if isinstance(module, torch.nn.Dropout)
        )
        or any(p.grad is not None for p in model.parameters())
    ):
        raise ValueError(
            "Loaded mode, dimensions, checkpointing, dropout or fresh gradients differ"
        )
    return {
        "semantic_config_sha256": digest(observed),
        "pooling": model[1].get_config_dict(),
        "transformer": model[0].get_config_dict(),
        "prompts": dict(model.prompts),
        "tokenizer_backend_sha256": digest(json.loads(tokenizer.backend_tokenizer.to_str())),
        "embedding_dimensions": 768,
        "maximum_context": 8192,
        "backend": expected_backend,
        "gradient_checkpointing": True,
        "training_mode": True,
        "diagnostic_cpu": diagnostic_cpu,
        "forward_or_backward_executed": False,
    }


def load_model(identity, config, *, diagnostic_cpu=False):
    """Use the unchanged primary loader; CPU/SDPA is an explicit loading-only diagnostic."""
    import os

    import torch

    contract.require_identity(identity)
    if type(diagnostic_cpu) is not bool:
        raise ValueError("Loading diagnostic mode must be an explicit boolean")
    if diagnostic_cpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("CPU loading diagnostic requires hidden, uninitialized CUDA")
    else:
        require_gpu_context()
    contract.require_run_dependencies(identity)
    verify_file(Path(primary_train.__file__), identity["sources"]["src/embed_optim/train.py"])
    expected, _ = recipe(
        identity,
        config.dataset_path,
        config.output_root,
        project=config.wandb_project,
        entity=config.wandb_entity,
    )
    require_same(config.as_dict(), expected.as_dict())
    contract.require_source_checkpoint(identity)
    effective = dataclasses.replace(config, flash_attention=False) if diagnostic_cpu else config
    model, loss, collator = primary_train._load_model_and_loss(effective)
    model.train()
    _verify_loaded_weights(model, identity["source"]["checkpoint"])
    require_collator(collator, model)
    observed = model_configuration(
        model, identity["source"]["checkpoint"], diagnostic_cpu=diagnostic_cpu
    )
    if loss.model is not model or loss.temperature != 0.02:
        raise ValueError("Factory loss is not the unchanged shared-model fixed-temperature loss")
    if diagnostic_cpu and (model.device.type != "cpu" or torch.cuda.is_initialized()):
        raise ValueError("CPU loading diagnostic entered a GPU path")
    if not diagnostic_cpu and model.device != torch.device("cuda", torch.cuda.current_device()):
        raise ValueError("Model is not on the existing worker's current device")
    contract.require_source_checkpoint(identity)
    return model, loss, observed


def require_gpu_context():
    """Inspect only the caller's already-initialized four-rank owned context."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() != 4 or dist.get_backend() != "nccl":
        raise ValueError("Factory requires an already admitted four-rank NCCL worker")
    if not torch.cuda.is_initialized() or torch.cuda.device_count() != 4:
        raise ValueError("Factory requires the worker's four isolated initialized CUDA devices")


def prepare_trainer(
    locations,
    calibration_directories,
    state,
    operator,
    seed,
    source_root,
    output_root,
    *,
    project,
    entity,
):
    """Actual internal constructor; no launch, training, implicit resume or remote write.

    The caller must supply the separately accepted resource/source/main-completion
    handoff. Returning this component is not that handoff or scientific admission.
    Every rank must call this once inside its existing owned NCCL process group.
    """
    import os

    import torch.distributed as dist
    from transformers import set_seed

    require_gpu_context()
    identity, dataset = collective_phase(
        lambda: contract.prepare_run(
            locations,
            calibration_directories,
            state,
            operator,
            seed,
            source_root,
        )
    )
    config, values = recipe(
        identity,
        locations.data_store / BRANCH,
        output_root,
        project=project,
        entity=entity,
    )

    def preflight():
        output = config.output_dir
        if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError("A fresh branch must not overwrite or follow an earlier output")
        factory = Path(__file__).resolve()
        if factory.parent != Path(source_root).resolve() / "src/embed_optim":
            raise ValueError("Factory was imported outside the selected source assembly")
        expected_logging = {
            "WANDB_PROJECT": project,
            "WANDB_ENTITY": entity,
            "WANDB_RUN_ID": identity["run_id"],
            "WANDB_RESUME": "never",
        }
        if any(os.environ.get(k) != v for k, v in expected_logging.items()):
            raise ValueError(
                "Worker logging identity is not explicitly isolated for this fresh run"
            )
        return file_identity(factory)

    factory_source = collective_phase(preflight)
    local_identity = digest(
        {
            "identity": identity,
            "config": config.as_dict(),
            "arguments": values,
            "factory_source": factory_source,
        }
    )
    identities = [None] * 4
    dist.all_gather_object(identities, local_identity)
    if len(set(identities)) != 1:
        raise ValueError("Ranks disagree on the source/calibration/recipe/factory identity")

    # The collective above ensures no rank creates the shared output before all
    # ranks finish the fresh-output check. TrainingArguments has no output write.
    def arguments():
        args = FactorialTrainingArguments(**values)
        contract.require_arguments(args, identity)
        if args.world_size != 4 or args.local_process_index != dist.get_rank():
            raise ValueError("TrainingArguments changed the admitted single-node topology")
        return args

    args = collective_phase(arguments)
    set_seed(seed)
    model, loss, observed = collective_phase(lambda: load_model(identity, config))
    trainer = RunBoundFactorialTrainer(
        run_identity=identity,
        optimizer_config=config.optimizer,
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
    )
    return trainer, {
        "scope": SCOPE,
        "scientific_admission": False,
        "execution_authorized": False,
        "run_identity_sha256": digest(identity),
        "factory_source": factory_source,
        "recipe": config.as_dict(),
        "requested_arguments": values,
        "loaded_model": observed,
        "training_executed": False,
        "source_release_or_resource_gate_waived": False,
    }
