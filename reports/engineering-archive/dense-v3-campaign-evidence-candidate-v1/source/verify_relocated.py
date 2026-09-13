"""Actual isolated read with producer-path and write access forbidden after imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
source = Path(sys.argv[2])
assert hashlib.sha256(source.read_bytes()).hexdigest() == '8f5c2f0d0c48b7f6b2899247aa20fac98fd8dd18fc220a3a760e60f205117912'
spec = importlib.util.spec_from_file_location('relocated_campaign_evidence', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
opened = []


def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, (str, bytes)):
            path = os.fsdecode(path)
            p = Path(path)
            if p.is_absolute():
                if not p.is_relative_to(root):
                    raise AssertionError('Attempted absolute-path read outside relocated evidence')
            elif len(p.parts) != 1 or path in ('.', '..'):
                raise AssertionError('Unexpected relative path access')
            opened.append(path)
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('Reader attempted a write')
    elif event in ('socket.__new__', 'subprocess.Popen', 'os.system'):
        raise AssertionError('Reader attempted network or subprocess execution')


sys.addaudithook(audit)
result = module.read_complete_training_population(root)
for name in ('torch', 'numpy', 'transformers', 'datasets', 'embed_optim'):
    assert name not in sys.modules, f'Unexpected numerical/project import: {name}'
print(json.dumps({'scope': 'isolated-relocated-training-evidence-read',
                  'all_original_rows_sha256': module.digest(result['original_completion_rows']),
                  'absolute_reads_limited_to_relocated_root': True,
                  'reader_write_network_or_child_attempts': 0,
                  'numerical_or_project_imports': [], 'file_open_events': len(opened),
                  'result': {k: v for k, v in result.items() if k != 'original_completion_rows'}}, sort_keys=True))
