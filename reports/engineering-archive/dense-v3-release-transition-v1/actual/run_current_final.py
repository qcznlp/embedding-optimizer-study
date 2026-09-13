"""Rerun every current-role case after the five legacy-entry routing corrections.

The independent original-analysis/factorial role processes retain their exact
assembled sources and unmodified selected tests. They are not restarted.
"""
import json
from pathlib import Path

from scripts import test_source_roles as runner

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
output = work / 'current-role-final'
roles = runner.load_roles(root)
files = runner.inventory(root)
output.mkdir(exist_ok=False)
role = {**roles['current'], 'root': str(root)}
with (output / 'inputs.json').open('x') as stream:
    json.dump({'files': files, 'roles': {'current': role}}, stream, indent=2, sort_keys=True)
result = runner.execute('current', role, output, files)
print(json.dumps({'complete': result['complete'], 'cases': result.get('cases'), 'scope': 'all current-role cases; historical role results are recorded independently'}), flush=True)
raise SystemExit(0 if result['complete'] else 1)
