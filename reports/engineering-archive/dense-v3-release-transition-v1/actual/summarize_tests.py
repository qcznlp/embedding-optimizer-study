"""Read actual JUnit results; summarize failure locations without dumping full values."""
import collections
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

work = Path(__file__).resolve().parent
tree = ET.parse(work / 'tests-before.xml')
groups = collections.defaultdict(list)
counts = collections.Counter()
for case in tree.iter('testcase'):
    outcome = next((kind for kind in ('failure', 'error', 'skipped') if case.find(kind) is not None), 'passed')
    counts[outcome] += 1
    if outcome in ('failure', 'error'):
        raw = case.find(outcome).get('message', '')
        raw = re.sub(r'(wandb_v1_|hf_|ghp_)[A-Za-z0-9_-]+', '[REDACTED]', raw)
        groups[case.get('classname')].append({'test': case.get('name'), 'kind': outcome, 'message': raw[:600]})
value = {'counts': dict(counts), 'groups': dict(groups)}
with (work / 'tests-summary.json').open('x') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
print(json.dumps({'counts': dict(counts), 'groups': {k: {'count': len(v), 'first': v[0]} for k, v in groups.items()}}, indent=2))
