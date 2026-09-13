"""Read-only final local-document/source checks; no remote or paper writes."""
import json
import re
from pathlib import Path
from urllib.parse import unquote

from embed_optim.primary_contract import file_identity
from embed_optim.primary_v3_validation_io import write_new

root = Path('/root/embedding-optimizer-story-refactor')
archive = root / 'reports/engineering-archive/dense-v3-geometry-reconstruction-v1'
docs = [root / name for name in ('AGENTS.md', 'PROJECT_STATUS.md', 'README.md')]
docs += [Path('/root/embedding-optimizer-study') / name for name in ('AGENTS.md', 'PROJECT_STATUS.md')]
docs += [archive / name for name in ('README.md', 'RUNNING.md', 'plan.md', 'commands.md',
    'next-bridge-boundary.md', 'attempt-1/README.md', 'attempt-2/README.md')]
checked = []
for path in docs:
    text = path.read_text()
    for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text):
        if target.startswith(('http:', 'https:', 'mailto:', '#')):
            continue
        target = unquote(target.strip('<>').split('#', 1)[0])
        target = re.sub(r':\d+$', '', target)
        resolved = (path.parent / target).resolve()
        assert resolved.exists(), (str(path), target)
        checked.append({'document': str(path), 'target': target})
    assert '20415 is still live' not in text and '20415 is live' not in text
result = {'scope': 'engineering_final_geometry_document_checks', 'passed': True,
    'documents': [{'path':str(p), **file_identity(p)} for p in docs],
    'local_links': checked, 'source': file_identity(__file__),
    'scientific_completion': False, 'remote_documents_verified': False}
write_new(Path(__file__).parent / 'document-checks.json', result)
print(json.dumps({'passed':True,'documents':len(docs),'local_links':len(checked)}), flush=True)
