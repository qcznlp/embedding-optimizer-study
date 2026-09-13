"""Anonymous, pinned-commit source readback; no credential or write API used."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path('/root/embedding-optimizer-story-refactor')
work = Path(__file__).resolve().parent
commit = '803c70dcc3a2c195c5f56ee5fc0e3c633d645524'
repo = 'qcznlp/embedding-optimizer-study'
def read(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'DenseOn-source-publication-readback', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()
metadata = json.loads(read('https://api.github.com/repos/' + repo))
assert metadata['full_name'] == repo and metadata['private'] is False and metadata['default_branch'] == 'main'
paths = ['README.md', 'AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'paper/current/main.tex',
         'paper/Makefile', 'configs/source_test_roles.json', 'scripts/test_source_roles.py',
         'reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed/manifest.json',
         'reports/engineering-archive/dense-v3-release-transition-v1/manifest.json']
records = {}
for name in paths:
    remote = read('https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + name)
    local = (root / name).read_bytes()
    assert remote == local, name
    records[name] = {'bytes': len(remote), 'sha256': hashlib.sha256(remote).hexdigest()}
result = {'complete': True, 'repository': repo, 'commit': commit, 'public': True, 'default_branch': 'main',
          'anonymous_readback': True, 'files': records, 'scope': 'Ten source, handoff, manuscript and complete-replay/verification inventory files, not a new scientific execution'}
with (work / 'published-source-readback.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(json.dumps(result), flush=True)
