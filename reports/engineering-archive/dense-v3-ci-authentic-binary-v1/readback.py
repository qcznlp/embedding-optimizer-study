"""Anonymous immutable HTTP readback of every published positive file."""
import hashlib
import json
from pathlib import Path
import requests

work = Path(__file__).resolve().parent
published = json.loads((work / 'published.json').read_text())
manifest = json.loads((work / 'publication/manifest.json').read_text())
files = dict(manifest['files'])
files['manifest.json'] = {'bytes': (work / 'publication/manifest.json').stat().st_size, 'sha256': published['manifest_sha256']}
session = requests.Session()
session.trust_env = False
observed = {}
for name, expected in files.items():
    url = f"https://huggingface.co/datasets/{published['repo_id']}/resolve/{published['revision']}/{published['prefix']}/{name}"
    with session.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        size = 0
        for chunk in response.iter_content(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    observed[name] = {'bytes': size, 'sha256': digest.hexdigest()}
    assert observed[name] == expected, name
receipt = {'complete': True, 'authenticated': False, 'revision': published['revision'], 'files': observed}
with (work / 'anonymous-readback.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
print(json.dumps({'complete': True, 'files': len(observed), 'bytes': sum(v['bytes'] for v in observed.values()), 'revision': published['revision']}))
