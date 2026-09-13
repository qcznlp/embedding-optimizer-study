"""Authenticate official CPython source before a separate compatibility build."""
import hashlib
import json
from pathlib import Path
import tarfile

work=Path(__file__).resolve().parent
archive_path=work/'Python-3.12.3.tar.xz'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest()=='56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1'
assert hashlib.md5(archive_path.read_bytes(),usedforsecurity=False).hexdigest()=='8defb33f0c37aa4bdd3a38ba52abde4e'
sbom=json.loads((work/'source.spdx.json').read_text())
with tarfile.open(archive_path) as archive:
    archive.extractall(work,filter='data')
source=work/'Python-3.12.3'
verified=0
for member in sbom['files']:
    expected=next(v['checksumValue'] for v in member['checksums'] if v['algorithm']=='SHA256')
    path=source/member['fileName']
    assert path.resolve().is_relative_to(source) and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,member['fileName']
    verified+=1
with (work/'source-verified.json').open('x') as stream:
    json.dump({'source_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               'official_spdx_sha256_files_verified':verified,'source_edits':False},stream,indent=2)
print('Official source members verified:',verified)
