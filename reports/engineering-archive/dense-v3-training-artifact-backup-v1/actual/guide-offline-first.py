import hashlib
import json
from pathlib import Path, PurePosixPath

destination = Path("/tmp/dense-v3-training-artifact-backup.j8qukV/download")
manifest_sha = "9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e"
root = destination / (
    "corrected-dense-correctness-v3/training-dynamics/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    + manifest_sha
)
if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
    raise ValueError("Require an absolute ordinary snapshot root")
manifest_path = root / "artifact_manifest.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise ValueError("Missing ordinary manifest")
raw = manifest_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != manifest_sha:
    raise ValueError("Manifest identity differs; do not replace the trusted hash")
manifest = json.loads(raw)
expected = set(manifest["files"]) | {"artifact_manifest.json"}
actual = set()
for path in root.rglob("*"):
    if path.is_symlink():
        raise ValueError("Symlinked payload refused")
    if path.is_file():
        actual.add(path.relative_to(root).as_posix())
if actual != expected or len(actual) != 149:
    raise ValueError("Incomplete or changed file population")
for name, record in manifest["files"].items():
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("Invalid payload path")
    path = root / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Payload differs: " + name)
print(json.dumps({"verified_files": len(actual), "manifest_sha256": manifest_sha}))
