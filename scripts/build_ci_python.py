"""Build an isolated, original-source CPython for CPU-emulated paper replay.

Select CPython's existing per-descriptor closing path. Never disable close_fds,
patch Python/study source, change a system interpreter or install dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

SOURCE_URL = "https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tar.xz"
SOURCE_SHA256 = "56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1"
BUILD_ENVIRONMENT = (
    "CC",
    "CFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "LIBS",
    "LIBFFI_CFLAGS",
    "LIBFFI_LIBS",
    "BZIP2_CFLAGS",
    "BZIP2_LIBS",
    "LIBSQLITE3_CFLAGS",
    "LIBSQLITE3_LIBS",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packages-directory", type=Path, action="append", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    output = args.output
    packages = args.packages_directory
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("This recorded CPU replay build requires Linux x86_64")
    if not 1 <= args.jobs <= 32:
        raise ValueError("Use between one and 32 CPU build jobs")
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new absolute output with an existing parent")
    if any(not path.is_absolute() or not path.is_dir() for path in packages):
        raise ValueError("Existing absolute scientific package directories required")
    for path in [output, *packages]:
        if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Ordinary canonical directory paths required")
        if "\n" in str(path) or "\r" in str(path):
            raise ValueError("Path-only dependency records must occupy one line")
    output.mkdir()
    archive = output / "Python-3.12.3.tar.xz"
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, archive.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official CPython archive hash differs; no extraction or execution")
    with tarfile.open(archive) as source_archive:
        source_archive.extractall(output, filter="data")
    source = output / "Python-3.12.3"
    prefix = output / "install"
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", *BUILD_ENVIRONMENT)
        if key in os.environ
    }
    env.update(CUDA_VISIBLE_DEVICES="", ac_cv_func_close_range="no")
    commands = []

    def run(name: str, command: list[str], cwd: Path) -> None:
        start = time.monotonic()
        with (output / (name + ".log")).open("xb") as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=stream)
        commands.append(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd),
                "exit_code": result.returncode,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(name, result.returncode, flush=True)
        result.check_returncode()

    run(
        "configure",
        [str(source / "configure"), "--prefix=" + str(prefix), "--with-ensurepip=no"],
        source,
    )
    run("build", ["make", "-j" + str(args.jobs)], source)
    run("install", ["make", "install"], source)
    # Path-only reuse of already installed, hash-verified packages. No executable
    # .pth content and no modification of the scientific environment itself.
    package_path = prefix / "lib/python3.12/site-packages/replay_dependencies.pth"
    with package_path.open("x") as stream:
        stream.write("".join(str(path) + "\n" for path in packages))
    python = prefix / "bin/python3.12"
    check = (
        "import _posixsubprocess, bz2, ctypes, json, sqlite3, ssl, sys, sysconfig; "
        "from pathlib import Path; "
        "assert sys.version_info[:3] == (3,12,3); "
        "assert sysconfig.get_config_var('HAVE_CLOSE_RANGE') == 0; "
        "assert Path(_posixsubprocess.__file__).is_relative_to(Path(sys.base_prefix)); "
        "print(json.dumps({'python':sys.version,'HAVE_CLOSE_RANGE':0,"
        "'subprocess_extension':_posixsubprocess.__file__}))"
    )
    run("runtime", [str(python), "-B", "-c", check], output)
    receipt = {
        "complete": True,
        "scope": "original-CPython-configured-FD-closing-CPU-replay-runtime",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_modified": False,
        "configure_environment": {
            key: env[key] for key in ("ac_cv_func_close_range", *BUILD_ENVIRONMENT) if key in env
        },
        "python": str(python),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "scientific_package_directories": [str(path) for path in packages],
        "package_installation_performed": False,
        "scientific_replay_claimed": False,
        "system_environment_modified": False,
    }
    (output / "runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
