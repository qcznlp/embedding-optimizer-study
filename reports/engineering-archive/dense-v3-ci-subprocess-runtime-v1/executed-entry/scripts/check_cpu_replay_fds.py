"""Verify actual subprocess isolation under the configured CPU replay runtime."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def main() -> None:
    if sys.version_info[:3] != (3, 12, 3) or sysconfig.get_config_var("HAVE_CLOSE_RANGE") != 0:
        raise ValueError("Expected the original-source CPython per-FD closing build")
    import _posixsubprocess

    if not Path(_posixsubprocess.__file__).is_relative_to(Path(sys.base_prefix)):
        raise ValueError("Subprocess extension must belong to this interpreter")
    records = []
    with tempfile.TemporaryFile() as marker:
        keep_fd = fcntl.fcntl(marker.fileno(), fcntl.F_DUPFD, 80)
        drop_fd = fcntl.fcntl(marker.fileno(), fcntl.F_DUPFD, 80)
        os.set_inheritable(keep_fd, True)
        os.set_inheritable(drop_fd, True)
        expected = os.fstat(keep_fd)
        try:
            for keep in (False, True):
                program = """import errno, os, sys
keep, keep_fd, drop_fd, device, inode = map(int, sys.argv[1:])
for fd in ([drop_fd] if keep else [keep_fd, drop_fd]):
    try:
        os.fstat(fd)
    except OSError as error:
        assert error.errno == errno.EBADF
    else:
        raise AssertionError('Unwanted inheritable descriptor leaked')
if keep:
    assert (os.fstat(keep_fd).st_dev, os.fstat(keep_fd).st_ino) == (device, inode)
print('Required descriptor isolation verified')
"""
                command = [sys.executable, "-B", "-c", program] + [
                    str(value)
                    for value in (int(keep), keep_fd, drop_fd, expected.st_dev, expected.st_ino)
                ]
                result = subprocess.run(
                    command,
                    close_fds=True,
                    pass_fds=(keep_fd,) if keep else (),
                    start_new_session=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                row = {
                    "keep_one_explicit_fd": keep,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                records.append(row)
                print(json.dumps(row), flush=True)
                result.check_returncode()
        finally:
            os.close(keep_fd)
            os.close(drop_fd)
    print(
        json.dumps(
            {"complete": True, "scope": "actual-owned-subprocess-FD-isolation", "cases": records}
        )
    )


if __name__ == "__main__":
    main()
