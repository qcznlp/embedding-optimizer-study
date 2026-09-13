"""Actual owned subprocess FD-isolation controls; no scientific code imported."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

assert sys.version_info[:3] == (3, 12, 3)
assert sysconfig.get_config_var('HAVE_CLOSE_RANGE') == 0
work=Path(__file__).resolve().parent
marker=tempfile.TemporaryFile(dir=work)
os.dup2(marker.fileno(), 88, inheritable=True)
os.dup2(marker.fileno(), 87, inheritable=True)
identity=(os.fstat(87).st_dev,os.fstat(87).st_ino)
commands=[]
try:
    for keep in (False, True):
        code='''import errno, os, sys
for fd in ([88] if sys.argv[1]=='keep' else [87,88]):
    try: os.fstat(fd)
    except OSError as e: assert e.errno == errno.EBADF
    else: raise AssertionError('unwanted inheritable descriptor leaked')
if sys.argv[1]=='keep':
    assert (os.fstat(87).st_dev,os.fstat(87).st_ino)==(int(sys.argv[2]),int(sys.argv[3]))
print('required FD isolation verified')
'''
        argv=[sys.executable,'-B','-c',code,'keep' if keep else 'close',*[str(x) for x in identity]]
        result=subprocess.run(argv,close_fds=True,pass_fds=(87,) if keep else (),
                              start_new_session=True,capture_output=True,text=True,timeout=60)
        commands.append({'keep_one_explicit_fd':keep,'returncode':result.returncode,
                         'stdout':result.stdout,'stderr':result.stderr})
        print(json.dumps(commands[-1]),flush=True)
        result.check_returncode()
finally:
    os.close(87)
    os.close(88)
    marker.close()
print(json.dumps({'complete':True,'scope':'owned-CPU-subprocess-FD-isolation-only',
                  'python':sys.version,'close_fds':True,'HAVE_CLOSE_RANGE':0,
                  'unwanted_fds_closed':True,'explicit_pass_fds_preserved':True,
                  'subprocess_source_modified':False,'scientific_source_modified':False}),flush=True)
