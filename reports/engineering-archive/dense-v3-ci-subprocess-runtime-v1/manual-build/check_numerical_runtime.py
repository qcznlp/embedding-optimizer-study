"""Verify a configured interpreter uses the unchanged numerical packages."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import sysconfig

assert sys.version_info[:3] == (3, 12, 3)
assert sysconfig.get_config_var('HAVE_CLOSE_RANGE') == 0
import _posixsubprocess
import bz2
import ctypes
import sqlite3
import ssl
import numpy
import scipy
import torch
import flash_attn_2_cuda

assert Path(_posixsubprocess.__file__).is_relative_to(Path(sys.base_prefix))
assert torch.__version__=='2.9.1+cu129' and torch.version.cuda=='12.9'
assert metadata.version('huggingface-hub')=='1.28.0'
assert numpy.__version__=='2.5.2'
assert all(callable(getattr(flash_attn_2_cuda,n)) for n in ['fwd','varlen_fwd','bwd','varlen_bwd','fwd_kvcache'])
print(json.dumps({'complete':True,'python':sys.version,'prefix':sys.base_prefix,
                  'subprocess_extension':_posixsubprocess.__file__,
                  'HAVE_CLOSE_RANGE':sysconfig.get_config_var('HAVE_CLOSE_RANGE'),
                  'packages':{n:metadata.version(n) for n in ['torch','numpy','scipy','pandas','huggingface-hub','flash-attn']},
                  'paths':{n:m.__file__ for n,m in [('torch',torch),('numpy',numpy),('scipy',scipy)]}},indent=2))
