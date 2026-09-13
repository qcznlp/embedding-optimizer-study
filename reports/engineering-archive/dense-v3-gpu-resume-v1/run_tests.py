"""Record actual focused test execution; never used as a scientific result."""
import os
from pathlib import Path
import unittest
import resume as r

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('CPU controls only')
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_resume.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
r.write(r.HERE / 'tests.json', dict(executed_at_utc=r.now(), tests_passed=result.testsRun,
    exit_code=0 if result.wasSuccessful() else 1, failures=len(result.failures), errors=len(result.errors),
    skipped=len(result.skipped), source=r.identity(r.HERE / 'resume.py'),
    test_source=r.identity(r.HERE / 'test_resume.py'), scientific_results=False))
raise SystemExit(0 if result.wasSuccessful() else 1)
