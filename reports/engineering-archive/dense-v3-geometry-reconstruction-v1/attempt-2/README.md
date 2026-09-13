# First complete focused run — test helper failed

Owned session 93840 is terminal/exit 1. Of 63 cases, 62 passed; the foreign-source
negative test tried `copy.copy` on a Python module and raised `TypeError` before
the archive source checker ran. `focused-failed.xml` and `source/` preserve the
exact source at that attempt. Production reconstruction and the source checker
did not fail and were not changed to accommodate the test.

The helper was corrected to construct a `ModuleType` with an explicitly foreign
file. Four independent-oracle negative controls were also added. The resulting
67-case retry and full regression require their own terminal evidence.

The failed test's complete synthetic input remains at
`/tmp/dense-v3-geometry-failed-tests.myLpUE/focused-fixture/fixture/archive`,
outside automatic pytest cleanup, with the original external test anchor
`7785d0bdba2fe406066facad11ba1a9100e004c41cb721ab1ee8d92bbb3ec9db`.
There are no actual model payloads or primary findings in that fixture.
