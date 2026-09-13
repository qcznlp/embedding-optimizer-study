"""Preserve the final exact-handle observations separately from the first archive."""
from pathlib import Path
import shutil
import resume as r

destination = Path('/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-gpu-resume-v1')
r.bound(destination / 'verification.json', 'd4fbc3a8367be9cc68cfc3f4a90e182d0788e640a69b4f4fb224e9f06c677f3f')
rows = []
for name in ('training-final.json', 'evaluation-final.json', 'probe-final.json', 'summary-final.json', 'resume-final.json', 'handoff.py'):
    source, target = r.HERE / name, destination / 'final-observations' / name
    value = r.identity(source)
    r.no_existing(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    r.need(r.identity(source) == r.identity(target) == value, 'Handoff copy differs')
    rows.append(dict(source=str(source), destination=str(target.relative_to(destination)), **value))
training = r.read(r.HERE / 'training-final.json')
resume = r.read(r.HERE / 'resume-final.json')
native = [v['complete'] for p in training['pools'].values() for v in p['runs'] if v.get('complete')]
backups = [r.read(p) for p in (r.BACKUP / 'run').glob('*/verified.json')]
r.need(len(native) == 8 and len(backups) == 8, 'This dated handoff must not infer new completions')
value = dict(recorded_at_utc=r.now(), training_observed_at_utc=training['observed_at_utc'], copies=rows,
    completed_branches=8, native_five_checkpoint_reads=40, immutable_hf_checkpoint_checks=40,
    actual_anonymous_restored_checkpoint_files=36, actual_anonymous_restored_checkpoint_bytes=3142951086,
    actual_gpu_resume_completed=0, new_recovery_coordinators={k:v['coordinator'] for k,v in resume['pools'].items()},
    ongoing_scientific_tasks_changed=False, scientific_completion=False, source_publication=False)
r.write(destination / 'handoff.json', value)
print(r.identity(destination / 'handoff.json'))
