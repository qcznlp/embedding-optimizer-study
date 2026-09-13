"""Append final dated observations without changing the original archive manifest."""
from pathlib import Path
import author as a

report = a.STORY / 'reports/engineering-archive/dense-v3-complete-document-handoff-v1'
output = report / 'final-observations'
output.mkdir(exist_ok=False)
bindings = {}
for name in ('evaluation-final.json', 'document-final.json', 'summary-final.json', 'resume-midpoint.json'):
    source = a.HERE / name
    a.copy_file(source, output / name, a.identity(source))
    bindings[str((output / name).relative_to(report))] = a.identity(output / name)
for name in ('AGENTS.md', 'CURRENT_EXPERIMENT.md', 'PROJECT_STATUS.md', 'README.md'):
    source = a.STORY / name
    a.copy_file(source, output / ('current-' + name), a.identity(source))
    bindings[str((output / ('current-' + name)).relative_to(report))] = a.identity(output / ('current-' + name))
for source in (a.HERE / 'author.py', a.HERE / 'document_component.py', a.HERE / 'authorization.json'):
    bindings[str(source)] = a.identity(source)
a.bound(a.STORY / 'paper/main.tex', '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e')
a.bound(report / 'archive-manifest.json', '60dd7bb39021af1dc7790851eb18405766f759f9cd9a6e0113918ad3004e84c4')
a.write(report / 'handoff.json', {'recorded_at_utc': a.now(), 'source': a.identity(__file__),
    'phase_outcome': 'PROGRESS', 'full_goal_complete': False,
    'primary_training_complete_runs': 12, 'factorial_training_complete_runs': 12,
    'primary_beir_complete_units': 840, 'factorial_beir_complete_units_at_observation': 2,
    'factorial_checkpoint_probes_complete': 60, 'all_checkpoint_hf_backups_complete': 120,
    'real_factorial_inference_complete': False, 'real_complete_paper_generated': False,
    'document_waiter': {'session': 65801, 'pid': 850717, 'start_ticks': 321699320},
    'summary_waiter': {'session': 4529, 'pid': 813139, 'start_ticks': 320676124},
    'completed_probe_tool_exits': {'54795': {'exit_code': 0, 'terminal': '066902'},
                                   '34578': {'exit_code': 0, 'terminal': '992eba'}},
    'all_eight_beir_workers_active_at_observation': True,
    'original_numerical_sources_and_authoritative_manuscript_unchanged': True,
    'new_complete_document_consumer_source_frozen': True,
    'old_gates_modified': False, 'protected_helper_accessed': False,
    'new_git_or_remote_source_write': False, 'gpu_resume_complete': False,
    'copied_source_full_paper_reconstruction_complete': False,
    'remaining': ['existing complete BEIR panel', 'genuine complete original inference',
        'actual full document generation and visual review', 'new factorial probe/result data durability',
        'queued genuine GPU resumes', 'complete current source/data paper reconstruction',
        'tracking/source/package/release audits and final publication'], 'bindings': bindings})
print({'handoff': str(report / 'handoff.json'), 'binding': a.identity(report / 'handoff.json'),
       'full_goal_complete': False}, flush=True)
