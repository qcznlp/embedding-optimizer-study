"""Seal actual data-only transport/recovery and original-source preservation evidence."""
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).parents[1]
STORY = Path('/root/embedding-optimizer-story-refactor')
WORK = Path('/tmp/dense-v3-prediction-artifact-backup.Aq3Smtuj')


def main():
    output = ARCHIVE/'verification.json'
    if output.exists():
        raise ValueError('Preserve existing archive verification')
    spec = importlib.util.spec_from_file_location('prediction_archive_backup', ARCHIVE/'source/backup.py')
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    t = b.load_transport()
    need = b.require
    preflight, expected = b.prepared(t, WORK)
    audit = t.read(ARCHIVE/'actual/remote-audit.json')
    recovered = t.read(ARCHIVE/'actual/download-verified.json')
    staged = t.read(ARCHIVE/'actual/staged-independent.json')
    replay = t.read(ARCHIVE/'actual/isolated-reconstruction.json')
    modes = t.read(ARCHIVE/'actual/upload-mode-preflight.json')
    b.check_modes(modes['modes'], preflight['prefix'])
    need(audit['revision'] == recovered['revision'] == '42689be00e1644aaf24721ecc239ec1c4d425a2a', 'Wrong revision')
    need(preflight['upload_files'] == audit['files'] == recovered['files'] == replay['files'] == 47,
         'Incomplete transport population')
    need(preflight['upload_bytes'] == recovered['bytes'] == replay['bytes'] == 12301350, 'Byte totals differ')
    need(audit['durability_verified'] is True and recovered['all_payload_hashes_match'] is True
         and audit['source_code_uploaded'] is False and audit['remote_paths_deleted_or_overwritten'] is False
         and audit['root_attributes_unchanged'] is True and audit['old_root_entries_unchanged'] == 20
         and audit['old_corrected_subtrees_unchanged'] == 14, 'Remote preservation scope differs')
    need(preflight['manifest'] == audit['manifest'] == recovered['manifest']
         and preflight['manifest']['sha256'] == replay['manifest_sha256']
         == '52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d', 'Manifest binding differs')
    need({k:v for k,v in staged.items() if k not in ('snapshot_root','verified_at_utc')}
         == {k:v for k,v in replay.items() if k not in ('snapshot_root','verified_at_utc')},
         'Recovered numerical read differs from staged read')
    need([replay[k] for k in ('locked_predictions','sensitivity_predictions','exact_fold_errors_reconstructed',
         'pooled_comparisons_reconstructed','plot_points_matched','csv_rows_matched')] == [840,5040,392,98,98,6875],
         'Reconstruction coverage differs')
    need(replay['only_supplied_snapshot_read'] is True and replay['numerical_fitting_or_new_bootstrap'] is False
         and replay['model_or_network_execution'] is False and replay['scientific_completion'] is False,
         'Misstated recovered read scope')
    root = Path(recovered['downloaded_root'])
    need(root == WORK/'download'/preflight['prefix'] and replay['snapshot_root'] == str(root), 'Recovery location differs')
    for name, item in expected.items():
        for base in (root, ARCHIVE/'snapshot'):
            t.compare_file(base/t.safe_name(name), item)
    t.compare_remote(expected, audit['remote_inventory'])
    for path in (WORK/'source').iterdir():
        need(t.file_identity(path) == t.file_identity(ARCHIVE/'source'/path.name), 'Archived source differs')
    reader = t.file_identity(ARCHIVE/'source/verify_recovered.py')
    need(reader['sha256'] == 'cda3c8859f1797044dfe172704ece602c063047edf91ec44490edd3e266806ff', 'Reader source differs')
    need(reader == t.file_identity(Path('/tmp/dense-v3-prediction-recovered-reader.vqV0PWBO/verify_recovered.py')),
         'Isolated copied reader differs')
    tests = t.read(ARCHIVE/'actual/tests.json')
    need([tests[k] for k in ('tests','failures','errors','skipped')] == [18,0,0,0]
         and tests['remote_mutations'] is False and tests['model_execution'] is False, 'Focused tests failed')
    original_counts = {}
    for parent, sha in ((b.ORIGINAL,b.VERIFICATION_SHA),(b.SENSITIVITY,b.SENSITIVITY_SHA)):
        receipt = t.read(parent/'verification.json',sha)
        for name,item in receipt['payloads'].items():
            need(Path(name).name != 'gpu.py', 'Protected helper is out of scope')
            t.compare_file(parent/t.safe_name(name),item)
        original_counts[parent.name] = len(receipt['payloads'])
    need(original_counts == {'dense-v3-weight-retrieval-v1':226,'dense-v3-predictor-sensitivity-v1':50}, 'Parent population differs')
    protected = t.read(b.ORIGINAL/'inputs/outcomes/verification.json')['protected_inputs']
    for name, item in protected.items():
        need(Path(name).name != 'gpu.py', 'Protected helper is out of scope')
        t.compare_file(Path(name), item)
    primary = Path('/root/embedding-optimizer-primary-v3')
    assembly = t.read(primary/'source-assembly.json','e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8')
    count = 0
    for base in (primary,Path('/root/embedding-optimizer-v3-experiment/launch/source-snapshot')):
        for name, item in assembly['files'].items():
            need(Path(name).name != 'gpu.py', 'Protected helper is out of scope')
            t.compare_file(base/t.safe_name(name),item['identity']); count += 1
    need(count == 112 and len(protected) == 11, 'Original source/protected population differs')
    need(t.file_identity(STORY/'paper/main.tex')['sha256'] == '45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e',
         'Manuscript changed')
    need(t.file_identity(ARCHIVE/'before/CURRENT_EXPERIMENT.md')['sha256'] == '99c0030e4159c2be79211cbb4551ac9eaf0192b788a263fd084963229c1b1bf5',
         'Handoff preimage differs')
    payloads = {}
    for path in sorted(ARCHIVE.rglob('*')):
        need(not path.is_symlink(), 'Symlinked archive')
        if path.is_file():
            need(t.SECRETS.search(path.read_bytes()) is None, 'Credential-shaped archive content')
            payloads[path.relative_to(ARCHIVE).as_posix()] = t.file_identity(path)
    guide = STORY/'docs/predictor-analysis-restoration.md'
    links = 0
    for path in (ARCHIVE/'README.md',guide):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'): continue
            resolved = (path.parent/target.split('#',1)[0]).resolve()
            need(resolved.exists() or resolved == output, 'Broken local handoff link'); links += 1
    value = dict(scope='actual-prediction-data-backup-and-recovered-error-verification',
        verified_at_utc=datetime.now(timezone.utc).isoformat(), revision=audit['revision'],
        files=47, bytes=12301350, manifest=preflight['manifest'],
        anonymous_complete_recovery=True, independent_recovered_reconstruction=t.file_identity(ARCHIVE/'actual/isolated-reconstruction.json'),
        locked_predictions=840, exploratory_predictions=5040, exact_fold_errors=392, pooled_comparisons=98,
        plotted_comparisons=98, original_archive_payloads_unchanged=original_counts,
        original_source_assembly_files_unchanged=count, protected_inputs_unchanged=len(protected),
        tests=tests, source_reader=reader, restoration_guide=t.file_identity(guide),
        local_links_checked=links, credential_findings=0, payloads=payloads,
        old_root_entries_unchanged=20, old_corrected_subtrees_unchanged=14, root_attributes_unchanged=True,
        source_published=False, old_remote_files_deleted_or_overwritten=False, new_inference=False,
        functional_recovery=False, model_or_gpu_execution=False, manuscript_modified=False,
        scientific_completion=False)
    t.write_new(output,value)
    print(json.dumps({k:v for k,v in value.items() if k != 'payloads'},sort_keys=True))


if __name__ == '__main__': main()
