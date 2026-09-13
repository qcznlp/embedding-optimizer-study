"""Observe a pending development document without granting publication admission."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from embed_optim import primary_v3_manuscript as manuscript
from embed_optim import paper_layout


def identity(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not a.output.is_absolute() or a.output.exists():
        raise ValueError('New absolute observation required')
    main = (a.paper / 'main.tex').read_text()
    includes = [(a.paper / name).read_text() for name in manuscript.RESULT_FILES]
    observations = {}
    try:
        observations['original_abstract_reader'] = manuscript.abstract_readout(main, includes)
    except ValueError as error:
        observations['original_abstract_reader'] = {'passed': False, 'error': str(error)}
    try:
        observations['original_frozen_float_layout_reader'] = paper_layout.audit_paper_layout(a.paper)
    except ValueError as error:
        observations['original_frozen_float_layout_reader'] = {'passed': False, 'error': str(error)}
    # Explicit pending draft text is unwrapped only for counting its visible words.
    # This is not submitted to or accepted by the original strict reader.
    declared = {}
    for content in includes:
        declared.update(manuscript.definitions(content))
    macros = {}
    for name in manuscript.ABSTRACT_MACROS:
        definition, = declared[name]
        value = definition['body'].strip()
        if value.startswith(r'\ResultPending'):
            body, end = manuscript.brace_group(value, len(r'\ResultPending'))
            if value[end:].strip():
                raise ValueError('Unexpected trailing draft finding')
            value = '[PENDING: ' + body + ']'
        macros[name] = value
    raw, = re.findall(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main, re.DOTALL)
    expanded = manuscript.expand_plain(raw, macros)
    observations['visible_draft_abstract'] = {'text': expanded, 'words_conservative': len(manuscript.legacy.ABSTRACT_WORD_PATTERN.findall(expanded)),
        'pending_findings_explicitly_counted_as_visible_text_only': True, 'final_abstract_admission': False}
    aux_path = a.paper / 'build/main.aux'
    aux = aux_path.read_text()
    observations['actual_compiled_float_pages'] = {name: paper_layout._page_for_unique_label(aux, name, aux_path)
        for name in sorted(paper_layout._observed_float_labels(aux) | {'paper-main-end'})}
    log = (a.paper / 'build/main.log').read_text()
    observations['final_log_warnings'] = [line for line in log.splitlines() if any(s in line for s in ('Overfull ', 'There were undefined references', 'There were undefined citations'))]
    for tool in ('pdfinfo', 'pdffonts'):
        run = subprocess.run([tool, str(a.paper / 'build/main.pdf')], capture_output=True, text=True, check=True)
        observations[tool] = run.stdout
    observations['inputs'] = {str(path.relative_to(a.paper)): identity(path) for path in sorted(a.paper.rglob('*')) if path.is_file() and path.suffix in ('.tex', '.bib', '.sty', '.bst', '.pdf')}
    observations.update({'scope': 'actual_development_paper_observation', 'factorial_results_pending': True,
        'old_readers_changed': False, 'manuscript_installed': False, 'final_publication_gate_passed': False,
        'scientific_completion': False})
    with a.output.open('x') as stream:
        stream.write(json.dumps(observations, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k:v for k,v in observations.items() if k not in ('inputs', 'pdfinfo', 'pdffonts')}))


if __name__ == '__main__':
    main()
