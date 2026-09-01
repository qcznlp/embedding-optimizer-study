SHELL := /bin/bash

ACL_STYLE_COMMIT := d5adc823ff0f80f98c80405ca0ab66c68e684409
ACL_STYLE_BASE := https://raw.githubusercontent.com/acl-org/acl-style-files/$(ACL_STYLE_COMMIT)
VENDOR := vendor
BUILD := build
STYLE_FILES := $(VENDOR)/acl.sty $(VENDOR)/acl_natbib.bst
RESULT_TABLES := generated/discovery.tex generated/per-task.tex generated/common-state.tex \
	generated/representation.tex generated/intervention.tex generated/confirmation.tex \
	generated/diagnostics.tex generated/causal-chain.tex generated/candidate-breadth.tex \
	generated/corpus-size-diagnostic-appendix.tex \
	generated/retrieval-dynamics-extension.tex
RESULT_FIGURES := ../reports/dense-discovery/figures/dense-training-dynamics-by-run.png \
	../reports/dense-discovery/figures/dense-lr-sensitivity.png
EXTENDED_RESULT_FIGURE := $(wildcard ../reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.pdf)
CANDIDATE_BREADTH_FIGURE := $(wildcard ../reports/candidate-breadth/candidate_breadth_calibration.pdf)
CORPUS_SIZE_FIGURE := ../reports/corpus-size-diagnostic/corpus_size_association.pdf
PYTHON ?= python3

.PHONY: all release vendor headlines clean distclean

all: headlines $(BUILD)/main.pdf
	$(PYTHON) -m embed_optim.paper_layout --paper-dir . --max-main-page 8

release:
	$(MAKE) clean
	$(PYTHON) -m embed_optim.paper_results --repo-root .. \
		--families dense \
		--scope-amendment configs/dense_scope_amendment.json
	$(MAKE) $(BUILD)/main.pdf
	$(PYTHON) -m embed_optim.paper_layout --paper-dir . --max-main-page 8

vendor: $(STYLE_FILES)

headlines:
	$(PYTHON) -m embed_optim.paper_results --repo-root .. --if-ready \
		--families dense \
		--scope-amendment configs/dense_scope_amendment.json

$(VENDOR)/.ready:
	mkdir -p $(VENDOR)
	touch $@

$(VENDOR)/acl.sty: | $(VENDOR)/.ready
	curl --fail --location --silent --show-error $(ACL_STYLE_BASE)/acl.sty --output $@

$(VENDOR)/acl_natbib.bst: | $(VENDOR)/.ready
	curl --fail --location --silent --show-error $(ACL_STYLE_BASE)/acl_natbib.bst --output $@

$(BUILD)/main.pdf: main.tex results.tex $(RESULT_TABLES) $(RESULT_FIGURES) \
	$(EXTENDED_RESULT_FIGURE) $(CANDIDATE_BREADTH_FIGURE) $(CORPUS_SIZE_FIGURE) \
	references.bib $(STYLE_FILES)
	mkdir -p $(BUILD)
	TEXINPUTS=$(abspath $(VENDOR))//: BSTINPUTS=$(abspath $(VENDOR))//: \
		latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD) main.tex

clean:
	@if [ -d $(BUILD) ]; then \
		TEXINPUTS=$(abspath $(VENDOR))//: BSTINPUTS=$(abspath $(VENDOR))//: \
		latexmk -C -outdir=$(BUILD) main.tex; \
	fi

distclean: clean
	rm -rf $(BUILD) $(VENDOR)
