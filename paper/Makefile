SHELL := /bin/bash

ACL_STYLE_COMMIT := d5adc823ff0f80f98c80405ca0ab66c68e684409
ACL_STYLE_BASE := https://raw.githubusercontent.com/acl-org/acl-style-files/$(ACL_STYLE_COMMIT)
VENDOR := vendor
BUILD := build
STYLE_FILES := $(VENDOR)/acl.sty $(VENDOR)/acl_natbib.bst
PYTHON ?= python3

.PHONY: all vendor headlines clean distclean

all: headlines $(BUILD)/main.pdf

vendor: $(STYLE_FILES)

headlines:
	$(PYTHON) -m embed_optim.paper_results --repo-root .. --if-ready

$(VENDOR)/.ready:
	mkdir -p $(VENDOR)
	touch $@

$(VENDOR)/acl.sty: | $(VENDOR)/.ready
	curl --fail --location --silent --show-error $(ACL_STYLE_BASE)/acl.sty --output $@

$(VENDOR)/acl_natbib.bst: | $(VENDOR)/.ready
	curl --fail --location --silent --show-error $(ACL_STYLE_BASE)/acl_natbib.bst --output $@

$(BUILD)/main.pdf: main.tex results.tex references.bib $(STYLE_FILES)
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
