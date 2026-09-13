SHELL := /bin/bash
.DEFAULT_GOAL := current

PYTHON ?= python3
export PYTHONPATH := $(abspath ../src)$(if $(PYTHONPATH),:$(PYTHONPATH))
CURRENT_OUTPUT ?= $(abspath build/current-document)
NUMERICAL_BUNDLE ?=
RELEASE_OUTPUT ?= $(abspath build/complete-reproduction)

.PHONY: all current release legacy-all legacy-release

# The default is the authenticated complete-result manuscript, not the old draft.
all: current

current:
	mkdir -p "$(dir $(CURRENT_OUTPUT))"
	CUDA_VISIBLE_DEVICES='' $(PYTHON) -m embed_optim.current_paper \
		--paper-dir current --output "$(CURRENT_OUTPUT)"

# This release-artifact gate performs the complete numerical/document join.
# It never publishes remotely or overwrites an existing attempt.
release:
	@test -n "$(NUMERICAL_BUNDLE)" || { echo 'Set NUMERICAL_BUNDLE to the accepted closed numerical bundle.' >&2; exit 2; }
	mkdir -p "$(dir $(RELEASE_OUTPUT))"
	CUDA_VISIBLE_DEVICES='' $(PYTHON) -m embed_optim.paper_reproduction \
		--numerical-bundle "$(NUMERICAL_BUNDLE)" \
		--paper-dir "$(abspath current)" --output "$(RELEASE_OUTPUT)"

# Original source/guards remain available explicitly, with their historical scope.
legacy-all:
	$(MAKE) -f legacy.Makefile MAKE='$(MAKE) -f legacy.Makefile' all

legacy-release:
	$(MAKE) -f legacy.Makefile MAKE='$(MAKE) -f legacy.Makefile' release
