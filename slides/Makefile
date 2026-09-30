PORT ?= 4187
DECK ?= Lecture_1B
DECK_DIR := lectures/$(DECK)
DECK_SOURCE := $(DECK_DIR)/$(DECK).qmd
DECK_OUTPUT := $(DECK_DIR)/$(DECK)
LECTURE_SOURCES := $(wildcard lectures/Lecture_*/*.qmd)

.PHONY: html pdf serve live open-html start-julia-server clean clean-pdf lectures lecture-list

lectures:
	@for source in $(LECTURE_SOURCES); do quarto render $$source || exit 1; done

lecture-list:
	@for source in $(LECTURE_SOURCES); do basename $$(dirname $$source); done

html: $(DECK_OUTPUT).html

$(DECK_OUTPUT).html: $(DECK_SOURCE) styles.css lecture-styles.css scripts/frames.html scripts/widget.html $(shell find $(DECK_DIR) -type f ! -name '$(DECK).html' ! -name '$(DECK).pdf' ! -path '*_files/*' ! -path '*/.quarto/*' ! -path '*/output/*' ! -path '*/.venv/*' ! -path '*/__marimo__/*' ! -name '* *' 2>/dev/null)
	quarto render $(DECK_SOURCE)

# 1920×1080 pages (16:9). Renders HTML, then screenshots via Chrome.
pdf: $(DECK_OUTPUT).pdf

$(DECK_OUTPUT).pdf: $(DECK_OUTPUT).html styles.css scripts/export_reveal_pdf.zsh scripts/annotate_pdf_links.py
	PORT=$(PORT) DECK=$(DECK_OUTPUT) zsh scripts/export_reveal_pdf.zsh

serve: html
	python3 -m http.server $(PORT) --bind 127.0.0.1

live:
	quarto preview $(DECK).qmd --port $(PORT) --host 127.0.0.1

open-html: html
	open $(DECK_OUTPUT).html

# Live Julia cells: serves the Pluto notebooks listed in $(DECK_DIR)/pluto.toml on port 1243.
start-julia-server:
	@test -f $(DECK_DIR)/pluto.toml || { echo "$(DECK) has no Pluto notebooks ($(DECK_DIR)/pluto.toml)"; exit 1; }
	julia scripts/serve_pluto.jl $(DECK)

clean-pdf:
	rm -f $(DECK_OUTPUT).pdf
	rm -rf qa/pdf-slides

clean: clean-pdf
	rm -f $(DECK_OUTPUT).html
	rm -rf $(DECK_OUTPUT)_files $(DECK_DIR)/.quarto
