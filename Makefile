# Reproduce the paper.
#
#   make reproduce     every step below, in order: raw data to paper
#   make               figures and paper from the shipped artifacts only
#
# One target per step; no target reruns the steps before it.
#
#   make download      METR data -> data/raw/ (the live results file wins)
#   make data          data/raw/ -> data/processed/canonical.csv
#   make fits          fit the paper's specs -> artifacts/canonical/fits/
#                      (SPECS=all for the full 52-spec grid)
#   make eval          5-fold CV predictions and metrics -> artifacts/canonical/cv/
#                      (~3 min on 8 cores: every spec refit on 10 folds)
#   make eval-standin  in-sample stand-in for eval (fast; its figures are watermarked)
#   make scores        the paper's scoring-rule suite -> artifacts/canonical/scores/
#   make figures       artifacts -> figs/
#   make paper         figs/ -> paper/main.pdf
#   make arxiv         paper/ -> arxiv_submission/ (+ .tar.gz), comments stripped
#   make test          fast checks (about a minute)
#
# EM_ACCEL=squarem (default) or none chooses how fits and eval run EM for
# the IRT specs: SQUAREM-accelerated or plain.  The shipped artifacts were
# made with the defaults, so `make data fits eval scores figures paper`
# reproduces them.  Python runs as `uv run python`; set PY to use another
# interpreter.

PY      ?= uv run python
SPECS   ?= paper
NJOB    ?= 8
EM_ACCEL ?= squarem

FIGURE_SCRIPTS = make_headline make_difficulty_vs_length \
                 make_score_figures make_success_curves \
                 make_model2_over_metr_plot make_conversion_future

.PHONY: all reproduce download data fits eval eval-standin scores figures \
        paper arxiv test

all: figures paper

reproduce: download data fits eval scores figures paper

download:
	$(PY) pipeline/download_data.py

data:
	$(PY) pipeline/build_data.py

fits:
	$(PY) pipeline/fit.py --specs $(SPECS) --njob $(NJOB) --em-accel $(EM_ACCEL)

eval:
	$(PY) pipeline/evaluate.py --mode cv --specs $(SPECS) --njob $(NJOB) \
		--em-accel $(EM_ACCEL)

eval-standin:
	$(PY) pipeline/evaluate.py --mode insample --specs $(SPECS) --njob $(NJOB)

scores:
	$(PY) pipeline/score.py --check

figures:
	for s in $(FIGURE_SCRIPTS); do $(PY) make_figs/$$s.py || exit 1; done

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

arxiv:
	$(PY) paper/make_arxiv.py

test:
	$(PY) tests/test_fast.py
