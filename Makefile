# All targets use the project venv created by scripts/setup_env.sh.
PY ?= .venv/bin/python
STEPS ?= 100000000
CAP ?= 125
EPISODES ?= 500

.PHONY: setup train train-dr train-nodr train-rsl export eval eval-mjx render parity isaac report test lint

setup:
	bash scripts/setup_env.sh

train: train-dr train-nodr train-rsl

train-dr:
	$(PY) scripts/train_brax.py --name brax_dr --dr --num-timesteps $(STEPS) --cap-minutes $(CAP)

train-nodr:
	$(PY) scripts/train_brax.py --name brax_nodr --no-dr --num-timesteps $(STEPS) --cap-minutes $(CAP)

train-rsl:
	$(PY) scripts/train_rsl_rl.py --name rsl_dr --cap-minutes 75

export:
	for p in brax_dr brax_nodr rsl_dr; do $(PY) scripts/export_onnx.py --policy $$p; done

parity:
	$(PY) scripts/check_parity.py

eval-mjx:
	for p in brax_dr brax_nodr rsl_dr; do $(PY) scripts/eval_mjx.py --policy $$p; done

eval: export parity eval-mjx
	for p in brax_dr brax_nodr rsl_dr; do $(PY) scripts/eval_sim2sim.py --policy $$p --episodes $(EPISODES); done

render:
	for p in brax_dr brax_nodr rsl_dr; do $(PY) scripts/render.py --policy $$p; done

isaac:
	$(PY) scripts/isaac_lab_check.py

report:
	$(PY) scripts/make_report.py

test:
	JAX_PLATFORMS=cpu $(PY) -m pytest -q tests

lint:
	$(PY) -m ruff check .
