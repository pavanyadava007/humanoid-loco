# All targets use the project venv created by scripts/setup_env.sh.
PY ?= .venv/bin/python
STEPS ?= 100000000
CAP ?= 125
EPISODES ?= 500

.PHONY: setup train train-dr train-nodr train-rsl export eval eval-mjx render parity isaac report test lint \
	isaac-setup isaac-train isaac-train-rough isaac-play isaac-play-rough isaac-eval isaac-eval-rough space

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

# Isaac Lab track: separate venv, NVIDIA Omniverse EULA accepted through OMNI_KIT_ACCEPT_EULA=YES
IPY ?= .venv-isaac/bin/python
ISAAC_ENV = OMNI_KIT_ACCEPT_EULA=YES

isaac-setup:
	bash scripts/isaac/setup_isaac.sh

isaac-train:
	$(ISAAC_ENV) $(IPY) scripts/isaac/train_g1.py --headless --task Isaac-Velocity-Flat-G1-v0 --name g1_flat --num_envs 4096 --seed 42 --cap-minutes 115

isaac-train-rough:
	$(ISAAC_ENV) $(IPY) scripts/isaac/train_g1.py --headless --task Isaac-Velocity-Rough-G1-v0 --name g1_rough --num_envs 4096 --seed 42 --cap-minutes 110

isaac-play:
	$(ISAAC_ENV) $(IPY) scripts/isaac/play_export.py --headless --enable_cameras --name g1_flat
	$(PY) scripts/isaac/make_video.py --name g1_flat
	$(PY) scripts/isaac/compare_models.py --name g1_flat

isaac-eval:
	$(ISAAC_ENV) $(IPY) scripts/isaac/eval_g1.py --all --name g1_flat --episodes 500
	$(PY) scripts/isaac/record_install.py

isaac-play-rough:
	$(ISAAC_ENV) $(IPY) scripts/isaac/play_export.py --headless --enable_cameras --task Isaac-Velocity-Rough-G1-Play-v0 --name g1_rough
	$(PY) scripts/isaac/make_video.py --name g1_rough

isaac-eval-rough:
	$(ISAAC_ENV) $(IPY) scripts/isaac/eval_g1.py --all --name g1_rough --task Isaac-Velocity-Rough-G1-v0 --policy checkpoints/isaac_g1_rough/policy.pt --episodes 500

space:
	$(PY) scripts/build_space.py

report:
	$(PY) scripts/make_report.py

test:
	JAX_PLATFORMS=cpu $(PY) -m pytest -q tests

lint:
	$(PY) -m ruff check .
