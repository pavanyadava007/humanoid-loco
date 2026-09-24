"""Pure-Python helpers of the Isaac Lab track (importable without Isaac Sim)."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from hloco.common import ROOT
from hloco.stats import wilson


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "isaac" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_eval_summary_counts_and_wilson():
    ev = _load("eval_g1")
    eps = [{"fell": i < 3, "survival_s": 5.0 if i < 3 else 20.0, "lin_vel_err": 0.1 * i, "yaw_rate_err": 0.2,
            "ended_low": i == 9} for i in range(10)]
    s = ev.summarize(eps)
    assert s["episodes"] == 10 and s["falls"] == 3
    assert s["fall_rate"] == pytest.approx(0.3)
    assert s["fall_rate_wilson95"] == pytest.approx(list(wilson(3, 10)))
    assert s["lin_vel_err_mean"] == pytest.approx(0.45)
    assert s["survival_s_mean"] == pytest.approx((3 * 5 + 7 * 20) / 10)
    assert s["ended_low_count"] == 1


def test_eval_conditions_unique_and_nominal_first():
    ev = _load("eval_g1")
    names = [c.name for c in ev.CONDITIONS]
    assert names[0] == "nominal" and len(names) == len(set(names))


def test_compare_joint_sets():
    cm = _load("compare_models")
    same = cm.compare(["a", "b"], ["a", "b"])
    assert same["one_to_one"] and same["same_order_for_common"]
    diff = cm.compare(["a", "b", "x"], ["b", "a", "y"])
    assert not diff["one_to_one"]
    assert diff["isaac_only"] == ["x"] and diff["mujoco_only"] == ["y"]
    assert not diff["same_order_for_common"]


def test_warmup_trim():
    mv = _load("make_video")
    assert mv.warmup_frames([16.0, 184.0, 133.6, 134.3, 135.7, 136.1]) == 2
    assert mv.warmup_frames([130.0, 131.0, 129.0]) == 0
    assert mv.warmup_frames([]) == 0
