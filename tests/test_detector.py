"""What the detector gets right, and the cases where it is fooled (negative controls)."""
from dataclasses import fields, replace

import numpy as np
import pytest

from precool_pulse.detect import diagnose
from precool_pulse.physics import CP_AIR, RHO_AIR
from precool_pulse.physics import Config
from precool_pulse.scenarios import SCENARIOS, _simulate_named, evaluate, run
from precool_pulse.sensing import Observation, SensorConfig, observe
from precool_pulse.study import noise_study, pulp_probe_study

ALL = list(range(8))
PERFECT = dict(sigma_t=0.0, sigma_v=0.0, lag_s=0.0)


def test_normal_batch_passes():
    r = run("normal")
    assert r["diag"]["status"] == "PASS" and r["truly_slow"] == []


def test_blockage_found_and_temperature_alone_misses_it():
    r = run("blockage")
    d = r["diag"]
    assert d["labels"][4] == "RESTRICTED-LIKE" and d["flagged"] == [4]
    assert d["uci_only"] == [4]
    assert d["temp_only"] == []          # the starved zone shows a HIGHER air temperature rise
    assert r["truly_slow"] == [4]        # independent check: its produce really cools >10% slower


def test_carton_bypass_found():
    r = run("package_bypass")
    assert r["diag"]["flagged"] == [2] and r["truly_slow"] == [2]


@pytest.mark.parametrize("name", ["bypass_gap", "package_bypass", "blockage"])
def test_with_perfect_mixing_uci_equals_true_heat_removal(name):
    """Conservation argument: rho * area * cp * speed * (T_return - T_supply) = heat leaving the produce."""
    res = _simulate_named(name)
    obs = observe(res, SensorConfig(**PERFECT))
    heat = RHO_AIR * res.cfg.zone_area * CP_AIR * obs.v_true * (obs.t_return_true - res.cfg.t_supply)
    assert np.allclose(heat, res.q[: len(obs.t)], rtol=1e-9)


def test_gap_bypass_is_located_by_the_signature_but_not_by_uci():
    r = run("bypass_gap")
    d = r["diag"]
    assert d["labels"][0] == "BYPASS-LIKE" and d["flagged"] == [0]
    assert r["truly_slow"] == ALL                    # the gap drops pressure for every zone
    assert d["uci_only"] == []                       # UCI cannot point at the gap, even with sensor noise
    exact = run("bypass_gap", noise=False)["diag"]["r_uci"]
    assert np.allclose(exact, 1.0, atol=1e-9)        # without noise UCI is identical in all eight zones
    assert d["batch_low"]                            # only the whole-batch check sees the loss


def test_small_gap_raises_alarm_although_cooling_is_barely_slower():
    r = run("bypass_gap_small")
    assert r["diag"]["flagged"] == [0] and r["truly_slow"] == []
    assert r["t78_ratio"].max() < 1.03


# ---- negative controls: the detector is expected to be wrong here ----

def test_mixed_return_air_blames_a_neighbouring_zone():
    r = run("mixed_return")
    assert r["diag"]["flagged"] == [0, 1]            # the gap is only at zone 1 (index 0)
    unmixed = run("bypass_gap", noise=False)["diag"]["uci_sum"]
    assert np.isclose(run("mixed_return", noise=False)["diag"]["uci_sum"], unmixed, rtol=1e-3)  # batch total survives mixing


def test_sensor_outside_the_jet_misses_the_bypass():
    r = run("jet_missed")
    assert r["diag"]["flagged"] == [] and r["truly_slow"] == ALL
    assert r["diag"]["status"] == "BATCH LOW"        # caught only because a fault-free baseline was stored
    assert diagnose(r["obs"])["status"] == "PASS"    # without that baseline: a complete miss


def test_warm_produce_hides_a_bypassing_zone():
    """Two physically different zones give the same sensor readings at the start: not identifiable."""
    res = _simulate_named("warm_twin")
    obs = observe(res, SensorConfig(**PERFECT))
    assert np.isclose(obs.v_true[3], obs.v_true[0], rtol=1e-12)
    assert np.isclose(obs.t_return_true[0, 3], obs.t_return_true[0, 0], rtol=1e-9)
    r = run("warm_twin")
    assert r["diag"]["status"] == "PASS"
    assert r["truly_slow"] == [3] and r["t78_ratio"][3] > 1.3


def test_cooler_pallet_triggers_a_false_alarm():
    r = run("cool_pallet")
    assert r["diag"]["flagged"] == [5] and r["truly_slow"] == []


def test_drifting_speed_sensor_triggers_a_false_alarm():
    r = run("sensor_gain_drift")
    assert r["diag"]["labels"][6] == "RESTRICTED-LIKE" and r["truly_slow"] == []


def test_dead_sensor_is_reported_as_a_sensor_fault():
    d = run("sensor_dead")["diag"]
    assert d["labels"][6] == "SENSOR FAULT" and not d["batch_low"]


def test_uniformly_weak_fan_is_invisible_without_a_baseline():
    r = run("weak_fan")
    assert r["diag"]["flagged"] == [] and r["truly_slow"] == ALL
    assert r["diag"]["status"] == "BATCH LOW"
    assert diagnose(r["obs"])["status"] == "PASS"


# ---- hygiene ----

def test_detector_input_contains_only_sensor_readings():
    """No leakage: the detector cannot see produce temperatures, flows or the scenario name."""
    names = {f.name for f in fields(Observation)}
    assert names == {"t", "v", "t_return", "t_supply", "v_true", "t_return_true"}
    obs = run("blockage")["obs"]
    blind = replace(obs, v_true=np.zeros(8), t_return_true=np.zeros_like(obs.t_return))
    assert diagnose(blind)["labels"] == diagnose(obs)["labels"]   # the noise-free fields are not used


def test_same_seed_same_result_and_different_seed_differs():
    a, b, c = run("blockage", seed=3), run("blockage", seed=3), run("blockage", seed=4)
    assert np.array_equal(a["obs"].v, b["obs"].v)
    assert not np.array_equal(a["obs"].v, c["obs"].v)


def test_bad_sensor_settings_and_unknown_scenario_are_rejected():
    res = _simulate_named("normal")
    for bad in ({"cross_mix": 0.6}, {"sigma_t": -1.0}, {"jet_capture": [1.0] * 3}, {"sigma_v": float("nan")},
                {"failed_zones": (-1,)}, {"failed_zones": (8,)}):
        with pytest.raises(ValueError):
            observe(res, SensorConfig(**bad))
    with pytest.raises(ValueError):
        run("no_such_case")
    with pytest.raises(ValueError):
        diagnose(observe(res, SensorConfig(failed_zones=tuple(ALL))))


def test_speed_sensor_offsets_cause_false_alarms():
    rows = {r["v_bias_sigma_m_s"]: r for r in noise_study(runs=40, biases=(0.0, 0.24), sigma_ts=(0.1,))}
    assert rows[0.0]["normal_false_alarm_rate"] == 0.0
    assert rows[0.24]["normal_false_alarm_rate"] > 0.5      # published accuracy figure, used as an offset spread


# ---- optional pulp-temperature input (a design change from Phase 1) ----

def test_pulp_probe_exposes_the_hidden_bypass_zone():
    assert run("warm_twin")["diag"]["flagged"] == []
    assert run("warm_twin", use_pulp=True)["diag"]["flagged"] == [3]


def test_pulp_probe_removes_the_cool_pallet_false_alarm():
    assert run("cool_pallet", use_pulp=True)["diag"]["flagged"] == []


def test_pulp_probe_does_not_help_with_a_missed_jet_or_a_weak_fan():
    for name in ("jet_missed", "weak_fan"):
        r = run(name, use_pulp=True)
        assert r["diag"]["flagged"] == [] and r["diag"]["status"] == "BATCH LOW"


def test_bad_pulp_readings_are_rejected():
    obs = run("normal")["obs"]
    for bad in ([27.0] * 7, [27.0] * 7 + [1.0]):          # wrong count; colder than the supply air
        with pytest.raises(ValueError):
            diagnose(obs, t_pulp=bad)


def test_evaporation_alone_raises_no_alarm():
    assert run("evaporation")["diag"]["status"] == "PASS"


def test_pulp_probe_needs_to_be_accurate_to_about_two_kelvin():
    rows = {r["pulp_sigma_K"]: r for r in pulp_probe_study(runs=40, sigmas=(0.5, 3.0))}
    assert rows[0.5]["normal_false_alarm_rate"] == 0.0 and rows[0.5]["warm_twin_zone_flagged_rate"] == 1.0
    assert rows[3.0]["normal_false_alarm_rate"] > 0.05


# ---- findings of the independent technical review, 11 Nov 2026 ----

def test_short_run_still_recognises_a_slow_zone():
    """Heavy load, little air, 4 h: no zone reaches 7/8 cooling, but the blocked zone must still count as slow."""
    mult = [1.0] * 8
    mult[4] = 9.0
    r = evaluate(Config(mass_per_zone=300.0, design_flow_per_zone=0.02, duration=4 * 3600.0, resistance_mult=mult), SensorConfig())
    assert not np.isfinite(r["t78_h"]).any()
    assert r["truly_slow"] == [4] and r["verdicts"][4].startswith("alarm, and")


def test_stored_baseline_fails_when_the_whole_batch_starts_cooler_unless_pulp_is_read():
    """The BATCH LOW check compares with an earlier batch. A healthy but cooler batch fools it; a pulp reading fixes that."""
    cool = Config(t_init=20.0)
    assert evaluate(cool, SensorConfig())["diag"]["status"] == "BATCH LOW"          # false alarm, Phase 1 design
    assert evaluate(cool, SensorConfig())["truly_slow"] == []
    assert evaluate(cool, SensorConfig(), use_pulp=True)["diag"]["status"] == "PASS"
