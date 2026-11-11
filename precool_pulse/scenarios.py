"""Named what-if cases. Zone numbers in descriptions count from 1 (top) to 8 (base); code indexes from 0.

"Truth" for every case comes from the simulated produce temperatures (was a zone really cooling
more than 10% slower than in the fault-free run?), never from the fault name that was injected.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .detect import Thresholds, diagnose
from .physics import Config, cell_effectiveness, simulate, solve_airflow
from .sensing import SensorConfig, observe

SLOW_TOLERANCE = 0.10   # a zone is "truly slow" if its 7/8 cooling time is >10% longer than fault-free


def _zone(index, value, default=0.0, n=8):
    out = [default] * n
    out[index] = value
    return out


def _warm_twin_t_init(bypass=0.4, zone=3):
    """Start temperature that makes a zone with carton bypass read exactly like a healthy zone at t=0."""
    base = Config()
    air = solve_airflow(base)
    bed = lambda m: 1 - (1 - cell_effectiveness(base, np.array([m]))[0]) ** base.n_cells
    m = air.m_through[zone]
    gap = (base.t_init - base.t_supply) * bed(m) / ((1 - bypass) * bed(m * (1 - bypass)))
    return _zone(zone, base.t_supply + gap, default=base.t_init)


SCENARIOS = {
    "normal": dict(desc="No fault.", cfg={}, sens={}),
    "bypass_gap": dict(desc="Large open gap (0.10 m2, e.g. 10 cm x 1 m of loose tarp) discharging at zone 1.",
                       cfg={"gap_area": _zone(0, 0.10)}, sens={}),
    "bypass_gap_small": dict(desc="Small gap (0.01 m2) at zone 1.", cfg={"gap_area": _zone(0, 0.01)}, sens={}),
    "blockage": dict(desc="Zone 5 vents restricted (9x flow resistance).",
                     cfg={"resistance_mult": _zone(4, 9.0, default=1.0)}, sens={}),
    "package_bypass": dict(desc="Zone 3: half the carton air passes around the produce.",
                           cfg={"pkg_bypass": _zone(2, 0.5)}, sens={}),
    "weak_fan": dict(desc="Fan delivers half its normal pressure; every zone equally starved.",
                     cfg={"fan_health": 0.5}, sens={}),
    "mixed_return": dict(desc="Large gap at zone 1, but return air mixes strongly between neighbouring zones.",
                         cfg={"gap_area": _zone(0, 0.10)}, sens={"cross_mix": 0.3}),
    "jet_missed": dict(desc="Large gap at zone 1, but the zone-1 sensors sit outside the bypass jet.",
                       cfg={"gap_area": _zone(0, 0.10)}, sens={"jet_capture": 0.0}),
    "warm_twin": dict(desc="Zone 4 has 40% carton bypass AND warmer produce, chosen so its readings match a healthy zone.",
                      cfg={"pkg_bypass": _zone(3, 0.4), "t_init": _warm_twin_t_init()}, sens={}),
    "cool_pallet": dict(desc="No fault; zone 6 produce simply starts cooler (20 C instead of 27 C).",
                        cfg={"t_init": _zone(5, 20.0, default=27.0)}, sens={}),
    "sensor_gain_drift": dict(desc="No fault; zone 7 speed sensor reads 40% low.",
                              cfg={}, sens={"v_gain": _zone(6, 0.6, default=1.0)}),
    "sensor_dead": dict(desc="No fault; zone 7 sensors return nothing.", cfg={}, sens={"failed_zones": (6,)}),
    "evaporation": dict(desc="No fault; produce also loses heat by evaporation (illustrative transpiration coefficient).",
                        cfg={"transpiration": 5e-10}, sens={}),
}


_CACHE = {}


def _simulate_cached(cfg):
    key = repr(cfg)
    if key not in _CACHE:
        if len(_CACHE) >= 64:
            _CACHE.clear()
        _CACHE[key] = simulate(cfg)
    return _CACHE[key]


def _simulate_named(name):
    return _simulate_cached(replace(Config(), **SCENARIOS[name]["cfg"]))


def _baseline_uci_sum(use_pulp=False):
    """Batch UCI of the default fault-free run with perfect sensors."""
    ref = _simulate_named("normal")
    return diagnose(observe(ref, SensorConfig(sigma_t=0, sigma_v=0)), t_pulp=ref.t_init if use_pulp else None)["uci_sum"]


def fault_free(cfg):
    """The same load and start temperatures with every fault removed: 'this tunnel's normal'."""
    return replace(cfg, resistance_mult=1.0, gap_area=0.0, pkg_bypass=0.0, fan_health=1.0)


def zone_verdicts(diag, truly_slow):
    """Compare what the display said with what the simulated produce did, zone by zone."""
    out = []
    for z in range(len(diag["labels"])):
        flagged, slow = z in diag["flagged"], z in truly_slow
        if flagged and slow:
            out.append("alarm, and this zone really cools slowly")
        elif flagged:
            out.append("alarm, but this zone cools normally")
        elif slow and diag["batch_low"]:
            out.append("no zone alarm; slow cooling caught only by the whole-batch check")
        elif slow:
            out.append("MISSED: this zone cools slowly and nothing warned")
        else:
            out.append("no alarm, cooling normal")
    return out


def stored_baseline(cfg, sens, thresholds=Thresholds(), use_pulp=False, pulp_sigma=1.0, noise=True):
    """Batch UCI the display would have stored from an EARLIER fault-free batch in the same tunnel.

    It is not built from the batch being judged: the earlier batch has the default start temperature, its own
    sensor noise and its own pulp readings. Without a pulp probe the stored value is only valid for batches
    that start at about the same temperature; that weakness is part of the Phase 1 design and is tested."""
    earlier = _simulate_cached(replace(fault_free(cfg), t_init=Config().t_init))
    quiet = SensorConfig(seed=sens.seed + 777, sigma_t=sens.sigma_t if noise else 0.0, sigma_v=sens.sigma_v if noise else 0.0)
    t_pulp = None
    if use_pulp:
        error = pulp_sigma * np.random.default_rng(sens.seed + 20_777).standard_normal(cfg.n_zones) if noise else 0.0
        t_pulp = earlier.t_init + error
    return diagnose(observe(earlier, quiet), thresholds, t_pulp=t_pulp)["uci_sum"]


def evaluate(cfg, sens, thresholds=Thresholds(), use_pulp=False, pulp_sigma=1.0, noise=True, name="custom", desc=""):
    """Run any set of inputs through the full chain: physics -> sensor readings -> labels -> independent check.

    use_pulp: also give the detector one pulp temperature per zone, probed at loading, with a random
    error of pulp_sigma kelvin (probe error plus one fruit not representing the whole zone)."""
    res, ref = _simulate_cached(cfg), _simulate_cached(fault_free(cfg))
    if not noise:
        sens = replace(sens, sigma_t=0.0, sigma_v=0.0)
    obs = observe(res, sens)
    t_pulp = None
    if use_pulp:
        error = pulp_sigma * np.random.default_rng(sens.seed + 10_000).standard_normal(cfg.n_zones) if noise else 0.0
        t_pulp = res.t_init + error
    diag = diagnose(obs, thresholds, baseline_uci_sum=stored_baseline(cfg, sens, thresholds, use_pulp, pulp_sigma, noise), t_pulp=t_pulp)
    t78, t78_ref = res.seven_eighths_time(), ref.seven_eighths_time()
    left, left_ref = res.remaining_fraction(), ref.remaining_fraction()
    truly_slow = []
    for z in range(cfg.n_zones):
        if np.isfinite(t78_ref[z]):                      # normal case: compare 7/8 cooling times (inf = never got there = slow)
            slow = t78[z] > (1 + SLOW_TOLERANCE) * t78_ref[z]
        else:                                            # run too short for either to reach 7/8: compare what is left to remove
            slow = left[z] > (1 + SLOW_TOLERANCE) * left_ref[z]
        if slow:
            truly_slow.append(z)
    with np.errstate(invalid="ignore"):
        ratio = t78 / t78_ref                              # nan where neither run reached 7/8 cooling
    return {"name": name, "desc": desc, "res": res, "ref": ref, "obs": obs, "diag": diag, "t78_h": t78 / 3600,
            "t78_ratio": ratio, "truly_slow": truly_slow, "verdicts": zone_verdicts(diag, truly_slow)}


def run(name, seed=0, noise=True, thresholds=Thresholds(), use_pulp=False, pulp_sigma=1.0):
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}'; choose from {sorted(SCENARIOS)}")
    return evaluate(replace(Config(), **SCENARIOS[name]["cfg"]), SensorConfig(seed=seed, **SCENARIOS[name]["sens"]),
                    thresholds, use_pulp, pulp_sigma, noise, name=name, desc=SCENARIOS[name]["desc"])
