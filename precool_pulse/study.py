"""Repeat the detector many times with random sensor errors. MODEL-GENERATED rates, not field accuracy."""
from dataclasses import replace

from .detect import diagnose
from .scenarios import SCENARIOS, _baseline_uci_sum, _simulate_named, run
from .sensing import SensorConfig, observe


def noise_study(runs=200, biases=(0.0, 0.02, 0.05, 0.11, 0.24), sigma_ts=(0.1, 0.5)):
    """v_bias = spread of fixed, unknown speed-sensor offsets (m/s). 0.11 and 0.24 m/s are the accuracy
    figures reported by Alimardani & Hoff (2005), measured at 20-45 C, not in cold air."""
    cases = {"normal": None, "blockage": 4, "package_bypass": 2}
    rows = []
    for sigma_t in sigma_ts:
        for bias in biases:
            row = {"v_bias_sigma_m_s": bias, "sigma_t_K": sigma_t, "runs": runs}
            for name, fault_zone in cases.items():
                res = _simulate_named(name)
                base = SensorConfig(sigma_t=sigma_t, v_bias_sigma=bias, **SCENARIOS[name]["sens"])
                any_flag = hit = exact = 0
                for seed in range(runs):
                    flagged = diagnose(observe(res, replace(base, seed=seed)), baseline_uci_sum=_baseline_uci_sum())["flagged"]
                    any_flag += bool(flagged)
                    hit += fault_zone in flagged
                    exact += flagged == [fault_zone]
                if fault_zone is None:
                    row["normal_false_alarm_rate"] = any_flag / runs
                else:
                    row[f"{name}_zone_flagged_rate"] = hit / runs
                    row[f"{name}_only_that_zone_rate"] = exact / runs
            rows.append(row)
    return rows


def pulp_probe_study(runs=200, sigmas=(0.5, 1.0, 2.0, 3.0)):
    """Detector WITH a pulp temperature per zone. sigma = random error of that pulp reading (K).
    The two confound cases were CONSTRUCTED to defeat the Phase 1 design (its rates are listed for reference),
    so the comparison shows that the pulp reading recovers them, not how often they occur in practice."""
    rows = []
    for sigma in sigmas:
        results = {name: [run(name, seed=seed, use_pulp=True, pulp_sigma=sigma)["diag"]["flagged"] for seed in range(runs)]
                   for name in ("normal", "cool_pallet", "warm_twin")}
        phase1 = {name: [run(name, seed=seed)["diag"]["flagged"] for seed in range(runs)] for name in ("cool_pallet", "warm_twin")}
        rows.append({"pulp_sigma_K": sigma, "runs": runs,
                     "phase1_design_cool_pallet_false_alarm_rate": sum(bool(f) for f in phase1["cool_pallet"]) / runs,
                     "phase1_design_warm_twin_zone_flagged_rate": sum(3 in f for f in phase1["warm_twin"]) / runs,
                     "normal_false_alarm_rate": sum(bool(f) for f in results["normal"]) / runs,
                     "cool_pallet_false_alarm_rate": sum(bool(f) for f in results["cool_pallet"]) / runs,
                     "warm_twin_zone_flagged_rate": sum(3 in f for f in results["warm_twin"]) / runs,
                     "warm_twin_only_that_zone_rate": sum(f == [3] for f in results["warm_twin"]) / runs})
    return rows
