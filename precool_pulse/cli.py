"""Command line: python -m precool_pulse {list | run NAME | all}. Writes CSV/JSON; all values MODEL-GENERATED."""
import argparse
import csv
import json
import sys
from pathlib import Path

from . import LABEL, benchmark, economics
from .scenarios import SCENARIOS, run
from .study import noise_study, pulp_probe_study


def _write_csv(path, rows, label=LABEL):
    with open(path, "w", newline="") as f:
        f.write(f"# {label}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _summary(r):
    d = r["diag"]
    zones = lambda idx: " ".join(str(z + 1) for z in idx) or "-"
    with_pulp = run(r["name"], use_pulp=True)["diag"]
    res = r["res"]
    return {"scenario": r["name"], "description": r["desc"], "status": d["status"], "batch_low": d["batch_low"],
            "zone_labels": " | ".join(f"Z{z + 1}:{lab}" for z, lab in enumerate(d["labels"]) if lab != "PASS") or "all PASS",
            "flagged_by_signature": zones(d["flagged"]), "flagged_by_uci_only": zones(d["uci_only"]),
            "flagged_by_temp_only": zones(d["temp_only"]), "flagged_by_speed_only": zones(d["vel_only"]),
            "status_with_pulp_probe": with_pulp["status"], "flagged_with_pulp_probe": zones(with_pulp["flagged"]),
            "truly_slow_zones": zones(r["truly_slow"]),
            "latent_share_of_heat_removed": round(float(res.energy_latent().sum() / res.energy_product().sum()), 4),
            "slowest_zone_7_8_time_h": round(float(r["t78_h"].max()), 3),
            "slowest_vs_fault_free": round(float(r["t78_ratio"].max()), 3),
            "pressure_across_load_Pa": round(r["res"].air.dp, 1),
            "energy_balance_rel_error": float(abs(res.energy_air().sum() + res.energy_latent().sum()
                                                  - res.energy_product().sum()) / res.energy_product().sum()),
            "data_kind": "MODEL-GENERATED"}


def _timeseries(r):
    obs, res = r["obs"], r["res"]
    rows = []
    for i, t in enumerate(obs.t):
        for z in range(res.cfg.n_zones):
            rows.append({"t_s": t, "zone": z + 1, "speed_m_s": round(float(obs.v[i, z]), 4),
                         "t_return_C": round(float(obs.t_return[i, z]), 3), "t_supply_C": round(float(obs.t_supply[i]), 3),
                         "t_produce_mean_C": round(float(res.t_prod_mean[i, z]), 3), "heat_removal_W": round(float(res.q[i, z]), 1)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="precool_pulse", description=LABEL)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    one = sub.add_parser("run")
    one.add_argument("scenario")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--out", default="outputs")
    every = sub.add_parser("all")
    every.add_argument("--out", default="outputs")
    every.add_argument("--runs", type=int, default=200)
    args = p.parse_args(argv)
    try:
        if args.cmd == "list":
            for name, s in SCENARIOS.items():
                print(f"{name:18s} {s['desc']}")
            return 0
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.cmd == "run":
            r = run(args.scenario, seed=args.seed)
            _write_csv(out / f"{args.scenario}_timeseries.csv", _timeseries(r))
            summary = dict(_summary(r), seed=args.seed, label=LABEL)
            (out / f"{args.scenario}_summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
            return 0
        rows = [_summary(run(name)) for name in SCENARIOS]
        _write_csv(out / "scenario_summary.csv", rows)
        _write_csv(out / "noise_study.csv", noise_study(runs=args.runs))
        _write_csv(out / "pulp_probe_study.csv", pulp_probe_study(runs=args.runs))
        _write_csv(out / "benchmark_lab_fac.csv", benchmark.compare("measured_final") + benchmark.compare("setpoint"),
                   label="Model columns are MODEL-GENERATED; measured columns are PUBLISHED MEASURED "
                         "(Sadie et al. 2025, doi:10.5281/zenodo.15726691, CC BY 4.0)")
        _write_csv(out / "benchmark_sensitivity.csv", benchmark.sensitivity(), label="MODEL-GENERATED sensitivity of the benchmark prediction")
        _write_csv(out / "economics_phase1_recomputed.csv", economics.phase1_table())
        _write_csv(out / "economics_fan_only.csv", economics.avoidable_table())
        (out / "economics_consistency.json").write_text(json.dumps(economics.consistency(), indent=2))
        assumed = "MODEL-GENERATED from SCENARIO ASSUMPTIONS (economics.RANGES). Nothing here was metered."
        _write_csv(out / "economics_montecarlo.csv", economics.monte_carlo(), label=assumed)
        _write_csv(out / "economics_sensitivity.csv", economics.sensitivity(), label=assumed)
        _write_csv(out / "economics_assumptions.csv", [{"input": k, "low": v[0], "base": v[1], "high": v[2]}
                                                       for k, v in economics.RANGES.items()], label="SCENARIO ASSUMPTIONS")
        (out / "economics_base_case.json").write_text(json.dumps(
            {"label": assumed, "base_case": economics.base_case(), "produce_value": economics.produce_value_breakeven()}, indent=2))
        for row in rows:
            print(f"{row['scenario']:18s} {row['status']:10s} signature:{row['flagged_by_signature']:6s} "
                  f"with pulp probe:{row['flagged_with_pulp_probe']:6s} truly slow:{row['truly_slow_zones']}")
        return 0
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
