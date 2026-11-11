"""Re-computation of the Phase 1 slide 9 energy claims. Inputs are the slide's own SCENARIO ASSUMPTIONS."""
P1 = dict(power_kw=15.0, batch_h=4.0, batches_per_day=3, days=180, tariff=8.0, grid_kg_per_kwh=0.675,
          load_kg=6000.0, cp_kj=3.7, delta_t=25.0, cop=2.5)


def phase1_table(minutes=(10, 15, 30, 45, 60)):
    """Slide 9 formula: the whole 15 kW stops for every minute saved (the slide calls this an upper bound)."""
    rows = []
    for m in minutes:
        kwh = P1["power_kw"] * m / 60 * P1["batches_per_day"] * P1["days"]
        rows.append({"minutes_saved": m, "kwh_per_year": round(kwh, 1), "rupees_per_year": round(kwh * P1["tariff"]),
                     "t_co2_per_year": round(kwh * P1["grid_kg_per_kwh"] / 1000, 2)})
    return rows


def consistency():
    """Does the slide's own baseline leave any electricity that ending early could save?"""
    field_heat_kwh = P1["load_kg"] * P1["cp_kj"] * P1["delta_t"] / 3600
    compressor_kwh = field_heat_kwh / P1["cop"]
    batch_kwh = P1["power_kw"] * P1["batch_h"]
    return {"field_heat_kwh_thermal": round(field_heat_kwh, 1), "electricity_for_field_heat_kwh": round(compressor_kwh, 1),
            "baseline_batch_kwh": batch_kwh, "left_for_fans_and_losses_kwh": round(batch_kwh - compressor_kwh, 1),
            "baseline_is_self_consistent": batch_kwh > compressor_kwh}


def avoidable_table(minutes=(10, 30, 60), fan_kw=(2.0, 3.5, 5.0)):
    """Narrower estimate: ending early only stops the tunnel fan and the heat that fan adds to the room.
    fan_kw values are SCENARIO ASSUMPTIONS (no fan was metered)."""
    rows = []
    batches = P1["batches_per_day"] * P1["days"]
    for f in fan_kw:
        for m in minutes:
            kwh = f * (1 + 1 / P1["cop"]) * m / 60 * batches
            rows.append({"fan_kw_assumed": f, "minutes_saved": m, "kwh_per_year": round(kwh, 1),
                         "rupees_per_year": round(kwh * P1["tariff"]),
                         "share_of_phase1_figure": round(kwh / (P1["power_kw"] * m / 60 * batches), 3)})
    return rows


# ---------------------------------------------------------------------------------------------
# Phase 2 rework. Every input below is a SCENARIO ASSUMPTION with a (low, base, high) range.
# None was metered or measured by the team. Only the grid factor is a verified published value.
# ---------------------------------------------------------------------------------------------
import numpy as np

RANGES = {
    # name: (low, base, high)
    "load_kg": (6000.0, 6000.0, 6000.0),          # NHB 6 MT norm, kept fixed
    "delta_t_K": (15.0, 25.0, 30.0),              # produce temperature drop
    "cp_kJ_per_kg_K": (3.6, 3.7, 3.9),
    "cop": (2.0, 2.5, 3.5),                       # refrigeration coefficient of performance
    "fan_kw": (2.0, 4.0, 7.5),                    # tunnel fan electrical power
    "batch_h": (3.0, 4.0, 6.0),
    "other_heat_share": (0.10, 0.20, 0.30),       # cartons, pallets, room gains, as a share of field heat
    "batches_per_year": (360.0, 540.0, 540.0),    # Phase 1: 3 a day for 180 days
    "tariff_rs_per_kwh": (6.0, 8.0, 10.0),
    "fault_share_of_batches": (0.05, 0.15, 0.30), # how often a fixable airflow fault occurs: NO DATA
    "found_and_fixed_share": (0.30, 0.55, 0.80),  # detector unvalidated, operator response unknown
    "minutes_lost_per_fault": (20.0, 55.0, 90.0), # the simulation's faults add 2% to 115% to a 4 h batch
    "kit_cost_rs": (50000.0, 50000.0, 100000.0),  # Phase 1 design target; not a quote
}
GRID_KG_PER_KWH = 0.675   # CEA CO2 Baseline Database v22.0, FY 2025-26 (published, verified 11 Nov 2026)


def batch_energy(p):
    """Electricity for one batch (kWh), built up from its parts so that it cannot fall below the field heat."""
    field_heat = p["load_kg"] * p["cp_kJ_per_kg_K"] * p["delta_t_K"] / 3600          # kWh thermal
    fan = p["fan_kw"] * p["batch_h"]                                                  # kWh electric
    compressor = (field_heat * (1 + p["other_heat_share"]) + fan) / p["cop"]          # fan heat must be removed too
    return {"field_heat_kwh_thermal": field_heat, "fan_kwh": fan, "compressor_kwh": compressor,
            "batch_kwh": fan + compressor, "kwh_per_tonne": (fan + compressor) / (p["load_kg"] / 1000),
            "average_power_kw": (fan + compressor) / p["batch_h"]}


def annual_saving(p):
    """Expected yearly electricity saving if found-and-fixed faults stop the batch earlier.
    Only fan energy and the fan's own heat are avoidable; the field heat has to be removed anyway.
    If batches run on a fixed timer, nothing is saved: the benefit is then colder produce, not energy."""
    hours = p["batches_per_year"] * p["fault_share_of_batches"] * p["found_and_fixed_share"] * p["minutes_lost_per_fault"] / 60
    kwh = hours * p["fan_kw"] * (1 + 1 / p["cop"])
    rupees = kwh * p["tariff_rs_per_kwh"]
    payback = np.where(np.asarray(rupees) > 0, p["kit_cost_rs"] / np.maximum(rupees, 1e-300), np.inf)
    payback = payback if np.ndim(payback) else float(payback)
    return {"kwh_per_year": kwh, "rupees_per_year": rupees, "t_co2_per_year": kwh * GRID_KG_PER_KWH / 1000,
            "share_of_annual_electricity": kwh / (batch_energy(p)["batch_kwh"] * p["batches_per_year"]),
            "payback_years_energy_only": payback}


def _case(which):
    return {k: v[{"low": 0, "base": 1, "high": 2}[which]] for k, v in RANGES.items()}


def base_case():
    p = _case("base")
    return {**{k: round(v, 2) for k, v in batch_energy(p).items()}, **{k: round(v, 4) for k, v in annual_saving(p).items()}}


def monte_carlo(n=20000, seed=0):
    """Draw every input from a triangular distribution over its range; report the spread of the outputs."""
    rng = np.random.default_rng(seed)
    p = {k: (np.full(n, lo) if lo == hi else rng.triangular(lo, mode, hi, n)) for k, (lo, mode, hi) in RANGES.items()}
    results = {**batch_energy(p), **annual_saving(p)}
    rows = []
    for name in ("batch_kwh", "kwh_per_tonne", "kwh_per_year", "rupees_per_year", "t_co2_per_year",
                 "share_of_annual_electricity", "payback_years_energy_only"):
        q = np.percentile(results[name], [10, 50, 90])
        rows.append({"output": name, "p10": round(float(q[0]), 3), "p50": round(float(q[1]), 3), "p90": round(float(q[2]), 3)})
    return rows


def sensitivity():
    """Move one input at a time from low to high, everything else at base: which assumption matters most?"""
    base = annual_saving(_case("base"))["rupees_per_year"]
    rows = []
    for name, (lo, _, hi) in RANGES.items():
        if lo == hi:
            continue
        low, high = (annual_saving({**_case("base"), name: v})["rupees_per_year"] for v in (lo, hi))
        rows.append({"input": name, "low_value": lo, "high_value": hi, "rupees_at_low": round(low), "rupees_at_high": round(high),
                     "swing_rupees": round(abs(high - low)), "base_rupees": round(base)})
    return sorted((r for r in rows if r["swing_rupees"] > 0), key=lambda r: -r["swing_rupees"])


def produce_value_breakeven(price_rs_per_kg=40.0):
    """How small a cut in produce loss would repay the kit in one season? price is a Phase 1 ASSUMPTION."""
    p = _case("base")
    tonnes = p["load_kg"] / 1000 * p["batches_per_year"]
    return {"tonnes_per_year": tonnes, "price_rs_per_kg_assumed": price_rs_per_kg,
            "value_of_0.1_percent_rs": round(tonnes * 1000 * 0.001 * price_rs_per_kg),
            "loss_cut_needed_to_repay_kit_percent": round(p["kit_cost_rs"] / (tonnes * 1000 * price_rs_per_kg) * 100, 4)}
