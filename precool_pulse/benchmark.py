"""Compare the cooling model, untuned, with PUBLISHED MEASURED pulp temperatures.

Data: Sadie et al. (2025), Precooling Trial Data, Zenodo, doi:10.5281/zenodo.15726691 (CC BY 4.0);
method in MethodsX (2026) doi:10.1016/j.mex.2026.103813. Valencia oranges in a 1.2 m deep laboratory
cooling tunnel at superficial air speeds of 0.05, 0.20 and 0.35 m/s; two pulp probes per tunnel.

What is taken from the experiment: air speed, bed depth, start temperature (first reading) and the air
temperature (see t_air below). Everything that sets the cooling RATE is a prior assumption, not fitted:
fruit diameter 0.08 m (the paper's simulated-fruit size), bulk density 540 kg/m3 (estimated from the
commercial pallet in the paper; the tunnel cross-section is not reported), orange properties from
handbooks (cp 3850 J/kg/K, k 0.55 W/m/K, density 960 kg/m3), Wakao packed-bed correlation.
The model gives the average temperature of a fruit; the probes read pulp inside one fruit, which lags.
"""
from pathlib import Path

import numpy as np

from .physics import Config, simulate

DATA = Path(__file__).resolve().parent.parent / "benchmark_data" / "lab_fac_pulp_10min.csv"
SPEEDS = (0.05, 0.20, 0.35)
ASSUMED = dict(depth=1.2, bulk_density=540.0, fruit_diameter=0.08, cp_prod=3850.0, k_prod=0.55, rho_prod=960.0)


def load_measured():
    d = np.genfromtxt(DATA, delimiter=",", skip_header=2)
    return d[:, 0] / 60, {s: d[:, 1 + 2 * i: 3 + 2 * i] for i, s in enumerate(SPEEDS)}   # hours, {speed: (n, 2)}


def time_to_fraction(t, temp, t_start, t_air, fraction):
    """First time (same unit as t) the remaining temperature gap falls to `fraction` of the starting gap."""
    y = (np.asarray(temp) - t_air) / (t_start - t_air)
    below = np.nonzero(y <= fraction)[0]
    if not below.size:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (y[i - 1] - fraction) / (y[i - 1] - y[i]) * (t[i] - t[i - 1]))


def predict(speed, t_start, t_air, hours=27.0, **overrides):
    a = dict(ASSUMED, **overrides)
    cfg = Config(n_zones=1, n_cells=15, zone_area=1.0, design_flow_per_zone=speed, t_supply=t_air, t_init=t_start,
                 mass_per_zone=a["bulk_density"] * a["depth"], fruit_diameter=a["fruit_diameter"],
                 cp_prod=a["cp_prod"], k_prod=a["k_prod"], rho_prod=a["rho_prod"], dt=10.0, duration=hours * 3600)
    res = simulate(cfg)
    return res.t / 3600, res.t_cells[:, 0, :]          # hours, (steps, cells) with cell 0 at the air inlet


def compare(t_air_source="measured_final", **overrides):
    """One row per air speed: measured half- and 7/8-cooling times of both probes against the model's
    range from the inlet-side fruit to the outlet-side fruit."""
    hours, measured = load_measured()
    rows = []
    for speed in SPEEDS:
        m = measured[speed]
        t_start = float(m[0].mean())
        t_air = float(m[-12:].mean()) if t_air_source == "measured_final" else -0.6
        th, cells = predict(speed, t_start, t_air, **overrides)
        row = {"air_speed_m_s": speed, "t_air_source": t_air_source, "t_air_C": round(t_air, 2), "t_start_C": round(t_start, 2)}
        for label, frac in (("half", 0.5), ("seven_eighths", 0.125)):
            meas = sorted(time_to_fraction(hours, m[:, k], m[0, k], t_air, frac) for k in (0, 1))
            lo = time_to_fraction(th, cells[:, 0], t_start, t_air, frac)
            mid = time_to_fraction(th, cells.mean(axis=1), t_start, t_air, frac)
            hi = time_to_fraction(th, cells[:, -1], t_start, t_air, frac)
            row.update({f"{label}_measured_fast_h": round(meas[0], 2), f"{label}_measured_slow_h": round(meas[1], 2),
                        f"{label}_model_inlet_h": round(lo, 2), f"{label}_model_bed_mean_h": round(mid, 2),
                        f"{label}_model_outlet_h": round(hi, 2),
                        f"{label}_both_probes_inside_model_range": bool(lo <= meas[0] and meas[1] <= hi)})
        rows.append(row)
    return rows


def sensitivity():
    """How much the predicted bed-average half-cooling time moves when the unreported inputs move."""
    hours, measured = load_measured()
    rows = []
    for name, values in (("bulk_density", (460.0, 540.0, 620.0)), ("fruit_diameter", (0.07, 0.08, 0.09)), ("k_prod", (0.45, 0.55, 0.65))):
        for value in values:
            row = {"changed_input": name, "value": value}
            for speed in SPEEDS:
                m = measured[speed]
                t_air, t_start = float(m[-12:].mean()), float(m[0].mean())
                th, cells = predict(speed, t_start, t_air, **{name: value})
                model = time_to_fraction(th, cells.mean(axis=1), t_start, t_air, 0.5)
                meas = np.mean([time_to_fraction(hours, m[:, k], m[0, k], t_air, 0.5) for k in (0, 1)])
                row[f"half_time_error_at_{speed:.2f}_m_s"] = round(float(model / meas - 1), 3)
            rows.append(row)
    return rows
