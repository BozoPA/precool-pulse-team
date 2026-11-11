"""Turn opening-minutes sensor readings into zone labels, plus simpler baselines to compare against.

The thresholds are HYPOTHESES (Phase 1 slide 7 says so too). No field result is claimed.
UCI = local air speed x (return air temperature - supply air temperature), per zone.

Optional extra input (NOT in the Phase 1 design): the produce pulp temperature of each zone, probed at
loading. With it, heat pickup is divided by (pulp temperature - supply temperature), which removes the
effect of pallets that simply start warmer or cooler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensing import Observation


@dataclass
class Thresholds:
    window: tuple = (60.0, 300.0)  # s after fan start that the detector averages over
    v_band: float = 0.20           # a zone's speed is "high"/"low" beyond +/-20% of the batch median
    dt_band: float = 0.20          # heat pickup is "low" below 80% of the batch median
    uci_band: float = 0.20         # UCI is "low" below 80% of the batch median
    batch_band: float = 0.15       # batch is "low" below 85% of the stored fault-free baseline


def diagnose(obs: Observation, thr: Thresholds = Thresholds(), baseline_uci_sum: float | None = None,
             t_pulp=None) -> dict:
    use = (obs.t >= thr.window[0]) & (obs.t <= thr.window[1])
    if not use.any():
        raise ValueError("no sensor samples inside the detection window")
    v = obs.v[use].mean(axis=0)
    d_t = (obs.t_return[use] - obs.t_supply[use, None]).mean(axis=0)
    if t_pulp is not None:
        gap = np.asarray(t_pulp, dtype=float) - obs.t_supply[use].mean()
        if gap.shape != d_t.shape or np.any(gap <= 0):
            raise ValueError("t_pulp needs one value per zone, each warmer than the supply air")
        d_t = d_t / gap            # share of the available temperature gap that the air picked up
    uci = v * d_t
    ok = np.isfinite(uci)
    if not ok.any():
        raise ValueError("every zone sensor has failed")

    def ratio(x):
        med = np.median(x[ok])
        return x / med if med > 0 else np.full_like(x, np.nan)

    r_v, r_dt, r_uci = ratio(v), ratio(d_t), ratio(uci)
    labels = []
    for z in range(len(uci)):
        if not ok[z]:
            labels.append("SENSOR FAULT")
            continue
        low_v, high_v = r_v[z] <= 1 - thr.v_band, r_v[z] >= 1 + thr.v_band
        low_dt = r_dt[z] <= 1 - thr.dt_band
        if low_v and low_dt:
            labels.append("DEAD ZONE OR SENSOR")
        elif high_v and low_dt:
            labels.append("BYPASS-LIKE")
        elif low_v:
            labels.append("RESTRICTED-LIKE")
        elif low_dt:
            labels.append("LOW HEAT PICKUP")
        else:
            labels.append("PASS")
    uci_sum = float(np.mean(uci[ok]) * len(uci))   # failed sensors are filled with the mean of the rest
    batch_low = baseline_uci_sum is not None and uci_sum < (1 - thr.batch_band) * baseline_uci_sum
    flagged = [z for z, lab in enumerate(labels) if lab != "PASS"]
    status = "CHECK ZONE" if flagged else ("BATCH LOW" if batch_low else "PASS")
    with np.errstate(invalid="ignore"):
        return {
            "status": status, "labels": labels, "flagged": flagged, "batch_low": bool(batch_low),
            "v": v, "dT": d_t, "uci": uci, "r_v": r_v, "r_dT": r_dt, "r_uci": r_uci, "uci_sum": uci_sum,
            # simpler detectors, for comparison
            "uci_only": [z for z in range(len(uci)) if ok[z] and r_uci[z] <= 1 - thr.uci_band],
            "temp_only": [z for z in range(len(uci)) if ok[z] and r_dt[z] <= 1 - thr.dt_band],
            "vel_only": [z for z in range(len(uci)) if ok[z] and abs(r_v[z] - 1) >= thr.v_band],
        }
