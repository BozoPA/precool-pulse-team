"""What the proposed return-air sensor rail would read, given the simulated air streams.

MODEL-GENERATED. The rail is NOT physically built; these are hypothetical sensor readings.
Each zone has one air-temperature sensor and one air-speed sensor; one extra sensor reads supply air.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import RHO_AIR, Result, per_zone


@dataclass
class SensorConfig:
    jet_capture: object = 1.0   # 1 = gap air fully mixed into the zone reading, 0 = sensor misses the jet,
                                # >1 = sensor sits inside the jet
    cross_mix: float = 0.0      # share of each neighbouring zone's air reaching a sensor (0 to 0.5)
    sigma_t: float = 0.1        # K, random noise of each temperature reading
    sigma_v: float = 0.05       # m/s, random noise of each speed reading
    v_gain: object = 1.0        # fixed calibration gain of each speed sensor
    v_gain_sigma: float = 0.0   # spread of unknown calibration gain errors, drawn once per run
    v_bias_sigma: float = 0.0   # m/s, spread of unknown fixed speed offsets, drawn once per run
    lag_s: float = 10.0         # s, first-order response time of the temperature sensors
    failed_zones: tuple = ()    # zones whose sensors return nothing
    max_time: float = 600.0     # s of readings to generate (the detector only uses the opening minutes)
    seed: int = 0


@dataclass
class Observation:
    t: np.ndarray
    v: np.ndarray              # m/s (samples, zones) as measured
    t_return: np.ndarray       # C   (samples, zones) as measured
    t_supply: np.ndarray       # C   (samples,) as measured
    v_true: np.ndarray         # m/s (zones,) noise-free local value at the sensor
    t_return_true: np.ndarray  # C   (samples, zones) noise-free, before sensor lag


def _mix(x, g):
    """Exchange a share g with each neighbouring zone (ends reflect). Keeps the total unchanged."""
    left = np.concatenate([x[..., :1], x[..., :-1]], axis=-1)
    right = np.concatenate([x[..., 1:], x[..., -1:]], axis=-1)
    return (1 - 2 * g) * x + g * (left + right)


def observe(res: Result, sens: SensorConfig) -> Observation:
    cfg, air = res.cfg, res.air
    n = cfg.n_zones
    if not 0 <= sens.cross_mix < 0.5:
        raise ValueError("cross_mix must be in [0, 0.5)")
    levels = (sens.sigma_t, sens.sigma_v, sens.v_gain_sigma, sens.v_bias_sigma, sens.lag_s)
    if not all(np.isfinite(x) and x >= 0 for x in levels):
        raise ValueError("noise levels and lag must be finite and not negative")
    if any(not (0 <= int(z) < n) for z in sens.failed_zones):
        raise ValueError(f"failed_zones must hold zone indexes from 0 to {n - 1}")
    keep = res.t < sens.max_time
    t = res.t[keep]
    cold = air.m_pkg_bypass + per_zone(sens.jet_capture, n, "jet_capture") * air.m_gap  # still at supply temp
    flow = air.m_contact + cold                                    # kg/s sampled by each zone's sensors
    heat = air.m_contact * res.t_out[keep] + cold * cfg.t_supply   # flow-weighted temperature sum
    flow, heat = _mix(flow, sens.cross_mix), _mix(heat, sens.cross_mix)
    v_true = flow / (RHO_AIR * cfg.zone_area)
    t_true = np.where(flow > 0, heat / np.where(flow > 0, flow, 1.0), cfg.t_supply)

    rng = np.random.default_rng(sens.seed)
    gain = per_zone(sens.v_gain, n, "v_gain") * (1 + sens.v_gain_sigma * rng.standard_normal(n))
    bias = sens.v_bias_sigma * rng.standard_normal(n)
    alpha = 1.0 if sens.lag_s == 0 else 1 - np.exp(-cfg.dt / sens.lag_s)
    lagged = np.empty_like(t_true)
    state = np.full(n, cfg.t_supply)        # sensors start cold-soaked at supply temperature
    for i in range(len(t)):
        state = state + alpha * (t_true[i] - state)
        lagged[i] = state
    v = v_true * gain + bias + sens.sigma_v * rng.standard_normal(t_true.shape)
    t_return = lagged + sens.sigma_t * rng.standard_normal(t_true.shape)
    t_supply = cfg.t_supply + sens.sigma_t * rng.standard_normal(len(t))
    for z in sens.failed_zones:
        v[:, z] = np.nan
        t_return[:, z] = np.nan
    return Observation(t=t, v=v, t_return=t_return, t_supply=t_supply, v_true=v_true, t_return_true=t_true)
