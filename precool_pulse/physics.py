"""Airflow network + packed-bed cooling model for one pallet face split into zones.

Everything here is MODEL-GENERATED. Assumptions (each one is a simplification, not a measurement):
  * Air paths are parallel resistances between a common supply side and a common return plenum,
    each with a quadratic pressure law (dp = R * m^2). The fan follows dp = Pmax * (1 - (Q/Qmax)^2).
  * Each zone's produce is a packed bed of spheres cut into cells along the flow direction.
    Air leaves each cell following the standard heat-exchanger effectiveness relation.
  * Evaporation from the produce is optional and crude (one transpiration coefficient, off by default).
    No respiration heat, no carton thermal mass, no wall gains.
  * Air properties are constant (dry air near 2-5 C).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RHO_AIR = 1.27    # kg/m3
CP_AIR = 1006.0   # J/kg/K
MU_AIR = 1.72e-5  # Pa.s
K_AIR = 0.0243    # W/m/K
PR_AIR = 0.71
H_FG = 2.5e6       # J/kg, latent heat of evaporation of water
P_ATM = 101325.0  # Pa


def p_sat(t_c):
    """Saturation vapour pressure of water (Pa), Magnus formula."""
    return 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))


def per_zone(value, n, name):
    """Turn a number or a list into an array with one value per zone."""
    arr = np.full(n, float(value)) if np.isscalar(value) else np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be one number or a list of {n} numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    return arr


@dataclass
class Config:
    n_zones: int = 8
    n_cells: int = 5                 # bed cells along the flow direction
    t_supply: float = 2.0            # C, cold supply air
    t_init: object = 27.0            # C, produce start temperature (number or per-zone list)
    mass_per_zone: float = 100.0     # kg produce per zone
    cp_prod: float = 3700.0          # J/kg/K (Phase 1 slide 9 uses 3.7 kJ/kg/K)
    fruit_diameter: float = 0.07     # m
    rho_prod: float = 1000.0         # kg/m3
    k_prod: float = 0.5              # W/m/K
    zone_area: float = 0.25          # m2 of return face per zone
    design_flow_per_zone: float = 0.1  # m3/s (1 L/s per kg of produce: assumed design point)
    design_dp: float = 150.0         # Pa across the load at the design point (assumed)
    fan_qmax_factor: float = 2.0     # fan free-delivery flow / design flow
    fan_health: float = 1.0          # multiplies the fan's maximum pressure
    resistance_mult: object = 1.0    # >1 = restricted vents in that zone
    gap_area: object = 0.0           # m2 of open gap (tarp/pallet) discharging into that zone
    gap_cd: float = 0.6              # orifice discharge coefficient
    pkg_bypass: object = 0.0         # fraction of carton air that passes around the produce
    transpiration: float = 0.0       # kg water per kg produce per second per Pa of vapour-pressure gap (0 = off)
    rh_supply: float = 0.9           # relative humidity of the supply air (only used with transpiration)
    dt: float = 5.0                  # s
    duration: float = 12 * 3600.0    # s

    def validate(self):
        n = self.n_zones
        if n < 1 or self.n_cells < 1:
            raise ValueError("n_zones and n_cells must be at least 1")
        for name in ("mass_per_zone", "cp_prod", "fruit_diameter", "rho_prod", "k_prod", "zone_area",
                     "design_flow_per_zone", "design_dp", "fan_health", "dt", "duration"):
            if not (np.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be a positive number")
        if self.fan_qmax_factor <= 1:
            raise ValueError("fan_qmax_factor must be greater than 1")
        if np.any(per_zone(self.resistance_mult, n, "resistance_mult") <= 0):
            raise ValueError("resistance_mult must be positive")
        if np.any(per_zone(self.gap_area, n, "gap_area") < 0):
            raise ValueError("gap_area cannot be negative")
        b = per_zone(self.pkg_bypass, n, "pkg_bypass")
        if np.any(b < 0) or np.any(b > 1):
            raise ValueError("pkg_bypass must be between 0 and 1")
        per_zone(self.t_init, n, "t_init")
        if not (np.isfinite(self.transpiration) and self.transpiration >= 0 and 0 <= self.rh_supply <= 1):
            raise ValueError("transpiration must be >= 0 and rh_supply between 0 and 1")
        if not np.isfinite(self.t_supply):
            raise ValueError("t_supply must be finite")


@dataclass
class Airflow:
    dp: float                 # Pa across the load
    m_through: np.ndarray     # kg/s entering each zone's cartons
    m_contact: np.ndarray     # kg/s that actually passes over produce
    m_pkg_bypass: np.ndarray  # kg/s through the cartons but around the produce
    m_gap: np.ndarray         # kg/s through open gaps
    m_fan: float              # kg/s delivered by the fan at this dp (from the fan curve)

    @property
    def m_zone(self):
        return self.m_through + self.m_gap


def solve_airflow(cfg: Config) -> Airflow:
    """Mass-conserving operating point: sum of all path flows = fan flow at the same dp."""
    cfg.validate()
    n = cfg.n_zones
    m_design = RHO_AIR * cfg.design_flow_per_zone
    r_zone = cfg.design_dp / m_design**2 * per_zone(cfg.resistance_mult, n, "resistance_mult")
    gap_coeff = cfg.gap_cd * per_zone(cfg.gap_area, n, "gap_area") * np.sqrt(2 * RHO_AIR)
    conductance = np.sum(1 / np.sqrt(r_zone)) + np.sum(gap_coeff)   # total m = conductance * sqrt(dp)
    p_max = cfg.fan_health * cfg.design_dp / (1 - 1 / cfg.fan_qmax_factor**2)
    m_max = cfg.fan_qmax_factor * n * m_design
    dp = p_max / (1 + p_max * conductance**2 / m_max**2)
    m_through = np.sqrt(dp / r_zone)
    bypass = per_zone(cfg.pkg_bypass, n, "pkg_bypass")
    return Airflow(dp=dp, m_through=m_through, m_contact=m_through * (1 - bypass),
                   m_pkg_bypass=m_through * bypass, m_gap=gap_coeff * np.sqrt(dp),
                   m_fan=m_max * np.sqrt(max(0.0, 1 - dp / p_max)))


def cell_effectiveness(cfg: Config, m_contact: np.ndarray) -> np.ndarray:
    """Fraction of the (produce - air) temperature gap the air closes while crossing one bed cell.

    Air-side film coefficient: Wakao packed-bed correlation Nu = 2 + 1.1 Pr^(1/3) Re^0.6.
    Internal conduction of the fruit: sphere linear-driving-force term r/(5k).
    """
    d = cfg.fruit_diameter
    velocity = m_contact / (RHO_AIR * cfg.zone_area)
    reynolds = RHO_AIR * velocity * d / MU_AIR
    h = (2.0 + 1.1 * PR_AIR ** (1 / 3) * reynolds**0.6) * K_AIR / d
    u_overall = 1 / (1 / h + (d / 2) / (5 * cfg.k_prod))
    ua_cell = u_overall * 6 / (cfg.rho_prod * d) * cfg.mass_per_zone / cfg.n_cells
    with np.errstate(divide="ignore"):
        ntu = np.where(m_contact > 0, ua_cell / (m_contact * CP_AIR), np.inf)
    return np.where(m_contact > 0, 1 - np.exp(-ntu), 0.0)


@dataclass
class Result:
    cfg: Config
    air: Airflow
    t: np.ndarray               # s, start of each step
    t_out: np.ndarray           # C, air leaving each zone's produce bed  (steps, zones)
    t_prod_mean: np.ndarray     # C, mass-average produce temperature     (steps, zones)
    q: np.ndarray               # W, sensible heat picked up by the air   (steps, zones)
    t_prod_final: np.ndarray    # C, per cell after the last step         (zones, cells)
    t_cells: np.ndarray = field(default=None)   # C, produce per bed cell, cell 0 = air inlet side (steps, zones, cells)
    q_latent: np.ndarray = field(default=None)  # W, heat leaving the produce as evaporated water (steps, zones)
    t_init: np.ndarray = field(default=None)

    def energy_air(self):
        """J of sensible heat carried away by the air (what air temperature sensors can see), per zone."""
        return self.q.sum(axis=0) * self.cfg.dt

    def energy_latent(self):
        """J that left the produce as evaporated water (invisible to air temperature sensors), per zone."""
        return self.q_latent.sum(axis=0) * self.cfg.dt

    def energy_product(self):
        """J lost by the produce, per zone."""
        cap_cell = self.cfg.mass_per_zone / self.cfg.n_cells * self.cfg.cp_prod
        return cap_cell * (self.t_init[:, None] - self.t_prod_final).sum(axis=1)

    def remaining_fraction(self):
        """Share of the starting (produce - supply) temperature gap still left at the end of the run, per zone."""
        gap = self.t_init - self.cfg.t_supply
        return np.where(gap != 0, (self.t_prod_final.mean(axis=1) - self.cfg.t_supply) / np.where(gap != 0, gap, 1.0), 0.0)

    def seven_eighths_time(self):
        """s until each zone's mass-average produce temperature has covered 7/8 of the way to supply.
        inf if not reached inside the simulated duration."""
        target = self.cfg.t_supply + (self.t_init - self.cfg.t_supply) / 8
        out = np.full(self.cfg.n_zones, np.inf)
        for z in range(self.cfg.n_zones):
            if self.t_init[z] > self.cfg.t_supply:
                hit = np.nonzero(self.t_prod_mean[:, z] <= target[z])[0]
            else:
                hit = np.nonzero(self.t_prod_mean[:, z] >= target[z])[0]
            if hit.size:
                out[z] = self.t[hit[0]]
        return out


def simulate(cfg: Config) -> Result:
    air = solve_airflow(cfg)
    n, cells = cfg.n_zones, cfg.n_cells
    steps = int(round(cfg.duration / cfg.dt))
    t_init = per_zone(cfg.t_init, n, "t_init")
    temp = np.repeat(t_init[:, None], cells, axis=1)
    cap_cell = cfg.mass_per_zone / cells * cfg.cp_prod
    air_cap = air.m_contact * CP_AIR            # W/K
    eff = cell_effectiveness(cfg, air.m_contact)
    if steps < 1:
        raise ValueError("duration must be at least one time step (dt)")
    # explicit time stepping: each step may only move a cell a small part of the way towards the air temperature
    step_fraction = float(np.max(air_cap * eff)) * cfg.dt / cap_cell
    if step_fraction > 0.2:
        raise ValueError(f"dt is too large for this load and airflow (step fraction {step_fraction:.2f} > 0.2); use a smaller dt")
    t_out = np.empty((steps, n))
    t_cells = np.empty((steps, n, cells))
    q = np.empty((steps, n))
    q_latent = np.zeros((steps, n))
    evaporates = cfg.transpiration > 0
    p_supply = cfg.rh_supply * p_sat(cfg.t_supply)
    for s in range(steps):
        t_cells[s] = temp
        t_air = np.full(n, cfg.t_supply)
        q_zone = np.zeros(n)
        p_vapour = np.full(n, p_supply)
        for k in range(cells):
            rise = eff * (temp[:, k] - t_air)   # air warms by this much crossing cell k
            q_cell = air_cap * rise             # W of sensible heat: produce -> air
            if evaporates:
                water = np.where(air.m_contact > 0, cfg.transpiration * cfg.mass_per_zone / cells
                                 * np.maximum(0.0, p_sat(temp[:, k]) - p_vapour), 0.0)      # kg/s
                w_now, p_limit = 0.622 * p_vapour / (P_ATM - p_vapour), np.maximum(p_sat(temp[:, k]), p_vapour)
                water = np.minimum(water, air.m_contact * (0.622 * p_limit / (P_ATM - p_limit) - w_now))   # air cannot pass saturation
                w = w_now + water / np.maximum(air.m_contact, 1e-12)
                p_vapour = w * P_ATM / (0.622 + w)
                q_latent[s] += H_FG * water
                temp[:, k] -= H_FG * water * cfg.dt / cap_cell
            temp[:, k] -= q_cell * cfg.dt / cap_cell
            t_air = t_air + rise
            q_zone += q_cell
        t_out[s] = t_air
        q[s] = q_zone
    return Result(cfg=cfg, air=air, t=np.arange(steps) * cfg.dt, t_out=t_out, t_prod_mean=t_cells.mean(axis=2),
                  q=q, t_prod_final=temp, t_init=t_init, t_cells=t_cells, q_latent=q_latent)
