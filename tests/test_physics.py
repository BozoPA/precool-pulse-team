"""Does the model obey its own conservation laws and a known exact solution?"""
from dataclasses import replace

import numpy as np
import pytest

from precool_pulse.physics import CP_AIR, Config, cell_effectiveness, p_sat, simulate, solve_airflow
from precool_pulse.scenarios import SCENARIOS, _simulate_named


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_mass_balance(name):
    air = _simulate_named(name).air
    assert abs(air.m_zone.sum() - air.m_fan) < 1e-9             # all paths together = fan delivery
    assert np.allclose(air.m_contact + air.m_pkg_bypass, air.m_through)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_energy_balance(name):
    res = _simulate_named(name)
    # heat lost by produce = sensible heat gained by air + heat carried off as evaporated water
    assert np.allclose(res.energy_air() + res.energy_latent(), res.energy_product(), rtol=1e-9)


def test_evaporation_removes_heat_that_air_temperature_sensors_cannot_see():
    wet, dry = _simulate_named("evaporation"), _simulate_named("normal")
    share = wet.energy_latent().sum() / wet.energy_product().sum()
    assert 0.03 < share < 0.40
    assert np.all(wet.energy_air() < wet.energy_product())
    assert np.all(wet.seven_eighths_time() < dry.seven_eighths_time())   # evaporation speeds cooling up
    assert np.all(dry.energy_latent() == 0)


def test_matches_exact_exponential_solution():
    """One zone, one bed cell: the exact answer is T(t) = Ts + (T0 - Ts) * exp(-t / tau)."""
    def max_error(dt):
        cfg = Config(n_zones=1, n_cells=1, duration=7200.0, dt=dt)
        res = simulate(cfg)
        m = res.air.m_contact[0]
        tau = cfg.mass_per_zone * cfg.cp_prod / (m * CP_AIR * cell_effectiveness(cfg, res.air.m_contact)[0])
        exact = cfg.t_supply + (cfg.t_init - cfg.t_supply) * np.exp(-res.t / tau)
        return np.max(np.abs(res.t_prod_mean[:, 0] - exact))
    coarse, fine = max_error(5.0), max_error(2.5)
    assert coarse < 0.02                      # K
    assert 1.7 < coarse / fine < 2.3          # halving the time step halves the error


def test_produce_cools_steadily_and_never_below_supply():
    res = _simulate_named("blockage")
    assert np.all(np.diff(res.t_prod_mean, axis=0) <= 1e-12)
    assert np.all(res.t_prod_final >= res.cfg.t_supply - 1e-9)


def test_no_temperature_gap_means_no_heat_flow():
    res = simulate(Config(t_init=2.0, t_supply=2.0, duration=600.0))
    assert np.all(res.q == 0)


def test_sealed_zone_gets_no_air_and_does_not_cool():
    mult = [1.0] * 8
    mult[2] = 1e14
    res = simulate(Config(resistance_mult=mult, duration=3600.0))
    assert res.air.m_through[2] < 1e-6
    assert abs(res.t_prod_final[2].mean() - 27.0) < 0.01


def test_open_gap_lowers_pressure_and_starves_every_zone_equally():
    normal, gap = _simulate_named("normal").air, _simulate_named("bypass_gap").air
    assert gap.dp < normal.dp
    assert np.allclose(gap.m_through, gap.m_through[0])
    assert gap.m_through[0] < normal.m_through[0]
    assert gap.m_fan > normal.m_fan           # the fan moves MORE air in total while the produce gets less


@pytest.mark.parametrize("bad", [
    {"mass_per_zone": -1}, {"dt": 0}, {"t_init": [27.0] * 7}, {"pkg_bypass": 1.5}, {"gap_area": -0.1},
    {"resistance_mult": 0}, {"t_supply": float("nan")}, {"fan_qmax_factor": 1.0}, {"n_zones": 0},
    {"t_init": float("inf")}, {"transpiration": -1e-9}, {"rh_supply": 1.5},
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        solve_airflow(replace(Config(), **bad))


def test_time_step_that_is_too_large_is_refused():
    """An explicit scheme goes unstable with a huge step; the model must refuse instead of returning nonsense."""
    for bad in ({"dt": 3000.0}, {"dt": 600.0, "mass_per_zone": 20.0, "design_flow_per_zone": 0.3}, {"dt": 50.0, "duration": 20.0}):
        with pytest.raises(ValueError):
            simulate(replace(Config(), **bad))
    simulate(Config(dt=60.0, duration=3600.0))            # a moderately coarse step is still accepted


def test_strong_evaporation_stays_physical_and_energy_still_balances():
    """Evaporation into 90% humid air can cool produce slightly BELOW the supply air (like a wet bulb), but never
    below the temperature at which the produce's vapour pressure equals that of the supply air."""
    cfg = Config(transpiration=5e-8, duration=6 * 3600.0)             # 100x the illustrative coefficient
    res = simulate(cfg)
    assert np.allclose(res.energy_air() + res.energy_latent(), res.energy_product(), rtol=1e-9)
    grid = np.linspace(-5.0, cfg.t_supply, 2001)
    floor = grid[np.argmin(np.abs(p_sat(grid) - cfg.rh_supply * p_sat(cfg.t_supply)))]
    assert floor - 0.01 <= res.t_prod_final.min() < cfg.t_supply
