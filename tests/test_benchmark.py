"""Model, untuned, against published measured pulp temperatures (Sadie et al. 2025, Zenodo 15726691)."""
import numpy as np

from precool_pulse.benchmark import SPEEDS, compare, load_measured, time_to_fraction


def test_measured_data_loads_unchanged():
    hours, measured = load_measured()
    assert len(hours) == 158 and hours[1] - hours[0] == 1 / 6          # 10-minute readings for 26 hours
    assert np.allclose(measured[0.05][0], [7.57, 7.537], atol=1e-3)    # first row of the source sheet


def test_time_to_fraction_on_an_exact_exponential():
    t = np.linspace(0, 10, 2001)
    assert abs(time_to_fraction(t, 20 * np.exp(-t / 2), 20.0, 0.0, 0.5) - 2 * np.log(2)) < 1e-4
    assert np.isnan(time_to_fraction(t, np.full_like(t, 20.0), 20.0, 0.0, 0.5))


def test_measured_probes_fall_inside_the_models_inlet_to_outlet_range():
    for row in compare("measured_final"):
        assert row["half_both_probes_inside_model_range"] and row["seven_eighths_both_probes_inside_model_range"]


def test_bed_average_half_cooling_time_within_45_percent_of_the_probe_average():
    for row in compare("measured_final"):
        measured = (row["half_measured_fast_h"] + row["half_measured_slow_h"]) / 2
        assert abs(row["half_model_bed_mean_h"] / measured - 1) < 0.45


def test_model_and_measurement_both_cool_faster_at_higher_air_speed():
    rows = compare("measured_final")
    assert [r["air_speed_m_s"] for r in rows] == list(SPEEDS)
    for key in ("half_model_bed_mean_h", "half_measured_slow_h"):
        assert rows[0][key] > rows[1][key] > rows[2][key]


def test_known_miss_when_the_nominal_air_setpoint_is_used():
    """The fruit never reached the -0.6 C set point, so 7/8 times against that target are under-predicted."""
    assert not all(r["seven_eighths_both_probes_inside_model_range"] for r in compare("setpoint"))
