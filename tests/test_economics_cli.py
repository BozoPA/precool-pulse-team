"""Phase 1 arithmetic re-checked, and the command line."""
from precool_pulse import LABEL, economics
from precool_pulse.cli import main


def test_phase1_slide9_table_reproduces():
    rows = {r["minutes_saved"]: r for r in economics.phase1_table()}
    assert [rows[m]["rupees_per_year"] for m in (10, 15, 30, 45, 60)] == [10800, 16200, 32400, 48600, 64800]
    assert [round(rows[m]["t_co2_per_year"], 1) for m in (10, 15, 30, 45, 60)] == [0.9, 1.4, 2.7, 4.1, 5.5]
    assert rows[10]["kwh_per_year"] == 1350 and rows[30]["kwh_per_year"] == 4050


def test_phase1_baseline_cannot_cover_its_own_field_heat():
    c = economics.consistency()
    assert round(c["field_heat_kwh_thermal"]) == 154
    assert c["electricity_for_field_heat_kwh"] > c["baseline_batch_kwh"]
    assert c["baseline_is_self_consistent"] is False


def test_fan_only_saving_is_a_fraction_of_the_phase1_figure():
    assert all(0 < r["share_of_phase1_figure"] < 0.5 for r in economics.avoidable_table())


def test_phase1_parts_list_adds_up():
    assert 499 + 5 * 125 + 9 * 149 + 880 + 449 + 960 + 1500 + 950 == 7204


def test_cli(tmp_path, capsys):
    assert main(["list"]) == 0
    assert main(["run", "no_such_case", "--out", str(tmp_path)]) == 2
    assert main(["run", "blockage", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "blockage_timeseries.csv").read_text().startswith(f"# {LABEL}")
    assert "MODEL-GENERATED" in (tmp_path / "blockage_summary.json").read_text()


def test_cli_all_writes_every_output(tmp_path):
    assert main(["all", "--out", str(tmp_path), "--runs", "5"]) == 0
    for name in ("scenario_summary.csv", "noise_study.csv", "pulp_probe_study.csv", "benchmark_lab_fac.csv",
                 "benchmark_sensitivity.csv", "economics_consistency.json", "economics_montecarlo.csv",
                 "economics_sensitivity.csv", "economics_assumptions.csv", "economics_base_case.json"):
        assert (tmp_path / name).stat().st_size > 0


# ---- Phase 2 economics rework ----

def test_reworked_baseline_is_self_consistent():
    b = economics.batch_energy(economics._case("base"))
    assert b["compressor_kwh"] * 2.5 > b["field_heat_kwh_thermal"]       # the compressor removes more than the field heat
    assert b["batch_kwh"] > 60                                            # Phase 1's 60 kWh was too low for its own load


def test_no_faults_or_no_fixes_means_no_saving():
    for key in ("fault_share_of_batches", "found_and_fixed_share", "minutes_lost_per_fault"):
        s = economics.annual_saving({**economics._case("base"), key: 0.0})
        assert s["kwh_per_year"] == 0 and s["payback_years_energy_only"] == float("inf")


def test_energy_saving_is_far_below_the_phase1_headline():
    rows = {r["output"]: r for r in economics.monte_carlo(n=5000)}
    assert rows["kwh_per_year"]["p10"] < rows["kwh_per_year"]["p50"] < rows["kwh_per_year"]["p90"]
    assert rows["kwh_per_year"]["p90"] < 1350          # even the optimistic end is below Phase 1's lowest figure
    assert rows["payback_years_energy_only"]["p50"] > 5
    assert economics.monte_carlo(n=5000) == economics.monte_carlo(n=5000)   # same seed, same answer


def test_sensitivity_is_sorted_and_produce_value_breakeven():
    swings = [r["swing_rupees"] for r in economics.sensitivity()]
    assert swings == sorted(swings, reverse=True)
    v = economics.produce_value_breakeven()
    assert v["tonnes_per_year"] == 3240 and v["value_of_0.1_percent_rs"] == 129600
