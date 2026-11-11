"""The dashboard runs end to end without a browser, and its messages come from the inputs."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from precool_pulse import LABEL

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(preset=None, **widgets):
    at = AppTest.from_file(APP, default_timeout=120).run()
    if preset:
        at.selectbox[0].set_value(preset)
        at.button[0].click().run()
    for key, value in widgets.items():
        getattr(at, value[0])(key=key).set_value(value[1])
    if widgets:
        at.run()
    assert not at.exception, at.exception
    return at


def test_default_page_passes_and_carries_the_simulation_label():
    at = _run()
    assert LABEL in at.warning[0].value
    assert "PASS" in at.success[0].value and not at.error


def test_blockage_preset_names_zone_5():
    at = _run("blockage")
    assert "CHECK ZONE" in at.error[0].value and "Zone 5: RESTRICTED-LIKE" in at.error[0].value


def test_warning_follows_the_inputs_not_the_preset_name():
    """Load 'normal', then weaken the fan by hand: the page must change to BATCH LOW."""
    at = _run("normal", fan_health=("slider", 0.5))
    assert "BATCH LOW" in at.warning[1].value


def test_pulp_probe_switch_changes_the_cool_pallet_result():
    assert "PASS" in _run("cool_pallet").success[0].value                      # probe on by default
    at = _run("cool_pallet", use_pulp=("checkbox", False))
    assert "Zone 6: LOW HEAT PICKUP" in at.error[0].value
    assert "misleading" in at.info[0].value


def test_impossible_inputs_show_an_error_instead_of_crashing():
    at = _run("normal", t_supply=("slider", 10.0), use_pulp=("checkbox", False))
    at.session_state["zones"]["Start temp (°C)"] = 5.0
    at.run()
    assert not at.exception and "cannot be simulated" in at.error[0].value


def test_gap_case_says_the_zone_labels_understate_the_problem():
    at = _run("bypass_gap")
    assert "Zone 1: BYPASS-LIKE" in at.error[0].value and "understate" in at.info[0].value
