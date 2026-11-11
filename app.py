"""PreCool Pulse dashboard. Run from this folder:  streamlit run app.py

Everything shown is computed by the precool_pulse package from the inputs on screen.
Presets only fill in inputs; no label or warning is stored with a preset.
"""
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from precool_pulse import LABEL, benchmark
from precool_pulse.cli import _timeseries
from precool_pulse.physics import Config
from precool_pulse.scenarios import SCENARIOS, evaluate, run
from precool_pulse.sensing import SensorConfig

INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
ZONE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, ORANGE_BAND = "#2a78d6", "#eb6834", "rgba(235,104,52,0.18)"
STATUS = {  # label -> (colour, icon, what the operator would check)
    "PASS": ("#0ca30c", "✓", ""),
    "BYPASS-LIKE": ("#d03b3b", "✖", "fast air, little heat pickup: check the tarp seal and gaps between pallets"),
    "RESTRICTED-LIKE": ("#d03b3b", "✖", "little air: check vent alignment, liners and over-tight stacking"),
    "DEAD ZONE OR SENSOR": ("#ec835a", "◆", "little air and little heat: a blocked face, or a sensor problem"),
    "LOW HEAT PICKUP": ("#fab219", "▲", "normal air but little heat pickup: air may be passing around the produce"),
    "SENSOR FAULT": ("#8a8984", "?", "no reading from this zone's sensors"),
}
ZONES = [f"Zone {z}" + (" (top)" if z == 1 else " (base)" if z == 8 else "") for z in range(1, 9)]
DEFAULTS = dict(t_supply=2.0, mass=100.0, flow=100.0, hours=12, fan_health=1.0, evaporation=False, cross_mix=0.0,
                jet_capture=1.0, sigma_t=0.1, sigma_v=0.05, v_bias=0.0, use_pulp=True, pulp_sigma=1.0, seed=0)
COUNTEREXAMPLES = ["bypass_gap", "bypass_gap_small", "mixed_return", "jet_missed", "warm_twin", "cool_pallet",
                   "sensor_gain_drift", "weak_fan"]

st.set_page_config(page_title="PreCool Pulse simulation", layout="wide")


def load_preset(name):
    """Copy a named case's INPUTS into the controls. Results are always recomputed from the controls."""
    cfg, sens = SCENARIOS[name]["cfg"], SCENARIOS[name]["sens"]
    values = dict(DEFAULTS, fan_health=cfg.get("fan_health", 1.0), evaporation=cfg.get("transpiration", 0) > 0,
                  cross_mix=sens.get("cross_mix", 0.0), jet_capture=sens.get("jet_capture", 1.0))
    for key, value in values.items():
        st.session_state[key] = value
    per_zone = lambda source, key, default: list(source[key]) if isinstance(source.get(key), list) else [default] * 8
    st.session_state["zones"] = pd.DataFrame({
        "Zone": ZONES,
        "Start temp (°C)": per_zone(cfg, "t_init", 27.0),
        "Vent resistance (×)": per_zone(cfg, "resistance_mult", 1.0),
        "Open gap (m²)": per_zone(cfg, "gap_area", 0.0),
        "In-carton bypass (0–1)": per_zone(cfg, "pkg_bypass", 0.0),
        "Speed sensor gain": per_zone(sens, "v_gain", 1.0),
        "Sensors working": [z not in sens.get("failed_zones", ()) for z in range(8)],
    })
    st.session_state["zones_version"] = st.session_state.get("zones_version", 0) + 1


def style(fig, title, height=360, unified=True, left=60, top=100):
    fig.update_layout(title=dict(text=f"{title}<br><sup>{LABEL}</sup>", font=dict(size=15, color=INK), y=0.97, yanchor="top"),
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK, size=13), height=height,
                      margin=dict(l=left, r=20, t=top, b=50), hovermode="x unified" if unified else "closest",
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def counterexample_rows():
    rows = []
    for name in COUNTEREXAMPLES:
        plain, pulp = run(name), run(name, use_pulp=True)
        says = lambda r: r["diag"]["status"] + (": zone " + ", ".join(str(z + 1) for z in r["diag"]["flagged"]) if r["diag"]["flagged"] else "")
        slow = plain["truly_slow"]
        rows.append({"Case": SCENARIOS[name]["desc"], "Display, Phase 1 design": says(plain),
                     "Display, with pulp probe": says(pulp),
                     "What really happened": ("zones " + ", ".join(str(z + 1) for z in slow) + " cool >10% slower") if slow else "all zones cool normally",
                     "Slowest zone vs fault-free": f"{plain['t78_ratio'].max():.2f}×"})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def benchmark_curves():
    hours, measured = benchmark.load_measured()
    out = []
    for speed in benchmark.SPEEDS:
        m = measured[speed]
        t_air, t_start = float(m[-12:].mean()), float(m[0].mean())
        th, cells = benchmark.predict(speed, t_start, t_air)
        keep = slice(None, None, 60)                       # one model point every 10 minutes
        out.append((speed, hours, m, th[keep], cells[keep]))
    return out


if "zones" not in st.session_state:
    load_preset("normal")

st.title("PreCool Pulse: is the cold air actually cooling the produce?")
st.warning(f"**{LABEL}.** This is a physics simulation of a proposed sensor rail. The rail has not been built "
           "and no value on this page was measured by the team.", icon="⚠️")

# ---------------- inputs ----------------
with st.sidebar:
    st.header("Inputs")
    preset = st.selectbox("Start from a preset case", list(SCENARIOS), format_func=lambda n: f"{n}: {SCENARIOS[n]['desc']}")
    if st.button("Load preset inputs"):
        load_preset(preset)
        st.rerun()
    st.caption("A preset only fills in the inputs below. Every result is recomputed from the inputs.")
    st.subheader("Air and produce")
    st.slider("Supply air temperature (°C)", -2.0, 10.0, key="t_supply", step=0.5)
    st.slider("Produce per zone (kg)", 20.0, 300.0, key="mass", step=10.0)
    st.slider("Design airflow per zone (L/s)", 20.0, 300.0, key="flow", step=10.0)
    st.slider("Fan strength (1 = healthy)", 0.3, 1.0, key="fan_health", step=0.05)
    st.slider("Hours simulated", 4, 24, key="hours")
    st.checkbox("Produce also loses heat by evaporation (illustrative)", key="evaporation")
    st.subheader("Sensor rail (proposed, not built)")
    st.slider("Return air mixing between neighbouring zones", 0.0, 0.45, key="cross_mix", step=0.05)
    st.slider("How much of a gap's jet the zone sensor sees (1 = fully mixed)", 0.0, 2.0, key="jet_capture", step=0.1)
    st.slider("Temperature noise (K)", 0.0, 1.0, key="sigma_t", step=0.05)
    st.slider("Speed noise (m/s)", 0.0, 0.3, key="sigma_v", step=0.01)
    st.slider("Unknown fixed speed offset per sensor (m/s)", 0.0, 0.3, key="v_bias", step=0.01)
    st.checkbox("Operator probes one fruit per zone at loading (pulp temperature)", key="use_pulp")
    st.slider("Error of that pulp reading (K)", 0.0, 4.0, key="pulp_sigma", step=0.5)
    st.number_input("Random seed", 0, 99999, key="seed", step=1)

with st.expander("Zone by zone: start temperature, faults and sensor condition (click a cell to edit)", expanded=False):
    zones = st.data_editor(
        st.session_state["zones"], key=f"zones_editor_{st.session_state['zones_version']}", hide_index=True, width="stretch",
        column_config={
            "Zone": st.column_config.TextColumn(disabled=True),
            "Start temp (°C)": st.column_config.NumberColumn(min_value=3.0, max_value=45.0, step=0.5),
            "Vent resistance (×)": st.column_config.NumberColumn(min_value=0.2, max_value=1000.0, step=0.5),
            "Open gap (m²)": st.column_config.NumberColumn(min_value=0.0, max_value=0.5, step=0.01),
            "In-carton bypass (0–1)": st.column_config.NumberColumn(min_value=0.0, max_value=0.95, step=0.05),
            "Speed sensor gain": st.column_config.NumberColumn(min_value=0.1, max_value=3.0, step=0.05),
        })

s = st.session_state
try:
    if zones.isna().any().any():
        raise ValueError("every cell in the zone table needs a value")
    cfg = Config(t_supply=s.t_supply, t_init=zones["Start temp (°C)"].tolist(), mass_per_zone=s.mass,
                 design_flow_per_zone=s.flow / 1000, fan_health=s.fan_health, duration=s.hours * 3600.0,
                 resistance_mult=zones["Vent resistance (×)"].tolist(), gap_area=zones["Open gap (m²)"].tolist(),
                 pkg_bypass=zones["In-carton bypass (0–1)"].tolist(), transpiration=5e-10 if s.evaporation else 0.0)
    if min(cfg.t_init) <= cfg.t_supply:
        raise ValueError("every zone's start temperature must be above the supply air temperature")
    sens = SensorConfig(jet_capture=s.jet_capture, cross_mix=s.cross_mix, sigma_t=s.sigma_t, sigma_v=s.sigma_v,
                        v_gain=zones["Speed sensor gain"].tolist(), v_bias_sigma=s.v_bias, seed=int(s.seed),
                        failed_zones=tuple(z for z in range(8) if not zones["Sensors working"].iloc[z]))
    r = evaluate(cfg, sens, use_pulp=s.use_pulp, pulp_sigma=s.pulp_sigma)
except ValueError as err:
    st.error(f"These inputs cannot be simulated: {err}")
    st.stop()

diag, res, ref = r["diag"], r["res"], r["ref"]

# ---------------- what the display would say ----------------
st.subheader("What the operator display would say (computed from the simulated sensor readings)")
alarms = [f"**Zone {z + 1}: {diag['labels'][z]}** — {STATUS[diag['labels'][z]][2]}" for z in diag["flagged"]]
if diag["status"] == "CHECK ZONE":
    st.error("**CHECK ZONE**\n\n" + "\n\n".join(alarms) + ("\n\nWhole batch is also below its fault-free baseline." if diag["batch_low"] else ""), icon="✖️")
elif diag["status"] == "BATCH LOW":
    st.warning("**BATCH LOW** — no single zone stands out, but the whole batch picks up less heat than its fault-free baseline.", icon="🔻")
else:
    st.success("**PASS** — no zone stands out and the batch matches its fault-free baseline.", icon="✅")

slow = r["truly_slow"]
if len(slow) == 1:
    truth = f"zone {slow[0] + 1} takes more than 10% longer to cool than without faults"
elif slow:
    truth = "zones " + ", ".join(str(z + 1) for z in slow) + " take more than 10% longer to cool than without faults"
else:
    truth = "every zone cools as fast as without faults"
if any(v.startswith("MISSED") or v.startswith("alarm, but") for v in r["verdicts"]):
    agreement = "**The display is misleading here** — see the zone table."
elif any(v.startswith("no zone alarm; slow") for v in r["verdicts"]):
    agreement = ("**The zone labels understate it:** some zones cool slowly without a zone alarm of their own; "
                 "only the whole-batch check shows it.")
else:
    agreement = "The display and the produce agree."
st.info(f"**What really happens to the produce in the model:** {truth}. {agreement}", icon="🔎")

col = st.columns(4)
col[0].metric("Pressure across the load", f"{res.air.dp:.0f} Pa", f"{res.air.dp - ref.air.dp:+.0f} Pa vs fault-free", delta_color="off", delta_arrow="off")
col[1].metric("Air reaching the produce", f"{res.air.m_contact.sum() / ref.air.m_contact.sum():.0%}", "of fault-free", delta_color="off", delta_arrow="off")
col[2].metric("Slowest zone, 7/8 cooling time", f"{r['t78_h'].max():.1f} h" if np.isfinite(r["t78_h"].max()) else "not reached",
              f"{r['t78_ratio'].max():.2f}× fault-free" if np.isfinite(r["t78_ratio"].max()) else None, delta_color="off", delta_arrow="off")
col[3].metric("Heat leaving as water vapour", f"{res.energy_latent().sum() / res.energy_product().sum():.0%}", "unseen by air sensors", delta_color="off", delta_arrow="off")

tab_map, tab_time, tab_fail, tab_repeat, tab_evidence = st.tabs(
    ["Zone map", "Over time", "When the signal fails", "Repeat with random sensor errors", "Evidence and export"])

# ---------------- zone map ----------------
with tab_map:
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Bar(
            y=ZONES, x=[1] * 8, orientation="h", marker_color=[STATUS[lab][0] for lab in diag["labels"]],
            marker_line=dict(color=SURFACE, width=2), text=[f"{STATUS[lab][1]} {lab}" for lab in diag["labels"]],
            textposition="outside", textfont=dict(color=INK, size=13), cliponaxis=False,
            customdata=np.stack([diag["v"], diag["dT"]], axis=1),
            hovertemplate="%{y}<br>air speed %{customdata[0]:.2f} m/s<br>heat pickup %{customdata[1]:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(visible=False, range=[0, 4.2])
        show(style(fig, "Computed zone status", height=440, unified=False, left=110, top=70))
    with right:
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=("Air speed ÷ batch median", "Heat pickup ÷ batch median", "UCI ÷ batch median"))
        for i, key in enumerate(("r_v", "r_dT", "r_uci"), start=1):
            fig.add_vrect(x0=0.8, x1=1.2, fillcolor=GRID, opacity=0.6, line_width=0, row=1, col=i)
            fig.add_trace(go.Bar(y=ZONES, x=diag[key], orientation="h", marker_color=BLUE, width=0.55, showlegend=False,
                                 marker_line=dict(color=SURFACE, width=2), hovertemplate="%{y}: %{x:.2f}×<extra></extra>"), row=1, col=i)
            fig.update_xaxes(range=[0, max(2.0, float(np.nanmax(diag[key])) * 1.05)], row=1, col=i)
        fig.update_yaxes(autorange="reversed")
        show(style(fig, "The readings behind the status (grey band = within ±20% of the batch median)", height=440, unified=False, left=110, top=110))
    yes_no = lambda idx: ["flag" if z in idx else "–" for z in range(8)]
    st.dataframe(pd.DataFrame({
        "Zone": ZONES, "Display says": [f"{STATUS[lab][1]} {lab}" for lab in diag["labels"]],
        "Display vs produce": r["verdicts"],
        "Air speed (m/s)": np.round(diag["v"], 2),
        "Heat pickup (share of pulp–supply gap)" if s.use_pulp else "Heat pickup (K)": np.round(diag["dT"], 2),
        "UCI only": yes_no(diag["uci_only"]), "Temperature only": yes_no(diag["temp_only"]), "Speed only": yes_no(diag["vel_only"]),
        "7/8 cooling time (h)": np.round(r["t78_h"], 2), "vs fault-free": [f"{x:.2f}×" for x in r["t78_ratio"]]}),
        hide_index=True, width="stretch", column_config={"Display vs produce": st.column_config.TextColumn(width="large")})
    st.caption(f"{LABEL}. 'UCI only', 'Temperature only' and 'Speed only' are simpler detectors shown for comparison. "
               "'Display vs produce' checks each label against the simulated produce temperatures, not against the fault that was set.")

# ---------------- over time ----------------
with tab_time:
    hours = res.t / 3600
    at = st.slider("Replay: elapsed time (h)", 0.0, float(s.hours), min(2.0, float(s.hours)), step=0.25)
    step = slice(None, None, 12)                                    # one point per minute
    fig = go.Figure()
    for z in range(8):
        fig.add_trace(go.Scatter(x=hours[step], y=res.t_prod_mean[step, z], name=ZONES[z], mode="lines",
                                 line=dict(color=ZONE_COLORS[z], width=2), hovertemplate="%{y:.1f} °C"))
    fig.add_trace(go.Scatter(x=hours[step], y=ref.t_prod_mean[step].mean(axis=1), name="Fault-free average", mode="lines",
                             line=dict(color=MUTED, width=2, dash="dash"), hovertemplate="%{y:.1f} °C"))
    fig.add_vline(x=at, line=dict(color=INK, width=1))
    fig.update_xaxes(title="Hours since fan start")
    fig.update_yaxes(title="Average produce temperature (°C)")
    show(style(fig, "Produce temperature in each zone", height=400))
    i = min(int(at * 3600 / cfg.dt), len(res.t) - 1)
    st.dataframe(pd.DataFrame([np.round(res.t_prod_mean[i], 1), np.round(ref.t_prod_mean[i], 1), np.round(res.q[i] + res.q_latent[i])],
                              index=[f"Produce at {at:.2f} h (°C)", "Fault-free (°C)", "Heat removal now (W)"], columns=ZONES),
                 width="stretch")
    obs = r["obs"]
    fig = go.Figure()
    for z in range(8):
        fig.add_trace(go.Scatter(x=obs.t / 60, y=obs.t_return[:, z], name=ZONES[z], mode="lines",
                                 line=dict(color=ZONE_COLORS[z], width=2), hovertemplate="%{y:.1f} °C"))
    fig.add_vrect(x0=1, x1=5, fillcolor=GRID, opacity=0.5, line_width=0, annotation_text="detector averages this window",
                  annotation_position="top left", annotation_font_color=MUTED)
    fig.update_xaxes(title="Minutes since fan start")
    fig.update_yaxes(title="Return air temperature reading (°C)")
    show(style(fig, "Simulated return-air sensor readings, opening minutes", height=380))

# ---------------- when the signal fails ----------------
with tab_fail:
    st.markdown("These cases are run through exactly the same code as the page above. In each one the display and "
                "the produce disagree, or agree only because of an extra measurement.")
    st.dataframe(counterexample_rows(), hide_index=True, width="stretch")
    st.markdown(
        "- **A tarp gap starves every zone equally**, so UCI is the same everywhere. Only the pair of readings "
        "(speed up, heat pickup down) points at the gap, and a small gap alarms loudly for almost no harm.\n"
        "- **Mixing of return air** makes a neighbouring zone look faulty; **a sensor outside the jet** sees nothing.\n"
        "- **Warm produce can hide a bypassing zone** and **a cool pallet can look faulty**. A pulp temperature per zone fixes both.\n"
        "- **A drifting speed sensor** looks like a restricted zone. **A weak fan** is invisible without a stored fault-free baseline.")
    st.caption(LABEL)

# ---------------- repeat with random sensor errors ----------------
with tab_repeat:
    repeats = 100
    counts, any_alarm = np.zeros(8), 0
    for seed in range(repeats):
        flagged = evaluate(cfg, replace(sens, seed=seed), use_pulp=s.use_pulp, pulp_sigma=s.pulp_sigma)["diag"]["flagged"]
        counts[flagged] += 1
        any_alarm += bool(flagged)
    st.metric(f"Runs with at least one zone alarm (of {repeats})", f"{any_alarm}")
    fig = go.Figure(go.Bar(x=ZONES, y=counts / repeats * 100, marker_color=BLUE, width=0.55,
                           marker_line=dict(color=SURFACE, width=2), hovertemplate="%{x}: flagged in %{y:.0f}% of runs<extra></extra>"))
    fig.update_yaxes(title="Flagged in % of runs", range=[0, 105])
    show(style(fig, f"How often each zone is flagged over {repeats} runs with fresh random sensor errors", height=360, unified=False))
    st.caption(f"{LABEL}. Same inputs each time; only the random sensor noise, sensor offsets and pulp-reading errors change. "
               "A zone flagged in some runs but not others means the result depends on sensor luck.")

# ---------------- evidence and export ----------------
with tab_evidence:
    st.markdown("**How far the evidence goes**")
    st.dataframe(pd.DataFrame({
        "Level": ["A", "B", "C", "D"],
        "Question": ["Does the code agree with itself?", "Does the model conserve mass and energy?",
                     "Does it match an independent published experiment?", "Does the sensor rail detect real faults?"],
        "Status": ["Done: automated tests", "Done: balance checks and an exact-solution check",
                   "Partial: produce cooling only, one published dataset (below)", "Not available: needs a physical rig"]}),
        hide_index=True, width="stretch")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"{v:.2f} m/s" for v in benchmark.SPEEDS])
    for i, (speed, mh, m, th, cells) in enumerate(benchmark_curves(), start=1):
        first = i == 1
        fig.add_trace(go.Scatter(x=th, y=cells[:, -1], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter(x=th, y=cells[:, 0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=ORANGE_BAND,
                                 name="Model: inlet-to-outlet range", showlegend=first, hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter(x=th, y=cells.mean(axis=1), mode="lines", line=dict(color=ORANGE, width=2),
                                 name="Model: bed average (untuned)", showlegend=first, hovertemplate="model %{y:.1f} °C"), row=1, col=i)
        for k, dash in ((0, "solid"), (1, "dot")):
            fig.add_trace(go.Scatter(x=mh, y=m[:, k], mode="lines", line=dict(color=BLUE, width=2, dash=dash),
                                     name=f"Measured pulp, probe {k + 1}", showlegend=first, hovertemplate="measured %{y:.1f} °C"), row=1, col=i)
        fig.update_xaxes(title="Hours", range=[0, 16], row=1, col=i)
    fig.update_yaxes(title="Temperature (°C)", row=1, col=1)
    fig = style(fig, "Cooling model against published measurements (Valencia oranges, laboratory tunnel)", height=460, top=110)
    fig.update_layout(title_text="Cooling model against published measurements (Valencia oranges, laboratory tunnel)"
                                 "<br><sup>Blue = PUBLISHED MEASURED (Sadie et al. 2025, CC BY 4.0). Orange = MODEL-GENERATED, no fitted rate parameters.</sup>")
    fig.update_layout(legend=dict(y=-0.34, yanchor="top"))
    show(fig)
    st.caption("Measured data: Sadie, Berry, Coetzee, Owen, Kleyn (2025), Zenodo, doi:10.5281/zenodo.15726691, CC BY 4.0. "
               "This checks how produce cools. It does not test the airflow model, the sensors or the detector.")

    st.markdown("**Export this run**")
    series = pd.DataFrame(_timeseries(r))
    summary = {"label": LABEL, "status": diag["status"], "batch_low": diag["batch_low"],
               "zone_labels": dict(zip(ZONES, diag["labels"])), "display_vs_produce": dict(zip(ZONES, r["verdicts"])),
               "truly_slow_zones": [z + 1 for z in slow], "seven_eighths_time_h": [round(float(x), 3) for x in r["t78_h"]],
               "inputs": {"physics": asdict(cfg), "sensors": asdict(sens), "use_pulp_probe": s.use_pulp, "pulp_sigma_K": s.pulp_sigma},
               "data_kind": "MODEL-GENERATED"}
    a, b = st.columns(2)
    a.download_button("Download sensor and produce time series (CSV)", f"# {LABEL}\n" + series.to_csv(index=False),
                      file_name="precool_pulse_timeseries.csv", mime="text/csv")
    b.download_button("Download result and all inputs (JSON)", json.dumps(summary, indent=2, default=float),
                      file_name="precool_pulse_run.json", mime="application/json")
