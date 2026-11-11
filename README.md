# PreCool Pulse: is the cold air actually cooling the produce?

Phase 2 prototype for the Yuva Yodha Energy Tech Hackathon 2026 (Challenge 01, Sustainable Agriculture).
Team: Parth Agrawal (team lead), Harshit Jain, Gorima Ghosh, Nilesh Sharma. IMI New Delhi.

> **SIMULATED / ILLUSTRATIVE — NOT FIELD-VALIDATED.** This repository is working *software*: a physics
> simulation of a proposed sensor rail for forced-air precoolers. The rail has not been built. No value
> here was measured by our team. Nothing in this repository proves the rail works in a packhouse.

## The idea being tested
A clip-on rail at the return-air face of a precooling load would read air temperature and air speed in
eight zones. Phase 1 proposed that these readings show which parts of the load the cold air is really
cooling. Before building hardware we wrote this simulation to find out when that is true and when it is not.

## What the software does
* Models the fan, the air paths (through the cartons, around the produce, through tarp gaps) and the
  cooling of the produce. Mass and energy are conserved.
* Models what the proposed sensors would read, including mixing between zones, noise, drift and dead sensors.
* Applies the detection logic (PASS / CHECK ZONE / BATCH LOW) and three simpler detectors.
* Checks every alarm against the simulated produce temperatures, not against the fault that was set.
* Compares the cooling model, with nothing fitted, to published measurements of oranges.
* Re-computes the energy case with stated assumptions and their uncertainty.

## What we found (all simulated)
* The signal finds restricted vents and air that bypasses produce inside a carton.
* It can be fooled: a tarp gap starves every zone equally; mixed return air blames a neighbour; warm produce
  can hide a faulty zone. See the "When the signal fails" tab and `docs/VALIDATION_STATUS.md`.
* One pulp temperature per zone at loading removes the largest blind spot.
* The air-speed sensor is the first hardware risk: its published accuracy is close to the air speeds involved.
* Energy savings alone would not pay for the kit; see `docs/ECONOMICS.md`.

## Run it
Python 3.11 or newer. Tested with Python 3.13 on Linux. Not yet tested on Windows or macOS.
```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python -m pytest tests -q                 # about one minute
.venv/bin/python -m precool_pulse list
.venv/bin/python -m precool_pulse run blockage --seed 1 --out outputs
.venv/bin/python -m precool_pulse all --out outputs
.venv/bin/streamlit run app.py                      # dashboard at http://localhost:8501
```
An ordinary laptop is enough. No internet, account or API key is needed after installation.

## Layout
| Path | Contents |
|---|---|
| `precool_pulse/` | `physics.py`, `sensing.py`, `detect.py`, `scenarios.py`, `study.py`, `economics.py`, `benchmark.py`, `cli.py` |
| `app.py` | Dashboard. Every status is recomputed from the inputs on screen |
| `tests/` | Automated tests: conservation, an exact solution, scenarios, failure cases, benchmark, dashboard |
| `benchmark_data/` | Published measurements by others (CC BY 4.0) and the data manifest |
| `sample_outputs/` | Results of one full run, for comparison with yours |
| `docs/` | Architecture diagram, validation status, economics note, screenshots |

## How far the evidence goes
| Level | Question | Status |
|---|---|---|
| A | Does the code agree with itself? | Done: automated tests |
| B | Does the model conserve mass and energy? | Done |
| C | Does it match an independent published experiment? | Partial: produce cooling only, one dataset |
| D | Does the sensor rail detect real faults? | Not available: needs a physical rig |

## Licence and credit
Our code: MIT (see `LICENSE`). The benchmark dataset is not ours: see `THIRD_PARTY.md`.

**Project ownership and development:** The PreCool Pulse concept and project direction belong to the four-member IMI New Delhi team. AI-assisted tools supported software implementation and documentation. The team remains responsible for reviewing the code, checking the evidence, and accurately presenting its limitations.
