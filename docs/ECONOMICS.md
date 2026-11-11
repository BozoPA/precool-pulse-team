# Economics rework — 11 Nov 2026

Every figure here is MODEL-GENERATED from SCENARIO ASSUMPTIONS (`deliverables/prototype/precool_pulse/economics.py`,
`RANGES`). Nothing was metered. The only verified published input is the grid factor: 0.675 t CO2/MWh, CEA CO2
Baseline Database Version 22.0, FY 2025-26.

## 1. What was wrong in Phase 1 slide 9
* The arithmetic is right, but the baseline contradicts itself. A 6-tonne batch cooled by 25 K holds 154 kWh of
  field heat, which needs about 62 kWh of electricity at COP 2.5. The slide's whole batch uses 60 kWh.
* The headline (1,350 to 4,050 kWh a year) assumes all 15 kW stop when a batch ends early. The slide itself
  calls that an upper bound. The field heat has to be removed whatever the batch length.

## 2. Corrected batch energy (base assumptions)
| Part | kWh per batch |
|---|---|
| Field heat to remove (thermal) | 154.2 |
| Fan electricity, 4 kW for 4 h | 16.0 |
| Compressor electricity at COP 2.5 (field heat + 20% other gains + fan heat) | 80.4 |
| **Total electricity** | **96.4** (16 kWh per tonne, 24 kW average) |
Across the assumption ranges: 74 to 113 kWh per batch (10th to 90th percentile).

## 3. What ending a batch earlier can save
Only the fan's electricity and the fan's own heat are avoidable. Expected saving per year =
batches x share of batches with a fixable fault x share found and fixed x minutes lost per fault x fan power x (1 + 1/COP).

| Output | 10th percentile | Median | 90th percentile |
|---|---|---|---|
| Electricity saved, kWh a year | 104 | 222 | 435 |
| Money saved, ₹ a year | 815 | 1,763 | 3,486 |
| CO2 avoided, t a year | 0.07 | 0.15 | 0.29 |
| Share of the precooler's yearly electricity | 0.2% | 0.5% | 1.0% |
| Payback of a ₹50,000 to ₹1,00,000 kit on energy alone, years | 18 | 37 | 83 |
(20,000 random draws, triangular ranges, fixed seed.)

* Even the optimistic end (435 kWh) is a third of Phase 1's lowest figure (1,350 kWh).
* If batches run on a fixed timer, the energy saving is zero; the benefit is colder, more even produce.
* The three inputs that matter most have no data behind them: how often faults occur, how often they get
  found and fixed, and how much time each costs.

## 4. Where the value would have to come from
At full load a 6-tonne precooler handles 3,240 t a year (Phase 1 assumption). At ₹40/kg (Phase 1 assumption),
cutting produce loss by **0.039%** repays a ₹50,000 kit in one season; 0.1% is worth ₹1.3 lakh a year.
Nobody has measured whether better airflow checks cut losses at all. This is the claim a pilot must test.

## 5. What may be said
* "Energy savings alone do not pay for the kit under our assumptions. The business case rests on produce
  quality and throughput, which a pilot must measure."
* Not: any rupee, kWh or CO2 saving as a result.

## 6. Still to verify
Fan power and COP of a real 6-tonne precooler; fault frequency in real packhouses; NHB cost norm; produce price.
