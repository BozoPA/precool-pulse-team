# Data manifest

| Item | Kind | Source | Licence | Changes made |
|---|---|---|---|---|
| `zenodo_15726691/*.xlsx`, `README.txt`, `LICENSE.txt` | PUBLISHED MEASURED (third party) | Sadie, E., Berry, T., Coetzee, C., Owen, M., Kleyn, M. (2025). *Precooling Trial Data* / *Cooling Tunnel System for Laboratory-Based Postharvest Precooling Trials*. Zenodo. https://doi.org/10.5281/zenodo.15726691 . Method: MethodsX (2026), https://doi.org/10.1016/j.mex.2026.103813 | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) | None. Downloaded 11 Nov 2026; hashes in `zenodo_15726691/SHA256SUMS.txt` |
| `lab_fac_pulp_10min.csv` | PUBLISHED MEASURED (reformatted) | Sheet "Lab FAC Trial Data" of `Laboratory Trial FAC Temp Data.xlsx` | CC BY 4.0, same attribution | Converted to CSV by `convert_xlsx_to_csv.py`; a minutes column and plain column names were added; values unchanged |
| everything in `../outputs/` | MODEL-GENERATED | this prototype | — | — |

Only `lab_fac_pulp_10min.csv` is used by the code so far. The other spreadsheets (commercial room, static
cooling, tunnel calibration, chilling-injury scores) are kept unmodified for later checks.
No data in this project was measured by the PreCool Pulse team.
