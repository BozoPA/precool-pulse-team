"""One-off: copy the lab forced-air trial sheet from the Zenodo xlsx into a plain CSV (values unchanged).
Run with a Python that has openpyxl:  python convert_xlsx_to_csv.py
Source: Sadie et al. (2025), Precooling Trial Data, Zenodo, doi:10.5281/zenodo.15726691, CC BY 4.0."""
import csv
from pathlib import Path

import openpyxl

here = Path(__file__).parent
rows = list(openpyxl.load_workbook(here / "zenodo_15726691" / "Laboratory Trial FAC Temp Data.xlsx",
                                   data_only=True, read_only=True).worksheets[0].iter_rows(values_only=True))
with open(here / "lab_fac_pulp_10min.csv", "w", newline="") as f:
    f.write("# PUBLISHED MEASURED DATA. Sadie et al. (2025) doi:10.5281/zenodo.15726691, CC BY 4.0. "
            "Pulp temperature (C) of Valencia oranges, 10-minute intervals; columns = air speed and sensor placement.\n")
    w = csv.writer(f)
    w.writerow(["minutes", "0.05_bottom", "0.05_top", "0.20_bottom", "0.20_top", "0.35_bottom", "0.35_top"])
    for i, r in enumerate(rows[2:]):
        if all(isinstance(c, (int, float)) for c in r[1:7]):
            w.writerow([i * 10] + list(r[1:7]))
