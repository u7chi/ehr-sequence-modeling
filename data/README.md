# Data setup

This framework expects three private pivot CSVs (not committed). Put them here (or
anywhere) and set the paths in `configs/*.yaml` under `data:`.

| file | columns | notes |
|---|---|---|
| `ccs_pivot_5_years_selected.csv` | `ID, age, gender, date, CAD` + 276 binary CCS diagnosis codes | one row per patient-date |
| `lab_pivot_5_years_selected_clean.csv` | `ID, age, gender, date, CAD` + 80 lab features | `-1` = not measured |
| `indexdate.csv` | `ID, indexDate` | per-patient prediction anchor |

`CAD` is the binary label (constant per patient). `date` is the event date;
recency in `set`/`qtime` mode is measured as days before `indexDate`.

To use data that lives elsewhere, either symlink it here:

```bash
ln -s /abs/path/ccs_pivot_5_years_selected.csv        ccs_pivot_5_years_selected.csv
ln -s /abs/path/lab_pivot_5_years_selected_clean.csv  lab_pivot_5_years_selected_clean.csv
ln -s /abs/path/indexdate.csv                         indexdate.csv
```

or edit `data.ccs_path` / `data.lab_path` / `data.index_path` in the config.
