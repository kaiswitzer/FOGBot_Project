# FOGbot (Web Version)

This project turns `FOGbot.py` into a simple website using **Streamlit**. Users upload files, click run, and download the Excel results.

## What the app does
- Trains a model on **past placements**
- Predicts placements for the **current master list**
- Enforces **location capacity**
- Outputs: `FGC_Final_Placements_Results.xlsx`

## Inputs (upload on the website)
All uploads can be **Excel (`.xlsx`) or CSV (`.csv`)**.

### 1) Past Placements with Language (training)
Required columns:
- `Placement Decision`
- `Gender`
- `Major`
- `Top Location Preference`
- `Application Score`
- `Resume Score`
- `Video Score`
- Either `Language` **or** `Can you speak a language other than English?`
- Either `Academic Level` **or** `AcademicLevel`

Formatting notes:
- Each row should represent **one past student**.
- Keep spelling/capitalization consistent for categories (especially `Placement Decision`).
- Scores should be numeric (blanks are allowed, but better data improves results).

### 2) Current Master List (to place)
Should contain the same feature columns as the training file (everything above **except** `Placement Decision`).

Formatting notes:
- Each row should represent **one current student**.
- The master list should **not** include `Placement Decision` (that column is what the model predicts).

### 3) Location capacity
Required columns:
- `Placement Location`
- `Capacity`

Formatting notes:
- `Capacity` must be numeric and \(\\ge 0\).
- Location names should match what you expect to assign (consistent spelling).

## “Minimum training examples per placement” (website setting)
A *placement* is each unique value in the training column `Placement Decision`.

Before training, the app filters out rare placement labels using the threshold you set:
- Placements with **≤ threshold** rows in the training file are removed (the code uses a strict `>` filter).
- This can improve stability by avoiding extremely rare labels, but it also reduces coverage: the model cannot predict labels that were removed.

## Run locally (no conda)
From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CLI script (optional)
`FOGbot.py` still works for local runs. It supports CSV or Excel for the standard filenames:
- `Past Placements with Language.csv` or `Past Placements with Language.xlsx`
- `*Master List.csv` or `*Master List.xlsx` (it picks the newest one)
- `location_capacity.csv` or `location_capacity.xlsx`

Run:

```bash
python FOGbot.py
```

## Deploy to Streamlit Community Cloud (recommended)
1. Create a GitHub repo and push this folder.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the app file to: `app.py`
4. Deploy.
Streamlit will install dependencies from `requirements.txt` automatically.

## Project structure
- `app.py`: Streamlit web UI
- `fogbot/core.py`: shared model + capacity logic
- `fogbot/io_utils.py`: Excel/CSV loader for uploads
- `FOGbot.py`: legacy CLI wrapper
