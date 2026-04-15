from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from fogbot import run_placements
from fogbot.io_utils import read_table


def _df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _missing_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    cols = set(df.columns.astype(str))
    return [c for c in required if c not in cols]


def _has_any_column(df: pd.DataFrame, options: list[str]) -> bool:
    cols = set(df.columns.astype(str))
    return any(o in cols for o in options)


def _validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, capacity_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    # Capacity
    cap_missing = _missing_required_columns(capacity_df, ["Placement Location", "Capacity"])
    if cap_missing:
        errors.append(f"Location capacity file is missing required columns: {', '.join(cap_missing)}.")
    else:
        cap_numeric = pd.to_numeric(capacity_df["Capacity"], errors="coerce")
        if cap_numeric.isna().any():
            errors.append("Location capacity `Capacity` must be numeric (no blanks/text).")
        elif (cap_numeric < 0).any():
            errors.append("Location capacity `Capacity` must be 0 or greater.")

    # Training
    train_required = [
        "Placement Decision",
        "Gender",
        "Major",
        "Top Location Preference",
        "Application Score",
        "Resume Score",
        "Video Score",
    ]
    train_missing = _missing_required_columns(train_df, train_required)
    if train_missing:
        errors.append(f"Training file is missing required columns: {', '.join(train_missing)}.")
    if not _has_any_column(train_df, ["Academic Level", "AcademicLevel"]):
        errors.append("Training file must include `Academic Level` or `AcademicLevel`.")
    if not _has_any_column(train_df, ["Language", "Can you speak a language other than English?"]):
        errors.append("Training file must include `Language` or `Can you speak a language other than English?`.")

    # Test (master list)
    test_required = [c for c in train_required if c != "Placement Decision"]
    test_missing = _missing_required_columns(test_df, test_required)
    if test_missing:
        errors.append(f"Master list file is missing required columns: {', '.join(test_missing)}.")
    if "Placement Decision" in set(test_df.columns.astype(str)):
        errors.append("Master list file should NOT include `Placement Decision` (that column is only for training).")
    if not _has_any_column(test_df, ["Academic Level", "AcademicLevel"]):
        errors.append("Master list file must include `Academic Level` or `AcademicLevel`.")
    if not _has_any_column(test_df, ["Language", "Can you speak a language other than English?"]):
        errors.append("Master list file must include `Language` or `Can you speak a language other than English?`.")

    return errors


st.set_page_config(page_title="FOGbot Placement Tool", layout="centered")

st.title("FOGbot Placement Tool")
st.write(
    "Upload your files, click **Run placements**, then download the Excel results. "
    "Files can be **.xlsx** or **.csv**."
)

with st.expander("How to format your uploads", expanded=True):
    st.markdown(
        """
### 1) Past Placements with Language (training data)
- **What it is**: historical placements the model learns from (**one row per past student**).
- **Required columns**:
  - `Placement Decision` (the true/historical placement label)
  - `Gender`, `Major`, `Top Location Preference`
  - `Application Score`, `Resume Score`, `Video Score`
  - Academic level: either `Academic Level` **or** `AcademicLevel`
  - Language: either `Language` **or** `Can you speak a language other than English?`
- **Formatting notes**:
  - Keep category spelling consistent (e.g., the exact same location name each time in `Placement Decision`).
  - Scores should be numeric. Blanks are okay, but better data improves results.

### 2) Current Master List (to place)
- **What it is**: the list you want to assign placements to (**one row per current student**).
- **Required columns**: everything above **except** `Placement Decision` (because this file is what we’re predicting).
- **Formatting notes**:
  - Column names must match exactly (including spaces/capitalization), except for the accepted alternates listed above.

### 3) Location capacity
- **What it is**: a table that limits how many students can be finally assigned to each location.
- **Required columns**:
  - `Placement Location` (location name)
  - `Capacity` (a number)
- **Formatting notes**:
  - Location names should match what you expect to assign (same spelling as placement labels).
  - If a location isn’t listed here, it effectively has a very large capacity in the current logic.
"""
    )

train_file = st.file_uploader(
    "1) Past Placements with Language (training data)",
    type=["xlsx", "xls", "csv"],
    help=(
        "Must include `Placement Decision` plus the feature columns listed in “How to format your uploads”. "
        "Download a template below if you want a ready-made header row."
    ),
)
test_file = st.file_uploader(
    "2) Current Master List (to place)",
    type=["xlsx", "xls", "csv"],
    help="Must include the same feature columns as training, but should NOT include `Placement Decision`.",
)
capacity_file = st.file_uploader(
    "3) Location capacity",
    type=["xlsx", "xls", "csv"],
    help="Must include `Placement Location` and `Capacity` (numeric).",
)

min_examples = st.number_input(
    "Minimum training examples per placement (filter threshold)",
    min_value=0,
    max_value=1000,
    value=10,
    step=1,
    help=(
        "A *placement* is each unique value in the training column `Placement Decision`.\n\n"
        "Before training, any placement label with **≤ this many** historical rows is removed from the training set "
        "(the code uses a strict `>` filter). This improves stability by dropping very rare labels, but it also "
        "means the model cannot predict those dropped placements."
    ),
)

st.subheader("Download templates")
st.caption("Optional: use these as starting points so your column headers match exactly.")

_training_cols = [
    "Placement Decision",
    "Gender",
    "Major",
    "Top Location Preference",
    "Application Score",
    "Resume Score",
    "Video Score",
    "Academic Level",
    "Language",
]
_master_cols = [c for c in _training_cols if c != "Placement Decision"]
_capacity_cols = ["Placement Location", "Capacity"]

training_template = pd.DataFrame([{c: "" for c in _training_cols} for _ in range(3)])
master_template = pd.DataFrame([{c: "" for c in _master_cols} for _ in range(3)])
capacity_template = pd.DataFrame(
    [
        {"Placement Location": "", "Capacity": 0},
        {"Placement Location": "", "Capacity": 0},
        {"Placement Location": "", "Capacity": 0},
    ]
)

colA, colB, colC = st.columns(3)
with colA:
    st.download_button(
        "Training template (XLSX)",
        data=_df_to_xlsx_bytes(training_template, sheet_name="Training"),
        file_name="fogbot_training_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Training template (CSV)",
        data=_df_to_csv_bytes(training_template),
        file_name="fogbot_training_template.csv",
        mime="text/csv",
    )
with colB:
    st.download_button(
        "Master list template (XLSX)",
        data=_df_to_xlsx_bytes(master_template, sheet_name="MasterList"),
        file_name="fogbot_master_list_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Master list template (CSV)",
        data=_df_to_csv_bytes(master_template),
        file_name="fogbot_master_list_template.csv",
        mime="text/csv",
    )
with colC:
    st.download_button(
        "Capacity template (XLSX)",
        data=_df_to_xlsx_bytes(capacity_template, sheet_name="Capacity"),
        file_name="fogbot_capacity_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Capacity template (CSV)",
        data=_df_to_csv_bytes(capacity_template),
        file_name="fogbot_capacity_template.csv",
        mime="text/csv",
    )

run = st.button("Run placements", type="primary", disabled=not (train_file and test_file and capacity_file))

if run:
    try:
        train_df = read_table(train_file)
        test_df = read_table(test_file)
        capacity_df = read_table(capacity_file)

        validation_errors = _validate_inputs(train_df=train_df, test_df=test_df, capacity_df=capacity_df)
        if validation_errors:
            st.error("Fix the following issues, then try again:")
            for msg in validation_errors:
                st.write(f"- {msg}")
            st.stop()

        results_df, assigned_counts = run_placements(
            train_df=train_df,
            test_df=test_df,
            capacity_df=capacity_df,
            min_training_examples_per_location=int(min_examples),
        )

        output = BytesIO()
        results_df.to_excel(output, index=False)
        output.seek(0)

        st.success("Done. Results are ready to download.")

        unassigned = int((results_df["Final_Placement"] == "Unassigned").sum())
        st.metric("Unassigned students", unassigned)

        st.subheader("Results preview")
        st.caption("Review the output here (sort/filter) without downloading.")
        st.dataframe(results_df, width="stretch")

        with st.expander("What the output columns mean", expanded=False):
            st.markdown(
                """
**Inputs (from the master list file)**
- **`Top Location Preference`**: The student’s self-reported top-choice location. This is used as a model input feature.

**Model outputs**
- **`Predicted_Placement`**: The model’s top predicted placement label (#1).
- **`Top2_Class`**: The model’s second choice placement label (#2).
- **`Top3_Class`**: The model’s third choice placement label (#3).

**Capacity-aware assignment**
- **`Final_Placement`**: The final assignment after enforcing capacity limits. For each student, the app tries `Predicted_Placement`, then `Top2_Class`, then `Top3_Class` until it finds a location that still has open capacity; otherwise it becomes `Unassigned`.

**Note**
- If a location is not present in the capacity table, the current logic treats it as having a very large capacity.
"""
            )

        with st.expander("Assignment counts (including Unassigned)"):
            counts_df = (
                pd.DataFrame(sorted(assigned_counts.items()), columns=["Placement", "Count"])
                .sort_values("Count", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(counts_df, width="stretch")

        st.download_button(
            "Download results Excel",
            data=output,
            file_name="FGC_Final_Placements_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(str(e))

