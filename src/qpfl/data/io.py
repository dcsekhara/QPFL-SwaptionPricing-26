from pathlib import Path
import re

import numpy as np
import pandas as pd


def read_data(
    path,
    sheet_name=0,
    date_col="Date",
    output="long",
    value_name="value",
    engine="openpyxl",
):
    """Read Excel data and parse surface columns into Tenor and Maturity (months)."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    assert date_col in df.columns, f"Expected date column '{date_col}' not found in the data."

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y")

    pattern = re.compile(
        r"^\s*Tenor\s*:\s*(\d+)\s*;\s*Maturity\s*:\s*([0-9]*\.?[0-9]+)\s*$"
    )
    parsed_cols = {}
    parse_ignore_cols = [date_col]
    for col in df.columns:
        if col in parse_ignore_cols:
            continue
        match = pattern.match(str(col))
        if match:
            tenor = int(match.group(1))
            maturity_float = float(match.group(2))
            maturity_months = maturity_float * 12
            int_maturity_months = int(round(maturity_months))
            assert (
                abs(maturity_months - int_maturity_months) < 1e-6
            ), f"Maturity {maturity_float} years does not convert to an integer number of months."
            maturity_months = int_maturity_months
            parsed_cols[col] = (tenor, maturity_months)
        else:
            assert False, "Unable to parse the column name: " + str(col)

    if not parsed_cols:
        raise ValueError("No columns matched the 'Tenor : x; Maturity : y' format.")

    if output == "wide":
        raise NotImplementedError("Wide format is not implemented yet.")

    if output == "long":
        id_vars = [date_col]
        value_vars = list(parsed_cols.keys())
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="swaption_details",
            value_name=value_name,
        )
        long_df[["Tenor", "Maturity"]] = (
            long_df["swaption_details"].map(parsed_cols).apply(pd.Series)
        )
        long_df = long_df.drop(columns=["swaption_details"])
        ordered_cols = id_vars + ["Tenor", "Maturity", value_name]
        long_df = long_df[ordered_cols]

        assert long_df[date_col].notna().all(), "Date column contains missing values."
        assert long_df["Tenor"].notna().all(), "Tenor column contains missing values."
        assert long_df["Maturity"].notna().all(), "Maturity column contains missing values."

        assert np.issubdtype(
            long_df[date_col].dtype, np.datetime64
        ), "Date column is not of datetime type."
        assert np.issubdtype(
            long_df["Tenor"].dtype, np.integer
        ), "Tenor column is not of integer type."
        assert np.issubdtype(
            long_df["Maturity"].dtype, np.integer
        ), "Maturity column is not of integer type."
        assert np.issubdtype(
            long_df[value_name].dtype, np.number
        ), f"{value_name} column is not of numeric type."
        return long_df[ordered_cols]

    raise ValueError("output must be either 'long' or 'wide'")

