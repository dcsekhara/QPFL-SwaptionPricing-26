import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_tenor_maturity_lines(
    long_df,
    n_days=15,
    seed=42,
    max_tenors=10,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
):
    """
    For random days, plot Price vs Maturity with multiple traces (one trace per Tenor).
    Same tenor keeps the same color across all subplots.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    unique_dates = np.array(sorted(df[date_col].unique()))
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available after cleaning.")

    n_days = min(n_days, len(unique_dates))
    rng = np.random.default_rng(seed)
    sampled_dates = np.sort(rng.choice(unique_dates, size=n_days, replace=False))

    n_cols = 3
    n_rows = int(np.ceil(n_days / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    tenor_values = sorted(df[tenor_col].unique())[:max_tenors]
    palette = px.colors.qualitative.Dark24
    tenor_colors = {t: palette[i % len(palette)] for i, t in enumerate(tenor_values)}

    for i, d in enumerate(sampled_dates):
        row = i // n_cols + 1
        col = i % n_cols + 1
        day_df = df[df[date_col] == d]

        for tenor in tenor_values:
            tdf = day_df[day_df[tenor_col] == tenor].sort_values(maturity_col)
            if tdf.empty:
                continue

            color = tenor_colors[tenor]
            fig.add_trace(
                go.Scatter(
                    x=tdf[maturity_col],
                    y=tdf[price_col],
                    mode="lines+markers",
                    name=f"Tenor {tenor}Y",
                    legendgroup=f"tenor_{tenor}",
                    showlegend=(i == 0),
                    line=dict(width=2, color=color),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        "Maturity: %{x}m<br>"
                        "Price: %{y:.5f}<br>"
                        f"Tenor: {tenor}Y<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Maturity (months)", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(
        height=320 * n_rows,
        width=1200,
        title=f"Price vs Maturity with Multiple Tenor Traces (Random {n_days} Days)",
        template="plotly_white",
        legend_title="Tenor",
    )
    return fig, sampled_dates


def plot_maturity_tenor_lines(
    long_df,
    n_days=15,
    seed=42,
    max_maturities=12,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
):
    """
    For random days, plot Price vs Tenor with multiple traces (one trace per Maturity).
    Same maturity keeps the same color across all subplots.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    unique_dates = np.array(sorted(df[date_col].unique()))
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available after cleaning.")

    n_days = min(n_days, len(unique_dates))
    rng = np.random.default_rng(seed)
    sampled_dates = np.sort(rng.choice(unique_dates, size=n_days, replace=False))

    n_cols = 3
    n_rows = int(np.ceil(n_days / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in sampled_dates],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    maturity_values = sorted(df[maturity_col].unique())[:max_maturities]
    palette = px.colors.qualitative.Dark24
    maturity_colors = {m: palette[i % len(palette)] for i, m in enumerate(maturity_values)}

    for i, d in enumerate(sampled_dates):
        row = i // n_cols + 1
        col = i % n_cols + 1
        day_df = df[df[date_col] == d]

        for maturity in maturity_values:
            mdf = day_df[day_df[maturity_col] == maturity].sort_values(tenor_col)
            if mdf.empty:
                continue

            color = maturity_colors[maturity]
            fig.add_trace(
                go.Scatter(
                    x=mdf[tenor_col],
                    y=mdf[price_col],
                    mode="lines+markers",
                    name=f"Maturity {maturity}m",
                    legendgroup=f"maturity_{maturity}",
                    showlegend=(i == 0),
                    line=dict(width=2, color=color),
                    marker=dict(size=5, color=color),
                    hovertemplate=(
                        "Tenor: %{x}Y<br>"
                        "Price: %{y:.5f}<br>"
                        f"Maturity: {maturity}m<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Tenor (years)", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(
        height=320 * n_rows,
        width=1200,
        title=f"Price vs Tenor with Multiple Maturity Traces (Random {n_days} Days)",
        template="plotly_white",
        legend_title="Maturity",
    )
    return fig, sampled_dates


def plot_price_vs_date_tenor_maturity_lines(
    long_df,
    date_col="Date",
    tenor_col="Tenor",
    maturity_col="Maturity",
    price_col="price",
    tenors=None,
    maturities=None,
    max_traces=None,
):
    """
    Plot Price vs Date (all dates), with each trace = one unique (Tenor, Maturity) pair.
    """
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, tenor_col, maturity_col, price_col])

    if tenors is not None:
        df = df[df[tenor_col].isin(tenors)]
    if maturities is not None:
        df = df[df[maturity_col].isin(maturities)]
    if df.empty:
        raise ValueError("No data left after filtering.")

    grouped = (
        df.groupby([tenor_col, maturity_col, date_col], as_index=False)[price_col]
        .mean()
        .sort_values(date_col)
    )

    pairs = list(
        grouped[[tenor_col, maturity_col]]
        .drop_duplicates()
        .sort_values([tenor_col, maturity_col])
        .itertuples(index=False, name=None)
    )

    if max_traces is not None:
        pairs = pairs[:max_traces]

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    fig = go.Figure()

    for i, (tenor, maturity) in enumerate(pairs):
        ts = grouped[
            (grouped[tenor_col] == tenor) & (grouped[maturity_col] == maturity)
        ].sort_values(date_col)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=ts[date_col],
                y=ts[price_col],
                mode="lines",
                name=f"T{tenor}Y_M{maturity}m",
                legendgroup=f"t{tenor}_m{maturity}",
                line=dict(width=1.8, color=color),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Price: %{y:.5f}<br>"
                    f"Tenor: {tenor}Y<br>"
                    f"Maturity: {maturity}m<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Price vs Date - Tenor_Maturity Lines",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=700,
        width=1300,
        legend_title="Tenor_Maturity",
    )
    return fig

