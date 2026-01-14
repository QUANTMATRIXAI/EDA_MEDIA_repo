# app.py  (FAST: Excel -> Parquet cache -> DuckDB queries)
import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.io.formats.style import Styler

st.set_page_config(page_title="GA Deep Dive (Fast)", layout="wide")

META_PATTERN = r"(fb|fbig|facebook|meta|insta)"
CACHE_DIR = Path(".ga_cache")
CACHE_DIR.mkdir(exist_ok=True)
PROFESSIONAL_PALETTE = ["#1f5065", "#2d6a7c", "#4b8199", "#75a6bf", "#a4c9e1", "#cfe0f2"]

st.markdown(
    """
    <style>
      .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 16px;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- helpers ----------------
def file_md5(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def sanitize_name(s: str | None) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s or "sheet")).strip("_") or "sheet"


def safe_ident(col: str, allowed: list[str]) -> str:
    if col not in allowed:
        raise ValueError(f"Invalid column: {col}")
    return '"' + col.replace('"', '""') + '"'


def quoted_alias(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def minmax_scale_per_series(long_df: pd.DataFrame, series_col="series", value_col="value") -> pd.DataFrame:
    d = long_df.copy()
    g = d.groupby(series_col)[value_col]
    vmin = g.transform("min")
    vmax = g.transform("max")
    denom = (vmax - vmin)
    d[value_col] = (d[value_col] - vmin) / denom.replace(0, pd.NA)
    d[value_col] = d[value_col].fillna(0)
    return d


def _is_csv_file(uploaded_file) -> bool:
    return Path(uploaded_file.name).suffix.lower() == ".csv"


def get_columns_header_only(uploaded_file, sheet_name: str | None) -> list[str]:
    uploaded_file.seek(0)
    if _is_csv_file(uploaded_file):
        hdr = pd.read_csv(uploaded_file, nrows=0)
    else:
        hdr = pd.read_excel(uploaded_file, sheet_name=sheet_name, nrows=0, engine="openpyxl")
    uploaded_file.seek(0)
    return list(hdr.columns)


def ensure_parquet(uploaded_file, sheet_name: str | None, date_col: str) -> Path:
    uploaded_file.seek(0)
    h = file_md5(uploaded_file)
    sheet_token = sanitize_name(sheet_name or ".csv")
    out = CACHE_DIR / f"{h}__{sheet_token}__{sanitize_name(date_col)}.parquet"
    if out.exists():
        uploaded_file.seek(0)
        return out

    if _is_csv_file(uploaded_file):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    uploaded_file.seek(0)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=[date_col])
    df.to_parquet(out, index=False)
    return out


def duckdb_connect_view(parquet_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_object_cache=true;")
    path_sql = parquet_path.as_posix().replace("'", "''")
    con.execute("CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('" + path_sql + "');")
    return con


def get_numeric_metrics(con: duckdb.DuckDBPyConnection, all_cols: list[str], exclude: set[str]) -> list[str]:
    desc = con.execute("DESCRIBE SELECT * FROM t;").df()
    numeric_types = re.compile(
        r"(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|REAL|DOUBLE|DECIMAL|NUMERIC)",
        re.I,
    )
    metrics = []
    for _, r in desc.iterrows():
        name = str(r["column_name"])
        ctype = str(r["column_type"])
        if name in exclude:
            continue
        if numeric_types.search(ctype):
            metrics.append(name)
    return [c for c in all_cols if c in metrics]


def _meta_where(src_sql: str, want_meta: bool) -> tuple[str, list]:
    if want_meta:
        return f"regexp_matches(lower({src_sql}), ?)", [META_PATTERN]
    return f"NOT regexp_matches(lower({src_sql}), ?)", [META_PATTERN]


def _meta_mode_where(src_sql: str, mode: str) -> tuple[str, list]:
    normalized = (mode or "").lower()
    if normalized == "meta":
        return f"regexp_matches(lower({src_sql}), ?)", [META_PATTERN]
    if normalized == "other":
        return f"NOT regexp_matches(lower({src_sql}), ?)", [META_PATTERN]
    return "1=1", []


def fetch_agg_for_plot(
    con: duckdb.DuckDBPyConnection,
    allowed_cols: list[str],
    source_col: str,
    date_col: str,
    metrics: list[str],
    region_col: str | None,
    regions_selected: list[str],
    agg_fn: str,
    start_d: pd.Timestamp,
    end_d: pd.Timestamp,
    want_meta: bool,
) -> pd.DataFrame:
    """
    Plot data: group by day (+ region if chosen). Metrics are aggregated and returned wide.
    """
    if not metrics:
        return pd.DataFrame()

    src = safe_ident(source_col, allowed_cols)
    dt = safe_ident(date_col, allowed_cols)
    rg = safe_ident(region_col, allowed_cols) if region_col else None
    agg_sql = "SUM" if agg_fn == "sum" else "AVG"

    metric_exprs = []
    for m in metrics:
        m_id = safe_ident(m, allowed_cols)
        metric_exprs.append(f'COALESCE({agg_sql}(TRY_CAST({m_id} AS DOUBLE)), 0) AS {quoted_alias(m)}')

    select_cols = [f"CAST({dt} AS DATE) AS day"]
    group_cols = ["day"]
    where_clauses = [f"CAST({dt} AS DATE) BETWEEN ? AND ?"]
    params: list = [start_d.date(), end_d.date()]

    mw, mw_params = _meta_where(src, want_meta)
    where_clauses.append(mw)
    params.extend(mw_params)

    if region_col:
        select_cols.append(f"CAST({rg} AS VARCHAR) AS region")
        group_cols.append("region")
        if regions_selected:
            placeholders = ",".join(["?"] * len(regions_selected))
            where_clauses.append(f"CAST({rg} AS VARCHAR) IN ({placeholders})")
            params.extend(regions_selected)

    sql = f"""
        SELECT
          {", ".join(select_cols)},
          {", ".join(metric_exprs)}
        FROM t
        WHERE {" AND ".join(where_clauses)}
        GROUP BY {", ".join(group_cols)}
        ORDER BY {", ".join(group_cols)}
    """
    return con.execute(sql, params).df()


def fetch_agg_day_only_for_corr(
    con: duckdb.DuckDBPyConnection,
    allowed_cols: list[str],
    source_col: str,
    date_col: str,
    metrics: list[str],
    region_col: str | None,
    regions_selected: list[str],
    agg_fn: str,
    start_d: pd.Timestamp,
    end_d: pd.Timestamp,
    want_meta: bool,
) -> pd.DataFrame:
    """
    Correlation data: group by day ONLY (even if region is selected), but keep region filter if selected.
    This avoids correlation explosion across many region lines.
    """
    if not metrics:
        return pd.DataFrame()

    src = safe_ident(source_col, allowed_cols)
    dt = safe_ident(date_col, allowed_cols)
    rg = safe_ident(region_col, allowed_cols) if region_col else None
    agg_sql = "SUM" if agg_fn == "sum" else "AVG"

    metric_exprs = []
    for m in metrics:
        m_id = safe_ident(m, allowed_cols)
        metric_exprs.append(f'COALESCE({agg_sql}(TRY_CAST({m_id} AS DOUBLE)), 0) AS {quoted_alias(m)}')

    select_cols = [f"CAST({dt} AS DATE) AS day"]
    group_cols = ["day"]
    if region_col:
        select_cols.append(f"CAST({rg} AS VARCHAR) AS region")
        group_cols.append("region")

    where_clauses = [f"CAST({dt} AS DATE) BETWEEN ? AND ?"]
    params: list = [start_d.date(), end_d.date()]

    mw, mw_params = _meta_where(src, want_meta)
    where_clauses.append(mw)
    params.extend(mw_params)

    if region_col and regions_selected:
        placeholders = ",".join(["?"] * len(regions_selected))
        where_clauses.append(f"CAST({rg} AS VARCHAR) IN ({placeholders})")
    params.extend(regions_selected)

    group_by_expr = ", ".join(group_cols)
    sql = f"""
        SELECT
          {", ".join(select_cols)},
          {", ".join(metric_exprs)}
        FROM t
        WHERE {" AND ".join(where_clauses)}
        GROUP BY {group_by_expr}
        ORDER BY {group_by_expr}
    """
    return con.execute(sql, params).df()


def fetch_weekly_aggregates(
    con: duckdb.DuckDBPyConnection,
    allowed_cols: list[str],
    source_col: str,
    date_col: str,
    metrics: list[str],
    region_col: str | None,
    regions_selected: list[str],
    agg_fn: str,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    source_mode: str,
    exp_start: pd.Timestamp,
) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()

    src = safe_ident(source_col, allowed_cols)
    dt = safe_ident(date_col, allowed_cols)
    rg = safe_ident(region_col, allowed_cols) if region_col else None
    agg_sql = "SUM" if agg_fn == "sum" else "AVG"

    metric_exprs = []
    for m in metrics:
        m_id = safe_ident(m, allowed_cols)
        metric_exprs.append(f'COALESCE({agg_sql}(TRY_CAST({m_id} AS DOUBLE)), 0) AS {quoted_alias(m)}')

    select_cols = [
        "CAST(FLOOR(DATEDIFF('day', ?, CAST({dt} AS DATE)) / 7) AS INTEGER) AS week_index".format(
            dt=dt
        )
    ]
    group_cols = ["week_index"]
    if region_col:
        select_cols.append(f"CAST({rg} AS VARCHAR) AS region")
        group_cols.append("region")

    where_clauses = [f"CAST({dt} AS DATE) BETWEEN ? AND ?"]
    params: list = [exp_start.date(), analysis_start.date(), analysis_end.date()]

    meta_clause, meta_params = _meta_mode_where(src, source_mode)
    where_clauses.append(meta_clause)
    params.extend(meta_params)

    if region_col and regions_selected:
        placeholders = ",".join(["?"] * len(regions_selected))
        where_clauses.append(f"CAST({rg} AS VARCHAR) IN ({placeholders})")
        params.extend(regions_selected)

    sql = f"""
        SELECT
          {", ".join(select_cols)},
          {", ".join(metric_exprs)}
        FROM t
        WHERE {" AND ".join(where_clauses)}
        GROUP BY {", ".join(group_cols)}
        ORDER BY {", ".join(group_cols)}
    """
    return con.execute(sql, params).df()


def build_weekly_summary(
    df_week: pd.DataFrame,
    metrics: list[str],
    region_col: str | None,
    exp_start: pd.Timestamp,
    exp_end: pd.Timestamp,
) -> pd.DataFrame:
    if df_week.empty or not metrics:
        return pd.DataFrame()

    week_df = df_week.copy()
    exp_start = pd.to_datetime(exp_start).normalize()
    exp_end = pd.to_datetime(exp_end).normalize()
    if "week_index" in week_df.columns:
        week_df["week_index"] = pd.to_numeric(week_df["week_index"], errors="coerce").fillna(0).astype(int)
        week_df["week_start"] = exp_start + pd.to_timedelta(week_df["week_index"] * 7, unit="D")
    else:
        week_df["week_start"] = pd.to_datetime(week_df["week_start"])
        week_df["week_index"] = ((week_df["week_start"] - exp_start).dt.days // 7).astype(int)

    scope_col = "Scope"
    if "region" in week_df.columns:
        week_df[scope_col] = week_df["region"].fillna("(missing)")
    else:
        week_df[scope_col] = "All"

    metric_cols = [m for m in metrics if m in week_df.columns]
    if not metric_cols:
        return pd.DataFrame()

    id_vars = ["week_start", scope_col]
    if "week_index" in week_df.columns:
        id_vars.insert(1, "week_index")
    long_df = week_df.melt(
        id_vars=id_vars,
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    )
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce").fillna(0)

    exp_week_count = max(((exp_end - exp_start).days // 7) + 1, 1)
    if "week_index" in long_df.columns:
        long_df["week_offset"] = long_df["week_index"].astype(int)
    else:
        long_df["week_offset"] = ((long_df["week_start"] - exp_start).dt.days // 7).astype(int)
    long_df["week_type"] = "Post"
    long_df.loc[long_df["week_offset"] < 0, "week_type"] = "Baseline"
    long_df.loc[
        (long_df["week_offset"] >= 0) & (long_df["week_offset"] <= exp_week_count - 1),
        "week_type",
    ] = "Experiment"

    baseline_mask = long_df["week_type"] == "Baseline"
    experiment_mask = long_df["week_type"] == "Experiment"
    post_mask = long_df["week_type"] == "Post"
    long_df.loc[baseline_mask, "week_label"] = "Baseline wk -" + long_df.loc[baseline_mask, "week_offset"].abs().astype(str)
    long_df.loc[experiment_mask, "week_label"] = "Exp wk " + (long_df.loc[experiment_mask, "week_offset"] + 1).astype(str)
    post_base = exp_week_count - 1
    long_df.loc[post_mask, "week_label"] = "Post wk " + (long_df.loc[post_mask, "week_offset"] - post_base).astype(str)

    baseline_ref = (
        long_df[baseline_mask]
        .groupby([scope_col, "Metric"])["Value"]
        .mean()
        .reset_index(name="Baseline")
    )
    long_df = long_df.merge(baseline_ref, on=[scope_col, "Metric"], how="left")

    long_df["Value vs Baseline"] = long_df["Value"].div(long_df["Baseline"])
    long_df["Value vs Baseline"] = long_df["Value vs Baseline"].replace([float("inf"), -float("inf")], pd.NA)
    long_df["Diff from Baseline"] = long_df["Value"] - long_df["Baseline"]

    long_df = long_df.sort_values([scope_col, "week_start", "Metric"])
    return long_df


def build_weekly_period_table(
    week_long: pd.DataFrame,
    focus_region: str,
    reference_regions: list[str],
) -> pd.DataFrame:
    if week_long.empty:
        return pd.DataFrame()

    scope_col = "Scope"
    working = week_long.copy()
    overall = (
        working.groupby(["Metric", "week_type", "week_label"], dropna=False)["Value"]
        .sum()
        .reset_index()
    )
    overall[scope_col] = "All"

    reference_label = "Reference avg"
    if reference_regions:
        ref_subset = working[working[scope_col].isin(reference_regions)]
        if not ref_subset.empty:
            ref_agg = (
                ref_subset.groupby(["Metric", "week_type", "week_label"], dropna=False)["Value"]
                .mean()
                .reset_index()
            )
            ref_agg[scope_col] = reference_label
            working = working[~working[scope_col].isin(reference_regions)]
            working = pd.concat([working, ref_agg], ignore_index=True)

    working = pd.concat([working, overall], ignore_index=True)

    baseline_avg = (
        working[working["week_type"] == "Baseline"]
        .groupby([scope_col, "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index(name="Baseline Avg")
    )
    post_avg = (
        working[working["week_type"] == "Post"]
        .groupby([scope_col, "Metric"], dropna=False)["Value"]
        .mean()
        .reset_index(name="Post Avg")
    )

    exp_week = (
        working[working["week_type"] == "Experiment"]
        .groupby([scope_col, "Metric", "week_label"], dropna=False)["Value"]
        .sum()
        .reset_index()
    )
    exp_pivot = exp_week.pivot_table(
        index=[scope_col, "Metric"], columns="week_label", values="Value"
    ).reset_index()

    table = baseline_avg.merge(exp_pivot, on=[scope_col, "Metric"], how="outer")
    table = table.merge(post_avg, on=[scope_col, "Metric"], how="outer")

    exp_cols = [c for c in table.columns if c.startswith("Exp wk")]
    exp_cols = sorted(
        exp_cols,
        key=lambda c: int(re.findall(r"\d+", c)[0]) if re.findall(r"\d+", c) else 0,
    )

    ordered_cols = [scope_col, "Metric"]
    if "Baseline Avg" in table.columns:
        ordered_cols.append("Baseline Avg")
    ordered_cols.extend(exp_cols)
    if "Post Avg" in table.columns:
        ordered_cols.append("Post Avg")

    table = table[ordered_cols]

    value_cols = [c for c in table.columns if c not in {scope_col, "Metric"}]
    value_rows = table.copy()
    value_rows["Mode"] = "Values"

    index_rows = table.copy()
    baseline_series = pd.to_numeric(table.get("Baseline Avg"), errors="coerce")

    def _make_index(series: pd.Series, baseline: pd.Series) -> pd.Series:
        denom = baseline.replace(0, pd.NA)
        return series.div(denom).mul(100)

    for col in value_cols:
        if col == "Baseline Avg":
            index_rows[col] = baseline_series.apply(
                lambda val: 100 if pd.notna(val) and val != 0 else pd.NA
            )
        else:
            index_rows[col] = _make_index(pd.to_numeric(table[col], errors="coerce"), baseline_series)
    index_rows["Mode"] = "Index"

    table = pd.concat([value_rows, index_rows], ignore_index=True)

    order_list = ["All"]
    if focus_region and focus_region != "All":
        order_list.append(focus_region)
    if reference_regions:
        order_list.append("Reference avg")
    remaining = sorted([s for s in table[scope_col].unique() if s not in order_list])
    order_list.extend(remaining)
    scope_rank = {scope: rank for rank, scope in enumerate(order_list)}

    table["scope_rank"] = table[scope_col].map(lambda s: scope_rank.get(s, len(scope_rank)))
    mode_rank = {"Values": 0, "Index": 1}
    table["mode_rank"] = table["Mode"].map(lambda m: mode_rank.get(m, 0))
    table = table.sort_values(["Metric", "scope_rank", "mode_rank"]).reset_index(drop=True)
    return table.drop(columns=["scope_rank", "mode_rank"])


def style_weekly_table(
    table: pd.DataFrame,
    focus_region: str,
    reference_regions: list[str],
) -> Styler:

    def highlight_scope(row: pd.Series) -> list[str]:
        scope = row["Scope"]
        style_parts = []
        if scope == focus_region:
            style_parts.append("background-color: #d9f2d9")
            style_parts.append("border: 2px solid #1b7f3b")
        elif scope in reference_regions:
            palette = ["#f8d7da", "#dbe9ff"]
            borders = ["#b02a37", "#1f4e8c"]
            idx = reference_regions.index(scope) if scope in reference_regions else 0
            style_parts.append(f"background-color: {palette[idx % len(palette)]}")
            style_parts.append(f"border: 2px dashed {borders[idx % len(borders)]}")
        elif scope == "All":
            style_parts.append("background-color: #f0f0f0")
            style_parts.append("border-top: 2px solid #8a8a8a")
        else:
            style_parts.append("background-color: #ffffff")
        style = "; ".join(style_parts)
        return [style for _ in row]

    def italicize_mode(row: pd.Series) -> list[str]:
        style = "font-style: italic" if row["Mode"] == "Index" else ""
        return [style for _ in row]

    fmt = {col: "{:.2f}" for col in table.columns if col not in {"Scope", "Mode"}}

    return (
        table.style.apply(highlight_scope, axis=1)
        .apply(italicize_mode, axis=1)
        .format(fmt)
        .set_properties(**{"text-align": "center", "font-size": "16px", "font-weight": "600"})
        .set_table_styles(
            [
                {"selector": "thead th", "props": [("text-align", "center"), ("font-size", "16px"), ("font-weight", "700")]},
                {"selector": "td", "props": [("padding", "8px 10px")]},
            ]
        )
    )


def add_weekly_ratio_metrics(week_df: pd.DataFrame) -> list[str]:
    ratio_specs = [
        ("Engaged sessions / Sessions", "Engaged sessions", "Sessions"),
        ("Items viewed / Sessions", "Items viewed", "Sessions"),
        ("Add to carts / Sessions", "Add to carts", "Sessions"),
        ("Items added to cart / Sessions", "Items added to cart", "Sessions"),
        ("Items added to cart / Add to carts", "Items added to cart", "Add to carts"),
        ("First time purchasers / Purchases", "First time purchasers", "Purchases"),
        ("Purchases / Add to carts", "Purchases", "Add to carts"),
        ("First time purchasers / Add to carts", "First time purchasers", "Add to carts"),
    ]
    added: list[str] = []
    for label, numerator, denominator in ratio_specs:
        if numerator not in week_df.columns or denominator not in week_df.columns:
            continue
        num = pd.to_numeric(week_df[numerator], errors="coerce")
        den = pd.to_numeric(week_df[denominator], errors="coerce").replace(0, pd.NA)
        week_df[label] = num.div(den)
        added.append(label)
    return added

def make_long(df_wide: pd.DataFrame, metrics: list[str], region_col: str | None, do_minmax: bool) -> pd.DataFrame:
    if df_wide.empty:
        return df_wide

    id_vars = ["day"] + (["region"] if region_col and "region" in df_wide.columns else [])
    keep = id_vars + [m for m in metrics if m in df_wide.columns]
    d = df_wide[keep].copy()

    long_df = d.melt(id_vars=id_vars, var_name="metric", value_name="value")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0)

    if "region" in long_df.columns:
        long_df["series"] = long_df["metric"].astype(str) + " | " + long_df["region"].astype(str)
    else:
        long_df["series"] = long_df["metric"].astype(str)

    return minmax_scale_per_series(long_df, "series", "value") if do_minmax else long_df


def plot_lines(long_df: pd.DataFrame, title: str, key: str, y_label: str):
    if long_df is None or long_df.empty:
        st.warning("No data after filters.")
        return
    fig = px.line(long_df, x="day", y="value", color="series", title=title)
    fig.update_yaxes(title=y_label)
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=key)


def corr_scorecard(df_wide: pd.DataFrame, metrics: list[str], target: str, exp_start: pd.Timestamp, exp_end: pd.Timestamp) -> pd.DataFrame:
    """
    Pearson correlation of each metric vs target:
      - Full period (df_wide as provided)
      - Experiment period (filtered by exp range)
    """
    if df_wide is None or df_wide.empty or target not in df_wide.columns:
        return pd.DataFrame()

    # ensure day is datetime for filtering
    d = df_wide.copy()
    d["day"] = pd.to_datetime(d["day"], errors="coerce")
    d = d.dropna(subset=["day"])

    def _corr_one(df_slice: pd.DataFrame, a: str, b: str):
        s = df_slice[[a, b]].dropna()
        if len(s) < 2:
            return None, 0
        x = pd.to_numeric(s[a], errors="coerce")
        y = pd.to_numeric(s[b], errors="coerce")
        s2 = pd.concat([x, y], axis=1).dropna()
        if len(s2) < 2:
            return None, 0
        if s2.iloc[:, 0].nunique() <= 1 or s2.iloc[:, 1].nunique() <= 1:
            return None, len(s2)
        return float(s2.iloc[:, 0].corr(s2.iloc[:, 1])), len(s2)

    # scopes (global + per-region if region column exists)
    scope_dfs: list[tuple[str, pd.DataFrame]] = [("All", d)]
    if "region" in d.columns:
        for region, region_df in d.groupby("region", dropna=False):
            if region_df.empty:
                continue
            label = "(missing)" if region is None else str(region)
            scope_dfs.append((label, region_df))

    rows: list[dict[str, object]] = []
    for scope_label, scope_df in scope_dfs:
        if scope_df.empty:
            continue
        d_exp = scope_df[(scope_df["day"] >= exp_start) & (scope_df["day"] <= exp_end)]

        for m in metrics:
            if m == target or m not in scope_df.columns:
                continue

            c_full, n_full = _corr_one(scope_df, target, m)
            c_exp, n_exp = _corr_one(d_exp, target, m)

            rows.append(
                {
                    "Scope": scope_label,
                    "Metric": m,
                    "Corr (Full)": None if c_full is None else round(c_full, 3),
                    "N (Full)": n_full,
                    "Corr (Experiment)": None if c_exp is None else round(c_exp, 3),
                    "N (Experiment)": n_exp,
                }
            )

    return pd.DataFrame(rows)


# ---------------- UI ----------------
st.title("📊 GA Deep Dive — Meta vs Other (Fast)")

uploaded = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.info("Upload your Excel or CSV file to begin.")
    st.stop()

is_csv = _is_csv_file(uploaded)
sheet = None
if not is_csv:
    uploaded.seek(0)
    xls = pd.ExcelFile(uploaded)
    uploaded.seek(0)
    sheet = st.sidebar.selectbox("Sheet", xls.sheet_names, index=0)
cols = get_columns_header_only(uploaded, sheet)

# Column selectors
default_source = "Session source" if "Session source" in cols else cols[0]
default_date = next((c for c in cols if "date" in c.lower() or "day" in c.lower()), cols[0])

st.sidebar.subheader("Columns")
source_col = st.sidebar.selectbox("Source column", cols, index=cols.index(default_source))
date_col = st.sidebar.selectbox("Date column", cols, index=cols.index(default_date))

# Region (optional)
region_candidates = ["(No region grouping)"]
for c in cols:
    cl = c.lower()
    if any(k in cl for k in ["region", "state", "city", "country", "dma", "geo"]):
        region_candidates.append(c)
region_choice = st.sidebar.selectbox("Region column", region_candidates, index=0)
region_col = None if region_choice == "(No region grouping)" else region_choice

# Parquet cache + DuckDB view
with st.sidebar:
    with st.spinner("Preparing fast cache (Excel/CSV → Parquet)…"):
        parquet_path = ensure_parquet(uploaded, sheet, date_col)

con = duckdb_connect_view(parquet_path)
allowed_cols = cols[:]

# Data min/max dates
dt_sql = safe_ident(date_col, allowed_cols)
minmax = con.execute(f"SELECT MIN(CAST({dt_sql} AS DATE)) AS min_d, MAX(CAST({dt_sql} AS DATE)) AS max_d FROM t;").df()
min_d = pd.to_datetime(minmax.loc[0, "min_d"]).date()
max_d = pd.to_datetime(minmax.loc[0, "max_d"]).date()

# Sidebar global settings
st.sidebar.subheader("Global settings")
agg_fn = st.sidebar.selectbox("Aggregation", ["sum", "mean"], index=0)

# Experiment period (sidebar)
# Experiment period (sidebar)
st.sidebar.subheader("Experiment period")
exp_range = st.sidebar.date_input(
    "Select experiment start/end",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)
exp_start = pd.to_datetime(exp_range[0])
exp_end = pd.to_datetime(exp_range[1])

# Metrics pool (numeric only)
exclude = {source_col, date_col}
if region_col:
    exclude.add(region_col)
metrics_pool = get_numeric_metrics(con, cols, exclude=exclude)
if not metrics_pool:
    st.error("No numeric metric columns found in this sheet.")
    st.stop()

# Region selection (global)
regions_selected: list[str] = []
if region_col:
    rg_sql = safe_ident(region_col, allowed_cols)
    regions_df = con.execute(
        f"""
        SELECT DISTINCT CAST({rg_sql} AS VARCHAR) AS region
        FROM t
        WHERE {rg_sql} IS NOT NULL
        ORDER BY region
        """
    ).df()
    regions = regions_df["region"].dropna().astype(str).tolist()
    st.sidebar.subheader("Regions (global filter)")
    regions_selected = st.sidebar.multiselect(
        "Select regions (affects charts + correlations)",
        options=regions,
        default=regions[:5] if len(regions) > 5 else regions,
    )

tab_main, tab_weekly = st.tabs(["Meta vs Other", "Week-by-week"])

with tab_main:
    # Graph state
    if "graph_count" not in st.session_state:
        st.session_state.graph_count = 1
    if "graph_dates" not in st.session_state:
        st.session_state.graph_dates = {}
    if "graph_metrics" not in st.session_state:
        st.session_state.graph_metrics = {}
    if "graph_minmax" not in st.session_state:
        st.session_state.graph_minmax = {}
    if "graph_corr_target" not in st.session_state:
        st.session_state.graph_corr_target = {}

    # Render graphs
    for i in range(st.session_state.graph_count):
        st.divider()
        st.subheader(f"📈 Graph {i+1}")

        # controls in ONE ROW (columns): date | y-metrics | minmax+corrTarget
        c_date, c_metrics, c_opts = st.columns([1.2, 2.8, 1.4], gap="small")

        # per-graph defaults
        if i not in st.session_state.graph_dates:
            st.session_state.graph_dates[i] = (min_d, max_d)
        if i not in st.session_state.graph_metrics:
            st.session_state.graph_metrics[i] = metrics_pool[:2] if len(metrics_pool) >= 2 else metrics_pool[:1]
        if i not in st.session_state.graph_minmax:
            st.session_state.graph_minmax[i] = True

        g_date = c_date.date_input(
            "Date range",
            value=st.session_state.graph_dates[i],
            min_value=min_d,
            max_value=max_d,
            key=f"date_{i}",
        )
        st.session_state.graph_dates[i] = g_date
        g_start = pd.to_datetime(g_date[0])
        g_end = pd.to_datetime(g_date[1])

        metrics_sel = c_metrics.multiselect(
            "Y-metrics (lines)",
            options=metrics_pool,
            default=st.session_state.graph_metrics[i],
            key=f"metrics_{i}",
        )
        st.session_state.graph_metrics[i] = metrics_sel

        do_minmax = c_opts.checkbox(
            "Min–Max (0–1)",
            value=st.session_state.graph_minmax[i],
            key=f"minmax_{i}",
            help="ON: each line scaled 0–1. OFF: raw values.",
        )
        st.session_state.graph_minmax[i] = do_minmax

        corr_target_options = metrics_sel[:] if metrics_sel else metrics_pool
        default_target = corr_target_options[0] if corr_target_options else None
        if i not in st.session_state.graph_corr_target:
            st.session_state.graph_corr_target[i] = default_target

        corr_target = c_opts.selectbox(
            "Corr target",
            options=corr_target_options if corr_target_options else ["(none)"],
            index=0,
            key=f"corr_target_{i}",
            help="Correlations shown below each chart: each metric vs this target.",
        )
        st.session_state.graph_corr_target[i] = corr_target

        # plots
        left, right = st.columns(2, gap="large")

        # META side
        df_meta_wide_plot = fetch_agg_for_plot(
            con, allowed_cols, source_col, date_col,
            metrics_sel, region_col, regions_selected,
            agg_fn, g_start, g_end, want_meta=True
        )
        meta_long = make_long(df_meta_wide_plot, metrics_sel, region_col, do_minmax)

        with left:
            st.markdown("### 🟦 Meta Sources")
            plot_lines(
                meta_long,
                title="Meta",
                key=f"meta_plot_{i}",
                y_label="Scaled (0–1)" if do_minmax else "Value",
            )

            # correlation scorecard for META (day-only)
            df_meta_corr = fetch_agg_day_only_for_corr(
                con, allowed_cols, source_col, date_col,
                metrics_sel, region_col, regions_selected,
                agg_fn, g_start, g_end, want_meta=True
            )
            corr_tbl = corr_scorecard(df_meta_corr, metrics_sel, corr_target, exp_start, exp_end)
            st.caption(f"Pearson correlation vs **{corr_target}** (Full vs Experiment)")
            if corr_tbl.empty:
                st.info("Not enough data / select at least 2 metrics to show correlations.")
            else:
                st.dataframe(corr_tbl, use_container_width=True, hide_index=True)

        # OTHER side
        df_other_wide_plot = fetch_agg_for_plot(
            con, allowed_cols, source_col, date_col,
            metrics_sel, region_col, regions_selected,
            agg_fn, g_start, g_end, want_meta=False
        )
        other_long = make_long(df_other_wide_plot, metrics_sel, region_col, do_minmax)

        with right:
            st.markdown("### 🟩 Other Sources")
            plot_lines(
                other_long,
                title="Other",
                key=f"other_plot_{i}",
                y_label="Scaled (0–1)" if do_minmax else "Value",
            )

            df_other_corr = fetch_agg_day_only_for_corr(
                con, allowed_cols, source_col, date_col,
                metrics_sel, region_col, regions_selected,
                agg_fn, g_start, g_end, want_meta=False
            )
            corr_tbl2 = corr_scorecard(df_other_corr, metrics_sel, corr_target, exp_start, exp_end)
            st.caption(f"Pearson correlation vs **{corr_target}** (Full vs Experiment)")
            if corr_tbl2.empty:
                st.info("Not enough data / select at least 2 metrics to show correlations.")
            else:
                st.dataframe(corr_tbl2, use_container_width=True, hide_index=True)

        # Add/Remove buttons ALWAYS below the LAST graph
        if i == st.session_state.graph_count - 1:
            b1, b2, _ = st.columns([1, 1, 6])
            if b1.button("+ Add Graph Below", use_container_width=True, key=f"add_{i}"):
                st.session_state.graph_count += 1
                st.rerun()
            if b2.button("− Remove Last", use_container_width=True, key=f"rem_{i}") and st.session_state.graph_count > 1:
                last = st.session_state.graph_count - 1
                st.session_state.graph_dates.pop(last, None)
                st.session_state.graph_metrics.pop(last, None)
                st.session_state.graph_minmax.pop(last, None)
                st.session_state.graph_corr_target.pop(last, None)
                st.session_state.graph_count -= 1
                st.rerun()

    with st.expander("Preview (first 50 rows)"):
        preview = con.execute("SELECT * FROM t LIMIT 50;").df()
        st.dataframe(preview, use_container_width=True)

with tab_weekly:
    tab_weekly.subheader("Week-by-week experiment view")
    tab_weekly.caption("Use the controls below to compare each week stitched to your experiment window.")
    controls_left, controls_right = tab_weekly.columns([1, 1], gap="large")
    baseline_weeks = controls_left.slider(
        "Baseline weeks before experiment",
        0,
        8,
        4,
        help="Include this many full weeks before the experiment to build the baseline index.",
    )
    post_weeks = controls_left.slider(
        "Weeks after experiment",
        0,
        4,
        1,
        help="Include these weeks after the experiment end if data is available.",
    )
    source_mode_label = controls_right.selectbox(
        "Source focus",
        ["Meta", "Other", "All"],
        index=0,
    )
    week_metrics_default = metrics_pool[:2] if len(metrics_pool) >= 2 else metrics_pool
    week_metrics = controls_right.multiselect(
        "Metrics to summarize weekly",
        options=metrics_pool,
        default=st.session_state.get("week_metrics_weekly", week_metrics_default),
        key="week_metrics_weekly",
    )

    if not week_metrics:
        tab_weekly.info("Select at least one metric to generate the weekly summary.")
    else:
        analysis_start = exp_start - pd.Timedelta(weeks=baseline_weeks)
        analysis_end = exp_end + pd.Timedelta(weeks=post_weeks)
        exp_weeks = max(((exp_end - exp_start).days // 7) + 1, 1)
        regions_label = ", ".join(regions_selected) if regions_selected else "All regions"
        context_lines = [
            f"Source focus {source_mode_label}. Aggregating {baseline_weeks} baseline, {exp_weeks} experiment, and {post_weeks} post week(s) covering {analysis_start.date()} to {analysis_end.date()}.",
            "Weekly table uses weekly sums; baseline/post are averages of those weekly sums.",
            f"Regions filter: {regions_label}.",
        ]
        context_block = "<br>".join(context_lines)
        mode_map = {"Meta": "meta", "Other": "other", "All": "all"}
        source_mode = mode_map.get(source_mode_label, "meta")
        week_df = fetch_weekly_aggregates(
            con, allowed_cols, source_col, date_col,
            week_metrics, region_col, regions_selected,
            "sum", analysis_start, analysis_end, source_mode, exp_start
        )
        ratio_metrics = add_weekly_ratio_metrics(week_df)
        week_long = build_weekly_summary(week_df, week_metrics, region_col, exp_start, exp_end)
        if week_long.empty:
            tab_weekly.warning("No weekly data is available for the selected filters.")
        else:
            scope_options = sorted(week_long["Scope"].unique())
            if not scope_options:
                tab_weekly.warning("No region/week scopes were generated.")
            else:
                focus_key = "weekly_focus_region"
                if focus_key not in st.session_state or st.session_state[focus_key] not in scope_options:
                    st.session_state[focus_key] = scope_options[0]
                focus_region = controls_right.selectbox(
                    "Focus region for indexing",
                    options=scope_options,
                    key=focus_key,
                    index=scope_options.index(st.session_state[focus_key]),
                )
                available_references = [s for s in scope_options if s != focus_region]
                ref_key = "weekly_reference_regions"
                if ref_key not in st.session_state:
                    st.session_state[ref_key] = []
                default_refs = [r for r in st.session_state[ref_key] if r in available_references]
                if not default_refs:
                    default_refs = available_references[:2]
                reference_regions = controls_right.multiselect(
                    "Reference regions",
                    options=available_references,
                    default=default_refs,
                    key=ref_key,
                )
                include_all_key = "weekly_include_all_scopes"
                include_all_scopes = controls_left.checkbox(
                    "Include all filtered regions",
                    value=st.session_state.get(include_all_key, False),
                    key=include_all_key,
                )
                with controls_left.expander("Weekly table context", expanded=True):
                    st.markdown(context_block, unsafe_allow_html=True)
                table = build_weekly_period_table(week_long, focus_region, reference_regions)
                if table.empty:
                    tab_weekly.warning("Unable to summarize the weekly periods for the chosen filters.")
                else:
                    focus_line = f"Focus={focus_region}; reference set={', '.join(reference_regions) or 'none'}."
                    with controls_right.expander("Selected focus & references", expanded=True):
                        st.markdown(focus_line)
                    reference_display = ["Reference avg"] if reference_regions else []
                    if not include_all_scopes:
                        allowed = {focus_region, *reference_display, "All"}
                        table = table[table["Scope"].isin(allowed)]
                    metric_names = [m for m in week_metrics if m in table["Metric"].unique()]
                    metric_tabs = tab_weekly.tabs(metric_names)
                    for metric, metric_tab in zip(metric_names, metric_tabs):
                        metric_table = table[table["Metric"] == metric].drop(columns=["Metric"])
                        with metric_tab:
                            styled = style_weekly_table(metric_table, focus_region, reference_display)
                            st.dataframe(styled, use_container_width=True)

                    if ratio_metrics:
                        ratio_long = build_weekly_summary(
                            week_df, ratio_metrics, region_col, exp_start, exp_end
                        )
                        ratio_table = build_weekly_period_table(
                            ratio_long, focus_region, reference_regions
                        )
                        if not ratio_table.empty:
                            tab_weekly.divider()
                            tab_weekly.subheader("Computed ratios (post-week aggregation)")
                            if not include_all_scopes:
                                allowed = {focus_region, *reference_display, "All"}
                                ratio_table = ratio_table[ratio_table["Scope"].isin(allowed)]
                            ratio_names = [m for m in ratio_metrics if m in ratio_table["Metric"].unique()]
                            ratio_tabs = tab_weekly.tabs(ratio_names)
                            for metric, ratio_tab in zip(ratio_names, ratio_tabs):
                                metric_table = ratio_table[ratio_table["Metric"] == metric].drop(columns=["Metric"])
                                with ratio_tab:
                                    styled = style_weekly_table(
                                        metric_table, focus_region, reference_display
                                    )
                                    st.dataframe(styled, use_container_width=True)
