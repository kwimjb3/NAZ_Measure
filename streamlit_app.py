import os
import traceback
from datetime import date, datetime
from time import perf_counter
from typing import Any, TypedDict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from databricks.connect import DatabricksSession
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from utils.data import build_variable_timing_caches


class MatchPayload(TypedDict):
    start_date: date
    end_date: date
    measure_start: date
    measure_end: date
    data_end_cap: date
    brand_cluster_code: str
    desired_premise: str
    desired_retailer_channel: list[str]
    max_controls_per_test: int
    selected_cfg_names: list[str]
    match_configs: list[dict[str, Any]]
    vpid_timing_df: pd.DataFrame
    vpids_count: int

from sklearn.metrics import pairwise_distances

# import helpers from matching_algo; stable marriage algorithm itself plus
# the wrapper used by the batch job so we can mimic its preprocessing exactly
from utils.matching_algo import _stable_marriage_from_distance, run_matching_stable

# --- Helper to get potential controls for a VPID ---
def get_potential_controls(vpid: int, pre_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    test_row = test_df.loc[test_df["VPID"] == vpid]
    if test_row.empty:
        return pd.DataFrame()
    
    wholesaler = test_row.iloc[0]["WHOLESALER_NUMBER"]
    channel = test_row.iloc[0]["RETAILER_CHANNEL"]
    premise = test_row.iloc[0].get("PREMISE", None)

    candidates = pre_df[
        (pre_df["WHOLESALER_NUMBER"] == wholesaler) &
        (pre_df["RETAILER_CHANNEL"] == channel)
    ]
    if premise is not None and "PREMISE" in pre_df.columns:
        candidates = candidates[candidates["PREMISE"] == premise]

    return candidates

# --- Helper to rank potential matches by distance ---
# This helper is used in the VPID inspection UI and should replicate
# the same distance calculations that the matching algorithm uses so
# that the "best" candidate shown on screen corresponds to the same
# numeric metric that the matcher relies on.  In particular the matcher
# fits its scaler on the *entire control universe* rather than on the
# single test row, which was the original source of divergence.
from utils.matching_algo import (
    _stable_marriage_from_distance,
    run_matching_stable,
    _build_distance_matrix_blocked,
    _data_cleaning,
    remove_outliers_iqr,
    prepare_offset_tables,
)


def rank_matches(
    vpid,
    pre_df_k: pd.DataFrame,
    all_test_vpids: set,
    cfg: dict,
) -> pd.DataFrame:
    cols = cfg["cols_to_standardize"]
    outlier_cols = cfg.get("outlier_cols", [])
    blocking_factors = cfg.get("blocking_factors", [])
    scaler_cls = cfg.get("scaler_type", MinMaxScaler)

    control_df = pre_df_k[~pre_df_k["VPID"].isin(all_test_vpids)].copy()
    test_df = pre_df_k[pre_df_k["VPID"] == vpid].copy()

    if control_df.empty or test_df.empty:
        return pd.DataFrame()

    control_df = _data_cleaning(remove_outliers_iqr(control_df, outlier_cols), cols)
    test_df = _data_cleaning(remove_outliers_iqr(test_df, outlier_cols), cols)

    if control_df.empty or test_df.empty:
        return pd.DataFrame()

    scaler = scaler_cls()
    scaler.fit(control_df[cols])
    control_scaled = control_df.copy()
    test_scaled = test_df.copy()
    control_scaled[cols] = scaler.transform(control_df[cols])
    test_scaled[cols] = scaler.transform(test_df[cols])

    dist, test_ids, control_ids = _build_distance_matrix_blocked(
        test_df=test_scaled,
        control_df=control_scaled,
        cols_to_standardize=cols,
        blocking_factors=blocking_factors,
    )

    if dist.size == 0 or vpid not in test_ids:
        return pd.DataFrame()

    t_idx = test_ids.index(vpid)

    # distances returned by `_build_distance_matrix_blocked` correspond to the
    # *unique* control IDs tracked in ``control_ids``.  control_scaled may
    # still contain duplicate rows for the same VPID (the pre-period tables
    # aren’t guaranteed to be deduped), which would make a direct assignment
    # like ``ranked["Distance"] = dist[t_idx]`` fail with a length mismatch.
    #
    # Create a mapping from VPID→distance and then merge it back, which keeps
    # duplicates while still allowing the DataFrame length to differ.
    drow = dist[t_idx]
    dist_map = pd.Series(drow, index=control_ids)

    ranked = control_scaled.copy()
    ranked["Distance"] = ranked["VPID"].map(dist_map)

    # any unexpected missing distances (e.g. VPID filtered out by blocking)
    # should be treated as extremely large so they fall to the bottom
    ranked["Distance"] = ranked["Distance"].fillna(np.inf)

    ranked = ranked.sort_values("Distance").reset_index(drop=True)
    ranked["Rank"] = range(1, len(ranked) + 1)
    return ranked


# helper that returns the chosen control according to a local stable-marriage
# run for a single test VPID.  Rather than recomputing distances itself, this
# helper simply delegates to ``run_matching_stable`` with the same
# configuration used by the batch job.  That function performs the outlier
# trimming, control-universe filtering and scaling exactly as the job does,
# so the returned control will match the job's choice whenever the job is run
# on the same inputs (even for a single VPID).
def stable_match_vpid(vpid: int, test_df: pd.DataFrame, control_df: pd.DataFrame, cfg: dict) -> Optional[str]:
    # build minimal dataframes for the call
    test_subset = test_df.loc[test_df["VPID"] == vpid].copy()
    if test_subset.empty:
        return None

    # copy to ensure the original isn’t mutated by the service call
    ctrl_pool = control_df.copy()

    # call the same matching routine the job uses
    result = run_matching_stable(
        control_df=ctrl_pool,
        test_df=test_subset,
        outlier_cols=cfg.get("outlier_cols", []),
        cols_to_standardize=cfg.get("cols_to_standardize", []),
        blocking_factors=cfg.get("blocking_factors", []),
        scaler_cls=cfg.get("scaler_type", MinMaxScaler),
    )
    if result.empty:
        return None
    return result.iloc[0]["Control_VPID"]

CANONICAL_CHANNELS = sorted(
    [
        "BAR",
        "CBD/THC RECREATIONAL",
        "CONCESSIONAIRE",
        "CONVENIENCE/ GAS",
        "DRUG STORE",
        "E-COMMERCE",
        "FACTORY/OFFICE",
        "GENERAL MERCHANDISE",
        "GOVERNMENT/NON MILITARY",
        "HEALTH/HOSPITAL",
        "HOTEL/ MOTEL",
        "LIQUOR/PACKAGE STORE",
        "MASS MERCH/SUPERCENTER",
        "MILITARY OFF PREMISE",
        "MILITARY ON-PREMISE",
        "NON RETAIL ACCOUNT",
        "OTHER OFF PREMISE",
        "OTHER ON PREMISE",
        "RECREATION/ ENTERTAINMENT",
        "RESTAURANT",
        "SCHOOL",
        "SMALL GROCERY STORE",
        "SPECIALTY RETAIL",
        "SUPERMARKET",
        "TASTING ROOM",
        "TRANSPORTATION",
        "UNASSIGNED",
        "WHOLESALE CLUB",
    ]
)

MATCH_CONFIG_CATALOG = {
    "minmax_CYTrend_blocking": {
        "name": "minmax_CYTrend_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "YOY_pre_period_TREND"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_CYLYTrend_blocking": {
        "name": "minmax_CYLYTrend_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_all_blocking": {
        "name": "minmax_all_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND", "CY_CONTRIBUTION"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_CYTrendContribution_blocking": {
        "name": "minmax_CYTrendContribution_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "CY_CONTRIBUTION", "YOY_pre_period_TREND"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_CYContribution_blocking": {
        "name": "minmax_CYContribution_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "CY_CONTRIBUTION"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "standard_CYTrendContribution_blocking": {
        "name": "standard_CYTrendContribution_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "YOY_pre_period_TREND", "CY_CONTRIBUTION"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": StandardScaler,
    },
    "robust_CYTrendContribution_blocking": {
        "name": "robust_CYTrendContribution_blocking",
        "outlier_cols": ["CY_pre_period_VOL", "LY_pre_period_VOL", "YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL", "YOY_pre_period_TREND", "CY_CONTRIBUTION"],
        "blocking_factors": ["WHOLESALER_NUMBER", "RETAILER_CHANNEL"],
        "scaler_type": RobustScaler,
    },
}


def parse_int_list(csv_text: str) -> list[int]:
    items = [x.strip() for x in (csv_text or "").split(",") if x.strip()]
    return [int(x) for x in items]


@st.cache_resource
def get_spark():
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        st.error(
            "Databricks environment variables not set! "
            "Please ensure DATABRICKS_HOST and DATABRICKS_TOKEN are defined."
        )
        return None

    try:
        spark = DatabricksSession.builder.host(host).token(token).getOrCreate()
        _ = spark.sql("SELECT 1 AS ok").collect()[0]["ok"]
        return spark
    except Exception as e:
        st.error(f"Databricks Connect error: {e}")
        st.code(traceback.format_exc(), language="text")
        return None


@st.cache_resource
def get_raw_tables_cached():
    spark = get_spark()
    if spark is None:
        return None, None, None, None

    try:
        from utils.data import get_raw_data_tables

        return get_raw_data_tables(spark)
    except Exception:
        return None, None, None, None


def build_true_group_trends_for_offset(
    membership_df_k: pd.DataFrame,
    pre_df_k: pd.DataFrame,
    post_df_k: pd.DataFrame,
) -> pd.DataFrame:
    """
    True group trend:
      pre_trend  = sum(CY_pre) / sum(LY_pre) - 1
      post_trend = sum(CY_post) / sum(LY_post) - 1
    """
    if membership_df_k is None or membership_df_k.empty:
        return pd.DataFrame(columns=["Group", "pre_trend", "post_trend", "n"])

    def safe_trend(cy: float, ly: float) -> float:
        return np.nan if (pd.isna(ly) or ly == 0) else (cy / ly) - 1.0

    pre_join = membership_df_k.merge(
        pre_df_k[["VPID", "CY_pre_period_VOL", "LY_pre_period_VOL"]],
        on="VPID",
        how="left",
    )
    post_join = membership_df_k.merge(
        post_df_k[["VPID", "CY_post_period_VOL", "LY_post_period_VOL"]],
        on="VPID",
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for grp in ["Test", "Control"]:
        pre_g = pre_join.loc[pre_join["Group"] == grp]
        post_g = post_join.loc[post_join["Group"] == grp]

        pre_cy = float(pd.to_numeric(pre_g["CY_pre_period_VOL"], errors="coerce").fillna(0).sum())
        pre_ly = float(pd.to_numeric(pre_g["LY_pre_period_VOL"], errors="coerce").fillna(0).sum())
        post_cy = float(pd.to_numeric(post_g["CY_post_period_VOL"], errors="coerce").fillna(0).sum())
        post_ly = float(pd.to_numeric(post_g["LY_post_period_VOL"], errors="coerce").fillna(0).sum())

        rows.append(
            {
                "Group": grp,
                "pre_trend": safe_trend(pre_cy, pre_ly),
                "post_trend": safe_trend(post_cy, post_ly),
                "n": int((membership_df_k["Group"] == grp).sum()),
            }
        )

    return pd.DataFrame(rows)


def pre_post_true_trend_chart(trends_df: pd.DataFrame, *, title: str) -> go.Figure | None:
    if trends_df is None or trends_df.empty:
        return None

    need = {"Group", "pre_trend", "post_trend", "n"}
    if not need.issubset(trends_df.columns):
        st.warning(f"Cannot plot — missing columns: {sorted(need - set(trends_df.columns))}")
        return None

    def get(grp: str):
        r = trends_df.loc[trends_df["Group"] == grp]
        if r.empty:
            return np.nan, np.nan, 0
        return float(r.iloc[0]["pre_trend"]), float(r.iloc[0]["post_trend"]), int(r.iloc[0]["n"])

    t_pre, t_post, n_t = get("Test")
    c_pre, c_post, n_c = get("Control")

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(f"Test (N={n_t})", f"Control (N={n_c})"),
    )
    x_pre = "Incoming \nPre-Activation"
    x_post = "\nPost-Activation"
    
    vals = np.array([t_pre, t_post, c_pre, c_post], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        pad = 0.02
    else:
        rng = float(np.nanmax(vals) - np.nanmin(vals))
        pad = max(0.015, 0.10 * rng)

    def bar_top(v: float) -> float:
        """
        Return the bar "top" edge in screen-y terms:
        - for positive bars, that's v
        - for negative bars, the top edge is 0 (since bars extend downward from 0)
        """
        if not np.isfinite(v):
            return np.nan
        return v if v >= 0 else 0.0

    def add_step_arrow(col: int, pre: float, post: float):
        if not (np.isfinite(pre) and np.isfinite(post)):
            return

        pre_top = bar_top(pre)
        post_top = bar_top(post)

        y_top = max(pre_top, post_top) + pad

        fig.add_shape(
            type="line",
            xref=f"x{col}",
            yref="y",
            x0=x_pre,
            y0=pre_top,
            x1=x_pre,
            y1=y_top,
            line=dict(color="#111111", width=2),
        )

        fig.add_shape(
            type="line",
            xref=f"x{col}",
            yref="y",
            x0=x_pre,
            y0=y_top,
            x1=x_post,
            y1=y_top,
            line=dict(color="#111111", width=2),
        )

        fig.add_annotation(
            x=x_post,
            y=post_top,
            xref=f"x{col}",
            yref="y",
            ax=x_post,
            ay=y_top,
            axref=f"x{col}",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=2,
            arrowcolor="#111111",
            text="",
        )

        delta_pts = (post - pre) * 100.0
        xdom = "x domain" if col == 1 else "x2 domain"
        fig.add_annotation(
            x=0.5,
            y=y_top,
            xref=xdom,
            yref="y",
            text=f"{delta_pts:+.1f}",
            showarrow=False,
            font=dict(size=14, color="#111111"),
            bgcolor="white",
            bordercolor="#111111",
            borderwidth=2,
            borderpad=6,
            yshift=14,
        )

    def add_panel(col: int, pre: float, post: float):
        fig.add_trace(
            go.Bar(
                x=[x_pre, x_post],
                y=[pre, post],
                marker_color=["#BDBDBD", "#F2C94C"],
                text=[f"{v*100:.1f}%" if np.isfinite(v) else "" for v in [pre, post]],
                textposition="outside",
                cliponaxis=False,
            ),
            row=1,
            col=col,
        )
        add_step_arrow(col, pre, post)

    add_panel(1, t_pre, t_post)
    add_panel(2, c_pre, c_post)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        margin=dict(l=10, r=10, t=70, b=10),
        showlegend=False,
    )
    fig.update_yaxes(
        title_text="YoY Trend (true group)",
        tickformat=".0%",
        zeroline=True,
        zerolinecolor="#E0E0E0",
    )
    return fig

st.set_page_config(page_title="NAZ Measure Matching", layout="wide")
st.title("NAZ Measure Matching")

st.header("📅 Dates")
col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("Start Date", value=date(2025, 3, 31))
    measure_start = st.date_input("Measure Start", value=date(2025, 6, 30))

with col2:
    end_date = st.date_input("End Date", value=date(2025, 6, 22))
    measure_end = st.date_input("Measure End", value=date(2025, 8, 24))

with col3:
    data_end_cap = st.date_input("Data End Cap", value=date(2025, 11, 30))

st.header("🏷️ Filters")
col4, col5 = st.columns(2)

with col4:
    brand_cluster_code = st.selectbox(
        "Brand Cluster Code",
        options=["MUL", "BDL", "BUD", "STA", "BHL", "KGA"],
        index=["MUL", "BDL", "BUD", "STA", "BHL", "KGA"].index("STA"),
    )

with col5:
    desired_premise = st.selectbox("Premise", options=["ON PREMISE", "OFF PREMISE"], index=0)

st.subheader("Retailer Channels")

CHANNEL_KEY_PREFIX = "channel_"
SELECT_ALL_KEY = "select_all_channels"


def channel_key(ch: str) -> str:
    return f"{CHANNEL_KEY_PREFIX}{ch}"


def get_selected_channels() -> list[str]:
    return [ch for ch in CANONICAL_CHANNELS if st.session_state.get(channel_key(ch), False)]


def compute_all_selected() -> bool:
    return all(st.session_state.get(channel_key(ch), False) for ch in CANONICAL_CHANNELS)


def set_all_channels(value: bool) -> None:
    for ch in CANONICAL_CHANNELS:
        st.session_state[channel_key(ch)] = value


def on_select_all_change() -> None:
    set_all_channels(st.session_state[SELECT_ALL_KEY])


def on_any_channel_change() -> None:
    st.session_state[SELECT_ALL_KEY] = compute_all_selected()


if SELECT_ALL_KEY not in st.session_state:
    st.session_state[SELECT_ALL_KEY] = False

for ch in CANONICAL_CHANNELS:
    st.session_state.setdefault(channel_key(ch), False)

st.checkbox("Select All Channels", key=SELECT_ALL_KEY, on_change=on_select_all_change)

with st.expander("Choose Retailer Channels", expanded=False):
    for ch in CANONICAL_CHANNELS:
        st.checkbox(ch, key=channel_key(ch), on_change=on_any_channel_change)

desired_retailer_channel = get_selected_channels()

st.header("⚙️ VPIDs + Variable Timing")

uploaded_file = st.file_uploader("Upload CSV (Column 1 = VPID, Column 2 = Offset Weeks)", type=["csv"])

vpids: list[int] = []
offsets: list[int] = []

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)

        if df_upload.shape[1] < 2:
            st.error("CSV must have at least 2 columns.")
            st.stop()

        df_upload = df_upload.iloc[:, :2]
        df_upload.columns = ["VPID", "identifier"]

        df_upload["VPID"] = pd.to_numeric(df_upload["VPID"], errors="coerce")
        df_upload["identifier"] = pd.to_numeric(df_upload["identifier"], errors="coerce")

        if df_upload.isna().any().any():
            st.error("CSV contains non-numeric values.")
            st.stop()

        vpids = df_upload["VPID"].astype(int).tolist()
        offsets = df_upload["identifier"].astype(int).tolist()

        st.success(f"Loaded {len(vpids)} VPIDs from CSV.")
        st.dataframe(df_upload, width="stretch")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    c1, c2 = st.columns(2)

    with c1:
        vpid_names_text = st.text_input("VPIDs (comma-separated)", placeholder="77985, 4201748, 4249563")
    with c2:
        vpid_offsets_text = st.text_input("Offsets (weeks, comma-separated)", placeholder="0, 0, 1")

    try:
        if vpid_names_text:
            vpids = parse_int_list(vpid_names_text)
        if vpid_offsets_text:
            offsets = parse_int_list(vpid_offsets_text)
    except ValueError as e:
        st.error(f"Parsing error: {e}")
        st.stop()

if vpids and offsets and len(vpids) != len(offsets):
    st.error("VPIDs count must match offsets count.")
    st.stop()

if vpids and offsets:
    vpid_timing_df = pd.DataFrame({"VPID": vpids, "identifier": offsets})
    st.caption(f"Total VPIDs Loaded: {len(vpids)}")
else:
    vpid_timing_df = pd.DataFrame(columns=["VPID", "identifier"])

max_controls_per_test = st.number_input("Max Controls Per Test", min_value=1, value=1, step=1)

st.header("Matching Config Selection")
selected_cfg_names = st.multiselect(
    "Match Config(s)",
    options=sorted(MATCH_CONFIG_CATALOG.keys()),
    default=["minmax_CYTrend_blocking"],
)

run = st.button("🚀 Run Matching", type="primary")


def init_state():
    st.session_state.setdefault("RESULTS", {})
    st.session_state.setdefault("LAST_RUN_CONFIG", {})


def safe_show(obj, *, title: str | None = None):
    if title:
        st.markdown(f"### {title}")
    try:
        st.dataframe(obj, width="stretch")
    except Exception:
        st.write(obj)


def parse_and_validate_inputs() -> tuple[MatchPayload | None, str | None]:
    if start_date is None or end_date is None:
        return None, "Start/End date required."
    if measure_start is None or measure_end is None:
        return None, "Measure start/end date required."
    if start_date > end_date:
        return None, "Start Date must be <= End Date."
    if measure_start > measure_end:
        return None, "Measure Start must be <= Measure End."

    if not brand_cluster_code or not brand_cluster_code.strip():
        return None, "Brand Cluster Code is required."
    if not desired_retailer_channel:
        return None, "Select at least one retailer channel."
    if not selected_cfg_names:
        return None, "Select at least one match config."

    if vpid_timing_df.empty:
        return None, "Provide at least one VPID."
    if len(vpid_timing_df["VPID"]) != len(vpid_timing_df["identifier"]):
        return None, "VPIDs count must match offsets count."

    match_configs = [MATCH_CONFIG_CATALOG[n] for n in selected_cfg_names]

    payload: MatchPayload = {
        "start_date": start_date,
        "end_date": end_date,
        "measure_start": measure_start,
        "measure_end": measure_end,
        "data_end_cap": data_end_cap,
        "brand_cluster_code": brand_cluster_code,
        "desired_premise": desired_premise,
        "desired_retailer_channel": list(desired_retailer_channel),
        "max_controls_per_test": int(max_controls_per_test),
        "selected_cfg_names": list(selected_cfg_names),
        "match_configs": match_configs,
        "vpid_timing_df": vpid_timing_df,
        "vpids_count": int(len(vpid_timing_df)),
    }
    return payload, None


def render_run_config(payload: MatchPayload) -> None:
    st.code(
        "\n".join(
            [
                "CONFIGURATION",
                f"Pre-Period: {payload['start_date']} → {payload['end_date']}",
                f"Measure:    {payload['measure_start']} → {payload['measure_end']}",
                f"Data Cap:   {payload['data_end_cap']}",
                f"Brand Cluster Code: '{payload['brand_cluster_code']}'",
                f"Premise:            '{payload['desired_premise']}'",
                f"Channels Selected:  {len(payload['desired_retailer_channel'])}",
                f"Match Config(s):    {payload['selected_cfg_names']}",
                f"VPIDs:              {payload['vpids_count']}",
            ]
        ),
        language="text",
    )


def run_matching_job(payload: MatchPayload) -> dict[str, Any]:
    spark = get_spark()
    if spark is None:
        raise RuntimeError("Spark / Databricks not available — cannot run matching.")

    hv, hp, hr, hc = get_raw_tables_cached()
    if hv is None:
        raise RuntimeError("Failed to load raw data tables (hv/hp/hr/hc).")

    from utils.matching_algo import run_matching_variable_timing

    t0 = perf_counter()
    _ = spark.sql("select 1 as ok").collect()[0]["ok"]

    config_summary, matched_dfs, group_dfs = run_matching_variable_timing(
        vpid_timing_df=payload["vpid_timing_df"],
        match_configs=payload["match_configs"],
        hv=hv,
        hp=hp,
        hr=hr,
        hc=hc,
        base_start=payload["start_date"],
        base_end=payload["end_date"],
        base_measure_start=payload["measure_start"],
        base_measure_end=payload["measure_end"],
        brand_cluster_code=payload["brand_cluster_code"],
        desired_premise=payload["desired_premise"],
        desired_retailer_channel=payload["desired_retailer_channel"],
        offset_col="identifier",
        max_controls_per_test=payload["max_controls_per_test"],
        data_end_cap=payload["data_end_cap"],
    )

    offsets_local = sorted(payload["vpid_timing_df"]["identifier"].astype(int).unique().tolist())
    vol_pre_by_k, vol_post_by_k = build_variable_timing_caches(
        hv,
        hp,
        hr,
        hc,
        offsets=offsets_local,
        base_start=payload["start_date"],
        base_end=payload["end_date"],
        base_measure_start=payload["measure_start"],
        base_measure_end=payload["measure_end"],
        brand_cluster_code=payload["brand_cluster_code"],
        desired_premise=payload["desired_premise"],
        desired_retailer_channel=payload["desired_retailer_channel"],
        vpids_for_post_period=None,
        data_end_cap=payload["data_end_cap"],
    )

    elapsed = perf_counter() - t0

    return {
        "config_summary": config_summary,
        "matched_dfs": matched_dfs,
        "group_dfs": group_dfs,
        "elapsed_seconds": elapsed,
        "ran_at": datetime.now(),
        "spark_version": getattr(spark, "version", None),
        "vol_pre_by_offset": vol_pre_by_k,
        "vol_post_by_offset": vol_post_by_k,
    }


def render_results(results: dict):
    config_summary = results.get("config_summary")
    matched_dfs = results.get("matched_dfs", {}) or {}
    group_dfs = results.get("group_dfs", {}) or {}
    vol_pre_by_offset = results.get("vol_pre_by_offset", {}) or {}
    vol_post_by_offset = results.get("vol_post_by_offset", {}) or {}

    # The matcher tracks which test VPIDs were "fixed" at each offset; we
    # can reconstruct that information from the payload the user submitted.
    tests_fixed_vpids_by_offset: dict[int, set] = {}
    last_payload = st.session_state.get("LAST_RUN_CONFIG", {})
    if last_payload:
        vpid_timing_df = last_payload.get("vpid_timing_df")
        if vpid_timing_df is not None and not vpid_timing_df.empty:
            # offsets are stored in the `identifier` column in the UI
            for k in sorted(vpid_timing_df["identifier"].astype(int).unique().tolist()):
                vpids_k = vpid_timing_df.loc[
                    vpid_timing_df["identifier"].astype(int) == k,
                    "VPID",
                ].astype(int).tolist()
                tests_fixed_vpids_by_offset[k] = set(vpids_k)

    st.subheader("Results")
    st.caption(
        " | ".join(
            x
            for x in [
                f"Elapsed: {results.get('elapsed_seconds', 0):.2f}s",
                f"Ran at: {results.get('ran_at'):%Y-%m-%d %H:%M:%S}" if results.get("ran_at") else None,
                f"Spark: {results.get('spark_version')}" if results.get("spark_version") else None,
            ]
            if x
        )
    )

    if config_summary is not None:
        safe_show(config_summary, title="📊 Config Summary")

    # gather all test VPIDs that appear anywhere in the match outputs;
    # useful for filtering control universes later on
    all_test_vpids = sorted(
        set().union(*[
            df["Test_VPID"].tolist() for df in matched_dfs.values()
            if df is not None and "Test_VPID" in df.columns
        ])
    )

    keys = list(matched_dfs.keys())
    if not keys:
        st.info("No match outputs to display yet.")
        return

    cfg_names = sorted({cfg for (cfg, _k) in keys})
    tabs = st.tabs(["📌 Summary"] + [f"⚙️ {c}" for c in cfg_names])

    with tabs[0]:
        if config_summary is not None:
            safe_show(config_summary)
        else:
            st.info("No summary available.")

    for i, cfg_name in enumerate(cfg_names, start=1):
        with tabs[i]:
            st.markdown(f"#### Matches — `{cfg_name}`")

            offsets_local = sorted({k for (c, k) in keys if c == cfg_name})
            for k in offsets_local:
                st.markdown(f"##### Offset weeks = {k}")

                df_pairs = matched_dfs.get((cfg_name, k))
                df_groups = group_dfs.get((cfg_name, k))

                pre_df_k = vol_pre_by_offset.get(k)
                post_df_k = vol_post_by_offset.get(k)
                if (
                    df_groups is not None
                    and not getattr(df_groups, "empty", False)
                    and pre_df_k is not None
                    and post_df_k is not None
                ):
                    trends_df = build_true_group_trends_for_offset(df_groups, pre_df_k, post_df_k)
                    fig = pre_post_true_trend_chart(trends_df, title=f"YoY Trend Shift — {cfg_name} (offset={k})")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)

                c1, c2 = st.columns(2)

                with c1:
                    st.caption("Matched pairs")
                    if df_pairs is None or getattr(df_pairs, "empty", False):
                        st.write("No matches.")
                    else:
                        # Show the full pairs table
                        safe_show(df_pairs)
                        
                        # ✅ Highlight the actual match for your single VPID as computed by
                        # the local stable-marriage logic (this will always return the
                        # closest control after scaling).
                        if len(vpids) == 1:
                            my_vpid = vpids[0]  # your single test VPID
                            # job result
                            matched_row = df_pairs[df_pairs["Test_VPID"] == my_vpid]
                            st.markdown("**Matched control(s) returned by job:**")
                            st.dataframe(matched_row)

                            # recompute pool and determine local best
                            try:
                                control_universe = pre_df_k.loc[
                                    ~pre_df_k["VPID"].isin(all_test_vpids)
                                ].copy()
                                cfg = MATCH_CONFIG_CATALOG[cfg_name]
                                best_local = stable_match_vpid(
                                    my_vpid, pre_df_k, control_universe, cfg
                                )
                                if best_local is not None:
                                    st.markdown(
                                        f"**Local stable-marriage best control:** {best_local}"
                                    )
                            except Exception:
                                pass  # if we fail to compute here, ignore

                with c2:
                    st.caption("Group membership")
                    if df_groups is None or getattr(df_groups, "empty", False):
                        st.write("No groups.")
                    else:
                        safe_show(df_groups)
    # --- VPID Inspection UI ---
    if matched_dfs:
        st.divider()
        st.header("🔍 Inspect Potential Matches for a VPID")

        # Collect all test VPIDs
        all_test_vpids = sorted(
            set().union(*[
                df["Test_VPID"].tolist() for df in matched_dfs.values() 
                if df is not None and "Test_VPID" in df.columns
            ])
        )

        selected_vpid = st.selectbox("Select a Test VPID", options=all_test_vpids)

        if selected_vpid is not None:
            st.caption(f"Showing potential matches for VPID: {selected_vpid}")

            # Loop through each match config
            for cfg_name in selected_cfg_names:
                # determine which offset applies to the selected VPID for this config
                offsets_for_cfg = sorted({k for (c, k) in matched_dfs if c == cfg_name})
                df_pairs = None
                chosen_offset = None
                for k in offsets_for_cfg:
                    cand = matched_dfs.get((cfg_name, k))
                    if cand is not None and not getattr(cand, "empty", False) and selected_vpid in cand["Test_VPID"].values:
                        df_pairs = cand
                        chosen_offset = k
                        break
                if df_pairs is None or getattr(df_pairs, "empty", False):
                    st.info(f"No matches found for config `{cfg_name}`")
                    continue
                if chosen_offset is not None:
                    st.caption(f"(offset = {chosen_offset})")

                st.subheader(f"Config: `{cfg_name}`")

                # Use pre-period data for ranking
                pre_df = results.get("vol_pre_by_offset", {}).get(chosen_offset)
                if pre_df is None:
                    st.warning("Pre-period data not available for ranking.")
                    continue

                # Build a control universe similar to the matcher by excluding test VPIDs.
                # The previous inspector passed the same dataframe for test and control,
                # which often returns the test itself or other test VPIDs as top candidates.
                control_universe = pre_df.loc[~pre_df["VPID"].isin(all_test_vpids)].copy()
                if control_universe is None or getattr(control_universe, "empty", False):
                    st.info("No available control universe after excluding test VPIDs.")
                    continue

                cfg = MATCH_CONFIG_CATALOG[cfg_name]
                ranked_df = rank_matches(selected_vpid, pre_df, set(all_test_vpids), cfg)

                if ranked_df.empty:
                    st.info(f"No potential matches found for VPID {selected_vpid} under config `{cfg_name}`.")
                else:
                    # Compute sets to explain why a candidate might not be chosen
                    used_control_vpids = set()
                    for _k, df in matched_dfs.items():
                        if df is None or getattr(df, "empty", False):
                            continue
                        used_control_vpids.update(df["Control_VPID"].tolist())

                    # Surface the actual control that the matching job chose (if available)
                    actual_job_match = None
                    if df_pairs is not None and not getattr(df_pairs, "empty", False):
                        mrow = df_pairs.loc[df_pairs["Test_VPID"] == selected_vpid]
                        if not mrow.empty:
                            actual_job_match = mrow.iloc[0]["Control_VPID"]
                    if actual_job_match is not None:
                        st.caption(f"Match chosen by job: {actual_job_match}")

                    # compute the local stable-marriage match via helper
                    stable_match = stable_match_vpid(
                        selected_vpid,
                        pre_df,
                        control_universe,
                        cfg,
                    )
                    if stable_match is not None:
                        st.caption(f"Local stable-marriage match: {stable_match}")

                    # Add explanatory columns to the ranked dataframe
                    ranked = ranked_df.copy()
                    if "VPID" in ranked.columns:
                        ranked["IsTest"] = ranked["VPID"].isin(all_test_vpids)
                        ranked["AssignedElsewhere"] = ranked["VPID"].isin(used_control_vpids)
                        if actual_job_match is not None:
                            ranked["ChosenByJob"] = ranked["VPID"] == actual_job_match
                        if stable_match is not None:
                            ranked["ChosenByStableMarriage"] = ranked["VPID"] == stable_match

                    # ensure the table is still sorted by distance/rank before showing it
                    if "Rank" in ranked.columns:
                        ranked = ranked.sort_values("Rank").reset_index(drop=True)

                    # surface the top candidate separately so it’s always obvious
                    if not ranked.empty:
                        top_vpid = ranked.iloc[0]["VPID"]
                        st.caption(f"Top-ranked control (min distance) → {top_vpid}")

                    # warn if job match diverges from the raw-distance minimum
                    if actual_job_match is not None and not ranked.empty:
                        first_vpid = ranked.iloc[0]["VPID"]
                        if actual_job_match != first_vpid:
                            st.warning(
                                "The control chosen by the matching job is not the top-ranked "
                                "candidate when sorting by distance alone. "
                                "This can happen because the matcher performs a global stable-"
                                "marriage process (and/or the candidate may already be assigned to "
                                "another test), so the best local neighbor isn’t always picked."
                            )

                    st.dataframe(ranked)

# ---------- UI flow ----------
init_state()

results = st.session_state.get("RESULTS", {})
if results.get("config_summary") is not None:
    st.divider()
    st.subheader("Last results (session)")
    render_results(results)

if run:
    st.divider()
    st.subheader("Run output")

    payload, err = parse_and_validate_inputs()
    if err:
        st.error(f"❌ {err}")
        st.stop()

    if payload is None:
        st.error("❌ Failed to parse inputs.")
        st.stop()

    st.session_state["LAST_RUN_CONFIG"] = payload
    render_run_config(payload)

    try:
        with st.status("Running matching…", expanded=True) as s:
            s.update(label="Step 1/5 — Creating/attaching Spark session…")
            s.update(label="Step 2/5 — Loading raw tables…")
            s.update(label="Step 3/5 — Running matching…")

            results_new = run_matching_job(payload)

            s.update(label="Step 4/5 — Saving results…")
            st.session_state["RESULTS"] = results_new

            s.update(label="Step 5/5 — Done.", state="complete")

        st.success(f"✅ Completed in {results_new.get('elapsed_seconds', 0):.2f} seconds")
        render_results(results_new)

    except Exception:
        st.error("❌ Exception raised during run.")
        st.code(traceback.format_exc(), language="text")
