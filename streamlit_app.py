import traceback
from datetime import date, datetime
from time import perf_counter

import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler  # noqa: F401
from typing import Any, TypedDict

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

CANONICAL_CHANNELS = sorted([
    'BAR','CBD/THC RECREATIONAL','CONCESSIONAIRE','CONVENIENCE/ GAS','DRUG STORE','E-COMMERCE',
    'FACTORY/OFFICE','GENERAL MERCHANDISE','GOVERNMENT/NON MILITARY','HEALTH/HOSPITAL','HOTEL/ MOTEL',
    'LIQUOR/PACKAGE STORE','MASS MERCH/SUPERCENTER','MILITARY OFF PREMISE','MILITARY ON-PREMISE',
    'NON RETAIL ACCOUNT','OTHER OFF PREMISE','OTHER ON PREMISE','RECREATION/ ENTERTAINMENT','RESTAURANT',
    'SCHOOL','SMALL GROCERY STORE','SPECIALTY RETAIL','SUPERMARKET','TASTING ROOM','TRANSPORTATION',
    'UNASSIGNED','WHOLESALE CLUB',
])

MATCH_CONFIG_CATALOG = {
    "minmax_CYTrend_blocking": {
        "name": "minmax_CYTrend_blocking",
        "outlier_cols": ["CY_pre_period_VOL","LY_pre_period_VOL","YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL","YOY_pre_period_TREND"],
        "blocking_factors": ["WHOLESALER_NUMBER","RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_CYLYTrend_blocking": {
        "name": "minmax_CYLYTrend_blocking",
        "outlier_cols": ["CY_pre_period_VOL","LY_pre_period_VOL","YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL","LY_pre_period_VOL","YOY_pre_period_TREND"],
        "blocking_factors": ["WHOLESALER_NUMBER","RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
    "minmax_all_blocking": {
        "name": "minmax_all_blocking",
        "outlier_cols": ["CY_pre_period_VOL","LY_pre_period_VOL","YOY_pre_period_TREND"],
        "cols_to_standardize": ["CY_pre_period_VOL","LY_pre_period_VOL","YOY_pre_period_TREND","CY_CONTRIBUTION"],
        "blocking_factors": ["WHOLESALER_NUMBER","RETAILER_CHANNEL"],
        "scaler_type": MinMaxScaler,
    },
}


def parse_int_list(csv_text: str) -> list[int]:
    items = [x.strip() for x in (csv_text or "").split(",") if x.strip()]
    return [int(x) for x in items]


@st.cache_resource
def get_spark():
    # Lazily import SparkSession so importing this module doesn't require PySpark.
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return None

    try:
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        return None


@st.cache_resource
def get_raw_tables_cached():
    spark = get_spark()
    if spark is None:
        return None, None, None, None

    try:
        # Import the heavy utils module only when Spark is available.
        from utils.data import get_raw_data_tables
        return get_raw_data_tables(spark)
    except Exception:
        return None, None, None, None


st.set_page_config(page_title="NAZ Measure Matching", layout="wide")
st.title("NAZ Measure Matching")


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("📅 Dates")
    start_date = st.date_input("Start Date", value=date(2025, 3, 31))
    end_date = st.date_input("End Date", value=date(2025, 6, 22))
    measure_start = st.date_input("Measure Start", value=date(2025, 6, 30))
    measure_end = st.date_input("Measure End", value=date(2025, 8, 24))
    data_end_cap = st.date_input("Data End Cap", value=date(2025, 11, 30))

    st.header("🏷️ Filters")
    brand_cluster_code = st.selectbox(
        "Brand Cluster Code",
        options=["MUL", "BDL", "BUD", "STA", "BHL", "KGA"],
        index=["MUL", "BDL", "BUD", "STA", "BHL", "KGA"].index("STA"),
    )
    desired_premise = st.selectbox("Premise", options=["ON PREMISE", "OFF PREMISE"], index=0)
    desired_retailer_channel = st.multiselect(
        "Retailer Channel",
        options=CANONICAL_CHANNELS,
        default=["BAR", "RESTAURANT"],
    )

    st.header("⚙️ VPIDs + Variable Timing")
    vpid_names_text = st.text_input("VPIDs (comma-separated)", value="", placeholder="77985, 4201748, 4249563")
    vpid_offsets_text = st.text_input("Offsets (weeks, comma-separated)", value="", placeholder="0, 0, 1")
    max_controls_per_test = st.number_input("Max Controls/Test", min_value=1, value=1, step=1)

    st.header("🧪 Matching Config Selection")
    selected_cfg_names = st.multiselect(
        "Match Config(s)",
        options=sorted(MATCH_CONFIG_CATALOG.keys()),
        default=["minmax_CYTrend_blocking"],
    )

    run = st.button("Run Matching", type="primary")


# -------------------------
# Main panel outputs + orchestration
# -------------------------

TOTAL_STEPS = 5


def init_state():
    st.session_state.setdefault("RESULTS", {})
    st.session_state.setdefault("LAST_RUN_CONFIG", {})  # optional: keep last inputs shown


def safe_show(obj, *, title: str | None = None):
    if title:
        st.markdown(f"### {title}")
    try:
        st.dataframe(obj, use_container_width=True)
    except Exception:
        st.write(obj)


def parse_and_validate_inputs() -> tuple[MatchPayload | None, str | None]:
    # Dates
    if start_date is None or end_date is None:
        return None, "Start/End date required."
    if measure_start is None or measure_end is None:
        return None, "Measure start/end date required."
    if start_date > end_date:
        return None, "Start Date must be <= End Date."
    if measure_start > measure_end:
        return None, "Measure Start must be <= Measure End."

    # Filters
    if not brand_cluster_code or not brand_cluster_code.strip():
        return None, "Brand Cluster Code is required."
    if not desired_retailer_channel:
        return None, "Select at least one retailer channel."
    if not selected_cfg_names:
        return None, "Select at least one match config."

    # VPIDs + offsets
    try:
        vpids = parse_int_list(vpid_names_text)
        offsets = parse_int_list(vpid_offsets_text)
    except ValueError as e:
        return None, f"VPID/offset parsing error: {e}"

    if not vpids:
        return None, "Provide at least one VPID."
    if len(vpids) != len(offsets):
        return None, f"VPIDs count ({len(vpids)}) must match offsets count ({len(offsets)})."

    # Build derived objects once
    vpid_timing_df = pd.DataFrame({"VPID": vpids, "identifier": offsets})
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
    "vpids_count": int(len(vpids)),
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
    """
    Returns a results dict ready for st.session_state.RESULTS.
    """
    # Step 1: Spark
    spark = get_spark()
    if spark is None:
        raise RuntimeError("Spark / Databricks not available — cannot run matching.")

    # Step 2: raw tables
    hv, hp, hr, hc = get_raw_tables_cached()
    if hv is None:
        raise RuntimeError("Failed to load raw data tables (hv/hp/hr/hc).")

    # Step 3: matching algo (lazy import)
    from utils.matching_algo import run_matching_variable_timing

    t0 = perf_counter()

    # (Optional) probe to fail fast if Spark is unhappy
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

    elapsed = perf_counter() - t0

    return {
        "config_summary": config_summary,
        "matched_dfs": matched_dfs,
        "group_dfs": group_dfs,
        "elapsed_seconds": elapsed,
        "ran_at": datetime.now(),
        "spark_version": getattr(spark, "version", None),
    }


def render_results(results: dict):
    config_summary = results.get("config_summary")
    matched_dfs = results.get("matched_dfs", {}) or {}
    group_dfs = results.get("group_dfs", {}) or {}

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

            offsets = sorted({k for (c, k) in keys if c == cfg_name})
            for k in offsets:
                st.markdown(f"##### Offset weeks = {k}")

                df_pairs = matched_dfs.get((cfg_name, k))
                df_groups = group_dfs.get((cfg_name, k))

                c1, c2 = st.columns(2)

                with c1:
                    st.caption("Matched pairs")
                    if df_pairs is None or getattr(df_pairs, "empty", False):
                        st.write("No matches.")
                    else:
                        safe_show(df_pairs)

                with c2:
                    st.caption("Group membership")
                    if df_groups is None or getattr(df_groups, "empty", False):
                        st.write("No groups.")
                    else:
                        safe_show(df_groups)


# ---------- UI flow ----------
init_state()

# Always show last results (if present) without rerun
results = st.session_state.get("RESULTS", {})
if results.get("config_summary") is not None:
    st.divider()
    st.subheader("Last results (session)")
    render_results(results)

# Run
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

    st.session_state["LAST_RUN_CONFIG"] = payload  # optional
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

