# %pip install tqdm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from typing import Optional, Sequence
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from tqdm.auto import tqdm
from time import perf_counter
from naz_measure.utils.data import *
from naz_measure.utils.matching_algo import (
    build_variable_timing_caches,
    precompute_dist_global_across_offsets,
    aggregate_config_metrics_variable_timing,
    _stable_marriage_from_distance,
)


CAL_DATE = "CAL_DATE"
REGION_COL = "RETAILER_SALES_REGION_CODE"
FORMAT_COL = "FORMAT"



# matching_validation.py



def safe_to_pandas(df: Any) -> pd.DataFrame:

    if isinstance(df, pd.DataFrame):
        return df
    return df.toPandas()
    # to_pandas_fn = getattr(df, "toPandas", None)
    # if callable(to_pandas_fn):
    #     return to_pandas_fn()
    # to_pandas_fn = getattr(df, "to_pandas", None)
    # if callable(to_pandas_fn):
    #     return to_pandas_fn()
    # raise TypeError("Expected a pandas DataFrame or an object with toPandas()/to_pandas().")

def extract_lifts(row: Mapping[str, Any]) -> Tuple[float, float]:
    t3 = float(row["test_post_period_trend"])
    t6 = float(row["test_pre_period_trend"])
    c3 = float(row["control_post_period_trend"])
    c6 = float(row["control_pre_period_trend"])
    config_lift = abs((t3 - t6) - (c3 - c6))
    baseline_lift = abs(t3 - t6)
    return config_lift, baseline_lift

def run_validation_iterations(
    filtered_df: Any,
    control_df: Any,
    match_configs: List[Dict[str, Any]],
    hv: Any, hp: Any, hr: Any, hc: Any,
    run_matching,
    aggregate_config_metrics,
    sample_size: int,
    n_iterations: int,
    rng_seed: int,
    debug: bool = False,
    show_progress: bool = True,
    vol_post_period_pd_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    filtered_df = safe_to_pandas(filtered_df)
    control_df = safe_to_pandas(control_df)

    rng = np.random.default_rng(rng_seed)
    control_pool = np.asarray(control_df["VPID"].unique())
    if sample_size > len(control_pool):
        raise ValueError("sample_size exceeds available control accounts.")

    results_rows: List[Dict[str, Any]] = []

    total_steps = n_iterations * max(1, len(match_configs))
    pbar = tqdm(total=total_steps, desc="Validations", unit="cfg", disable=not show_progress)

    for i in range(n_iterations):
        test_sample = rng.choice(control_pool, size=sample_size, replace=False)

        iter_res_test = filtered_df[filtered_df["VPID"].isin(test_sample)].copy()
        iter_res_ctrl = filtered_df[~filtered_df["VPID"].isin(test_sample)].copy()

        for cfg in match_configs:
            t0 = perf_counter()

            matched_df = run_matching(
                control_df=iter_res_ctrl,
                test_df=iter_res_test,
                outlier_cols=cfg["outlier_cols"],
                cols_to_standardize=cfg["cols_to_standardize"],
                blocking_factors=cfg["blocking_factors"],
                scaler_cls=cfg.get("scaler_type", MinMaxScaler),
            )
            matched_df = safe_to_pandas(matched_df)

            if debug:
                print(f"[iter {i} cfg {cfg['name']}] types: matched_df={type(matched_df)}, "
                      f"iter_res_ctrl={type(iter_res_ctrl)}, iter_res_test={type(iter_res_test)}")

            if matched_df.empty:
                dt = perf_counter() - t0
                results_rows.append({
                    "iteration": i,
                    "config": cfg["name"],
                    "config_lift": np.nan,
                    "baseline_lift": np.nan,
                    "delta_abs": np.nan,
                    "sample_size": sample_size,
                    "matched_pairs": 0,
                    "iteration_runtime_s": float(dt),
                    "test_pre_period_trend": np.nan,
                    "control_pre_period_trend": np.nan,
                    "test_post_period_trend": np.nan,
                    "control_post_period_trend": np.nan,
                })
                if show_progress:
                    pbar.set_postfix(iter=i, cfg=cfg["name"], matched=0)
                    pbar.update(1)
                continue

            test_vpids = matched_df[["Test_VPID"]].drop_duplicates()
            test_vpids.columns = ["VPID"]
            test_vpids.insert(0, "Group", "Test")

            ctrl_vpids = matched_df[["Control_VPID"]].drop_duplicates()
            ctrl_vpids.columns = ["VPID"]
            ctrl_vpids.insert(0, "Group", "Control")

            membership_df = pd.concat([test_vpids, ctrl_vpids], ignore_index=True).drop_duplicates(["Group", "VPID"])
            extra = aggregate_config_metrics(
                cfg_name=cfg["name"],
                membership_df=membership_df,
                result_pd_test=iter_res_test,
                result_pd_control=iter_res_ctrl,
                hv=hv, hp=hp, hr=hr, hc=hc,
                vol_post_period_pd_cache=vol_post_period_pd_cache,
            )

            config_lift, baseline_lift = extract_lifts(extra)
            delta_abs = abs(config_lift) - abs(baseline_lift)

            matched_pairs = int(len(ctrl_vpids))
            dt = perf_counter() - t0

            results_rows.append({
                "iteration": i,
                "config": cfg["name"],
                "config_lift": float(config_lift),
                "baseline_lift": float(baseline_lift),
                "delta_abs": float(delta_abs),
                "sample_size": sample_size,
                "matched_pairs": matched_pairs,
                "iteration_runtime_s": float(dt),
                "test_pre_period_trend": float(extra.get("test_pre_period_trend", np.nan)),
                "control_pre_period_trend": float(extra.get("control_pre_period_trend", np.nan)),
                "test_post_period_trend": float(extra.get("test_post_period_trend", np.nan)),
                "control_post_period_trend": float(extra.get("control_post_period_trend", np.nan)),
            })

            if show_progress:
                pbar.set_postfix(iter=i, cfg=cfg["name"], matched=matched_pairs)
                pbar.update(1)

    pbar.close()

    iter_df = pd.DataFrame(results_rows)
    return iter_df


def compute_averages_per_config(iter_df: pd.DataFrame,
                             config_col: str = "config",
                             config_lift_col: str = "config_lift",
                             baseline_lift_col: str = "baseline_lift",
                             delta_col: str = "delta_abs") -> pd.DataFrame:
    required = {
        config_col, config_lift_col, baseline_lift_col, delta_col,
        "iteration_runtime_s",
        "test_pre_period_trend", "control_pre_period_trend", "test_post_period_trend", "control_post_period_trend",
    }
    missing = [c for c in required if c not in iter_df.columns]
    if missing:
        raise ValueError(f"iter_df missing required columns: {missing}")

    cols = [config_col, config_lift_col, baseline_lift_col, delta_col,
            "iteration_runtime_s",
            "test_pre_period_trend", "control_pre_period_trend", "test_post_period_trend", "control_post_period_trend"]
    if "iteration" in iter_df.columns:
        cols.append("iteration")
    df = iter_df[cols].copy()

    per_config = (
        df.groupby(config_col, as_index=False)
          .agg(
              mean_config_lift=(config_lift_col, "mean"),
              mean_delta_abs=(delta_col, "mean"),
              total_runtime_s=("iteration_runtime_s", "sum"),
              mean_test_pre_period_trend=("test_pre_period_trend", "mean"),
              mean_control_pre_period_trend=("control_pre_period_trend", "mean"),
              mean_test_post_period_trend=("test_post_period_trend", "mean"),
              mean_control_post_period_trend=("control_post_period_trend", "mean"),
          )
    )
    if "iteration" in iter_df.columns:
        base_iter = iter_df[["iteration", baseline_lift_col]].drop_duplicates(subset=["iteration"])
        mean_baseline_lift = float(base_iter[baseline_lift_col].mean())
    else:
        mean_baseline_lift = float(df[baseline_lift_col].mean())

    per_config["mean_baseline_lift"] = mean_baseline_lift
    return per_config

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, m+1)
    q_sorted = pvals[order] * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(pvals, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q

def run_delta_abs_tests_less(
    iter_df: pd.DataFrame,
    config_col: str = "config",
    delta_col: str = "delta_abs",
    config_lift_col: str = "config_lift",
    include_fdr: bool = False,
    pval_col_name: str = "p_value_one_sided_less_than_0",
    mae_col_name: str = "MAE",
) -> pd.DataFrame:
    df = iter_df[[config_col, delta_col, config_lift_col]].dropna(subset=[delta_col])
    if df.empty:
        raise ValueError("No data after filtering/dropping NA in delta column.")

    rows = []
    for cfg, g in df.groupby(config_col):
        d = g[delta_col].to_numpy()
        mean_d = float(np.mean(d))
        try:
            t_res = stats.ttest_1samp(d, popmean=0.0, alternative="less")
            p_t = float(t_res.pvalue)
        except TypeError:
            t_res = stats.ttest_1samp(d, popmean=0.0)
            p_t = float(t_res.pvalue / 2.0) if mean_d < 0 else float(1.0 - t_res.pvalue / 2.0)

        mae = float(g[config_lift_col].abs().mean())
        rows.append({
            config_col: cfg,
            f"mean_{delta_col}": mean_d,
            mae_col_name: mae,
            pval_col_name: p_t,
        })

    out = pd.DataFrame(rows).sort_values(pval_col_name).reset_index(drop=True)

    if include_fdr:
        out["q_value_bh"] = _bh_fdr(out[pval_col_name].to_numpy())

    cols = [config_col, f"mean_{delta_col}", mae_col_name, pval_col_name]
    if include_fdr:
        cols.append("q_value_bh")
    return out[cols]

def _compute_id_props_from_counts(id_counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(id_counts.values())
    if total == 0:
        return {k: 0.0 for k in id_counts}
    return {k: id_counts[k] / total for k in id_counts}

def _proportional_int_allocation(total: int, props: Dict[int, float], max_avail: Dict[int, int]) -> Dict[int, int]:
    keys = list(props.keys())
    raw = {k: total * float(props[k]) for k in keys}
    alloc = {k: int(np.floor(raw[k])) for k in keys}
    remainder = total - sum(alloc.values())
    while remainder > 0:
        progressed = False
        frac = sorted(((k, raw[k] - alloc[k]) for k in keys), key=lambda x: x[1], reverse=True)
        for k, _ in frac:
            room = max(0, int(max_avail.get(k, np.inf)) - alloc[k])
            if room > 0:
                alloc[k] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break  
    for k in keys:
        alloc[k] = int(min(alloc[k], max_avail.get(k, alloc[k])))
    return alloc

def run_validation_iterations_variable_timing(
    vpid_timing_df: pd.DataFrame,
    match_configs: List[Dict[str, any]],
    hv: DataFrame, hp: DataFrame, hr: DataFrame, hc: DataFrame,
    base_start: date, base_end: date,
    base_measure_start: date, base_measure_end: date,
    sample_size_total: int,
    n_iterations: int,
    rng_seed: int,
    show_progress: bool = True,
    offset_col: str = "offset_weeks",
    target_id_counts: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    if offset_col not in vpid_timing_df.columns:
        raise KeyError(
            f"Column '{offset_col}' not found in vpid_timing_df. "
            f"Columns: {list(vpid_timing_df.columns)}"
        )

    timing_df = vpid_timing_df.rename(columns={offset_col: "offset_weeks"}).copy()
    timing_df["offset_weeks"] = timing_df["offset_weeks"].astype(int)

    offsets = sorted(timing_df["offset_weeks"].unique().tolist())
    vol_pre_period_by_offset, vol_post_period_by_offset = build_variable_timing_caches(
        hv, hp, hr, hc,
        offsets=offsets,
        base_start=base_start, base_end=base_end,
        base_measure_start=base_measure_start, base_measure_end=base_measure_end,
        vpids_for_post_period=None,
    )

    real_test_vpids: set = set(timing_df["VPID"].tolist())

    control_pools_by_offset: Dict[int, pd.DataFrame] = {
        k: df6m[df6m["VPID"].isin(set(df6m["VPID"]) - real_test_vpids)]
        for k, df6m in vol_pre_period_by_offset.items()
    }

    if target_id_counts is not None:
        id_counts = {int(k): int(target_id_counts.get(int(k), 0)) for k in offsets}
    else:
        id_counts = timing_df.groupby("offset_weeks").size().rename("count").to_dict()
    id_props = _compute_id_props_from_counts(id_counts)
    max_avail = {k: int(control_pools_by_offset[k]["VPID"].nunique()) for k in offsets}
    alloc = _proportional_int_allocation(sample_size_total, id_props, max_avail)

    target_strata_props_by_offset: Dict[int, Dict[Tuple[Any, Any], float]] = {}
    for k in offsets:
        df6m = vol_pre_period_by_offset[k]
        tests_k = timing_df.loc[timing_df["offset_weeks"] == k, "VPID"].unique()
        test_attrs_k = (
            df6m[df6m["VPID"].isin(tests_k)][["VPID", REGION_COL, FORMAT_COL]]
            .drop_duplicates()
        )
        if test_attrs_k.empty:
            target_strata_props_by_offset[k] = {}
            continue

        counts = (
            test_attrs_k
            .groupby([REGION_COL, FORMAT_COL])
            .size()
            .to_dict()
        )
        target_strata_props_by_offset[k] = _compute_id_props_from_counts(counts)

    rng = np.random.default_rng(rng_seed)
    results_rows: List[Dict[str, any]] = []

    total_steps = n_iterations * max(1, len(match_configs))
    pbar = tqdm(
        total=total_steps,
        desc="Validations (variable timing)",
        unit="cfg",
        disable=not show_progress,
    )

    for i in range(n_iterations):

        test_samples_by_offset: Dict[int, List[Any]] = {}

        for k in offsets:
            pool_df = control_pools_by_offset[k]
            if pool_df.empty:
                test_samples_by_offset[k] = []
                continue

            total_n_k = min(alloc[k], int(pool_df["VPID"].nunique()))
            if total_n_k <= 0:
                test_samples_by_offset[k] = []
                continue

            target_props = target_strata_props_by_offset.get(k, {})

            if not target_props:
                pool_vpids = np.asarray(pool_df["VPID"].unique().tolist())
                test_samples_by_offset[k] = rng.choice(
                    pool_vpids, size=total_n_k, replace=False
                ).tolist()
                continue

            max_avail_strata: Dict[Tuple[Any, Any], int] = {}
            for (reg, fmt) in target_props.keys():
                mask = (pool_df[REGION_COL] == reg) & (pool_df[FORMAT_COL] == fmt)
                max_avail_strata[(reg, fmt)] = int(pool_df.loc[mask, "VPID"].nunique())

            n_per_stratum = _proportional_int_allocation(
                total=total_n_k,
                props=target_props,
                max_avail=max_avail_strata,
            )

            chosen_vpids: List[Any] = []
            for (reg, fmt), n_s in n_per_stratum.items():
                if n_s <= 0:
                    continue
                mask = (pool_df[REGION_COL] == reg) & (pool_df[FORMAT_COL] == fmt)
                stratum_vpids = pool_df.loc[mask, "VPID"].drop_duplicates().to_numpy()
                if len(stratum_vpids) == 0:
                    continue
                size = min(n_s, len(stratum_vpids))
                picked = rng.choice(stratum_vpids, size=size, replace=False).tolist()
                chosen_vpids.extend(picked)

            test_samples_by_offset[k] = chosen_vpids

        tests_fixed_vpids_by_offset: Dict[int, set] = {
            k: set(test_samples_by_offset.get(k, [])) for k in offsets
        }
        iter_test_vpids: set = set().union(*tests_fixed_vpids_by_offset.values())

        for cfg in match_configs:
            t0 = perf_counter()

            all_test_vpids_iter = real_test_vpids | iter_test_vpids

            dist_global, test_ids, control_ids, test_offset_by_vpid = precompute_dist_global_across_offsets(
                vol_pre_period_by_offset=vol_pre_period_by_offset,
                tests_fixed_vpids_by_offset=tests_fixed_vpids_by_offset,
                all_test_vpids=all_test_vpids_iter,
                outlier_cols=cfg["outlier_cols"],
                cols_to_standardize=cfg["cols_to_standardize"],
                blocking_factors=cfg["blocking_factors"],
                scaler_cls=cfg.get("scaler_type", MinMaxScaler),
            )

            if dist_global.size == 0 or not test_ids or not control_ids:
                dt = perf_counter() - t0
                results_rows.append({
                    "iteration": i,
                    "config": cfg["name"],
                    "config_lift": np.nan,
                    "baseline_lift": np.nan,
                    "delta_abs": np.nan,
                    "sample_size": sample_size_total,
                    "matched_pairs": 0,
                    "iteration_runtime_s": float(dt),
                    "test_pre_period_trend": np.nan,
                    "control_pre_period_trend": np.nan,
                    "test_post_period_trend": np.nan,
                    "control_post_period_trend": np.nan,
                })
                pbar.update(1)
                continue

            matches = _stable_marriage_from_distance(test_ids, control_ids, dist_global)
            if not matches:
                dt = perf_counter() - t0
                results_rows.append({
                    "iteration": i,
                    "config": cfg["name"],
                    "config_lift": np.nan,
                    "baseline_lift": np.nan,
                    "delta_abs": np.nan,
                    "sample_size": sample_size_total,
                    "matched_pairs": 0,
                    "iteration_runtime_s": float(dt),
                    "test_pre_period_trend": np.nan,
                    "control_pre_period_trend": np.nan,
                    "test_post_period_trend": np.nan,
                    "control_post_period_trend": np.nan,
                })
                pbar.update(1)
                continue

            out_by_offset: Dict[int, List[Dict[str, Any]]] = {k: [] for k in offsets}
            for test_vpid, ctrl_vpid, d in matches:
                k = test_offset_by_vpid[test_vpid]
                out_by_offset[k].append({
                    "Test_VPID": test_vpid,
                    "Control_VPID": ctrl_vpid,
                    "Distance": float(d),
                })

            memberships_all: List[pd.DataFrame] = []
            for k in offsets:
                rows_k = out_by_offset.get(k, [])
                if not rows_k:
                    continue
                matched_df_k = pd.DataFrame(rows_k)

                test_members = (
                    matched_df_k[["Test_VPID"]]
                    .drop_duplicates()
                    .rename(columns={"Test_VPID": "VPID"})
                )
                test_members.insert(0, "Group", "Test")
                test_members["offset_weeks"] = k

                ctrl_members = (
                    matched_df_k[["Control_VPID"]]
                    .drop_duplicates()
                    .rename(columns={"Control_VPID": "VPID"})
                )
                ctrl_members.insert(0, "Group", "Control")
                ctrl_members["offset_weeks"] = k

                membership_df_k = pd.concat(
                    [test_members, ctrl_members],
                    ignore_index=True,
                ).drop_duplicates(["Group", "VPID"])
                memberships_all.append(membership_df_k)

            if not memberships_all:
                dt = perf_counter() - t0
                results_rows.append({
                    "iteration": i,
                    "config": cfg["name"],
                    "config_lift": np.nan,
                    "baseline_lift": np.nan,
                    "delta_abs": np.nan,
                    "sample_size": sample_size_total,
                    "matched_pairs": 0,
                    "iteration_runtime_s": float(dt),
                    "test_pre_period_trend": np.nan,
                    "control_pre_period_trend": np.nan,
                    "test_post_period_trend": np.nan,
                    "control_post_period_trend": np.nan,
                })
                pbar.update(1)
                continue

            membership_df_all = pd.concat(memberships_all, ignore_index=True)

            extra = aggregate_config_metrics_variable_timing(
                cfg_name=cfg["name"],
                membership_df=membership_df_all,
                vol_pre_period_by_offset=vol_pre_period_by_offset,
                vol_post_period_by_offset=vol_post_period_by_offset,
            )

            t3 = float(extra.get("test_post_period_trend", np.nan))
            t6 = float(extra.get("test_pre_period_trend", np.nan))
            c3 = float(extra.get("control_post_period_trend", np.nan))
            c6 = float(extra.get("control_pre_period_trend", np.nan))

            config_lift = abs((t3 - t6) - (c3 - c6))
            baseline_lift = abs(t3 - t6)
            delta_abs = abs(config_lift) - abs(baseline_lift)

            dt = perf_counter() - t0
            matched_pairs_total = len(matches)

            results_rows.append({
                "iteration": i,
                "config": cfg["name"],
                "config_lift": float(config_lift),
                "baseline_lift": float(baseline_lift),
                "delta_abs": float(delta_abs),
                "sample_size": sample_size_total,
                "matched_pairs": int(matched_pairs_total),
                "iteration_runtime_s": float(dt),
                "test_pre_period_trend": float(extra.get("test_pre_period_trend", np.nan)),
                "control_pre_period_trend": float(extra.get("control_pre_period_trend", np.nan)),
                "test_post_period_trend": float(extra.get("test_post_period_trend", np.nan)),
                "control_post_period_trend": float(extra.get("control_post_period_trend", np.nan)),
            })
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results_rows)

