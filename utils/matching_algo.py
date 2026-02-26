
import sys
from pathlib import Path

repo_root = Path("/workspaces/NAZ_Measure")  # adjust to your Codespace repo
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# def override_sys_breakpoint(frame=None):
#     from IPython.core.debugger import set_trace
#     set_trace(frame=frame)
# sys.breakpointhook = override_sys_breakpoint
# %reload_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from typing import Optional, Dict, Iterable, Optional, Sequence, Tuple, List
# from naz_measure.utils.data import get_raw_data_tables, get_post_period_data_df
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    _PYSPARK_AVAILABLE = True
except Exception:
    SparkSession = None
    DataFrame = None
    F = None
    _PYSPARK_AVAILABLE = False
from datetime import date
from dateutil.relativedelta import relativedelta
try:
    from .data import *
except Exception:
    # Allow running this file directly (for quick tests) by ensuring the
    # project root is on sys.path and importing the package-style module.
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.data import *
from tqdm.auto import tqdm  
import copy

def _data_cleaning(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=cols)

def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df_final = df
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        new_vals = pd.to_numeric(df_final[col], errors="coerce")
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_final = df_final[(new_vals >= lower) & (new_vals <= upper)]
    return df_final

def _get_pool_of_potential_control_accounts(
    df: pd.DataFrame,
    blocking_factors: list[str],
    used_control_vpids: set,
    test_row: pd.Series,
    writable: bool = False,
) -> pd.DataFrame:
    missing_cols = [bf for bf in blocking_factors if bf not in df.columns]
    if missing_cols:
        raise KeyError(f"Blocking factor(s) {missing_cols} not in DataFrame columns.")

    mask = pd.Series(True, index=df.index)
    for bf in blocking_factors:
        mask &= df[bf].eq(test_row.get(bf, np.nan))

    mask &= ~df["VPID"].isin(used_control_vpids)
    pool = df.loc[mask]
    return pool.copy() if writable else pool

def get_best_match(
    df: pd.DataFrame,
    blocking_factors: list[str],
    test_row: pd.Series,
    outlier_cols: list[str],
    cols_to_standardize: list[str],
    used_control_vpids: set,
) -> pd.DataFrame:
    pool = _get_pool_of_potential_control_accounts(df, blocking_factors, used_control_vpids, test_row, writable=True)
    if pool.empty:
        return pd.DataFrame()

    missing = [c for c in cols_to_standardize if c not in pool.columns]
    if missing:
        raise KeyError(f"Pool missing required columns {missing}. Available: {list(pool.columns)}.")

    test_vec = test_row[cols_to_standardize].to_numpy(dtype=float)
    pool_mat = pool[cols_to_standardize].to_numpy(dtype=float)
    pool.loc[:, "DIST"] = np.linalg.norm(pool_mat - test_vec, axis=1)

    return pool.nsmallest(1, "DIST")

def process_best_match(
    best_match: pd.DataFrame,
    test_row: pd.Series,
    matched_results: list,
    used_control_vpids: set,
    blocking_factors: list[str],
    outlier_cols: list[str],
    cols_to_standardize: list[str],
) -> tuple[list[dict], set]:
    if best_match.empty:
        return matched_results, used_control_vpids
    
    control_vpid = best_match.iloc[0]["VPID"]
    entry = {"Test_VPID": test_row["VPID"], "Control_VPID": control_vpid}

    for bf in blocking_factors:
        entry[bf] = test_row.get(bf, np.nan)

    for col in cols_to_standardize:
        entry[f"Test_{col}"] = test_row[col]
        entry[f"Control_{col}"] = best_match.iloc[0][col]

    entry["Distance"] = best_match.iloc[0]["DIST"]
    matched_results.append(entry)
    used_control_vpids.add(control_vpid)

    return matched_results, used_control_vpids



def safe_trend(cy: float, ly: float) -> float:

    if pd.isna(ly) or ly == 0:
        return np.nan
    return (cy / ly) - 1



def run_matching_variable_timing(
    vpid_timing_df: pd.DataFrame,
    match_configs: List[dict],
    hv: "DataFrame", hp: "DataFrame", hr: "DataFrame", hc: "DataFrame",
    base_start: date, base_end: date,
    base_measure_start: date, base_measure_end: date,
    brand_cluster_code: str,
    desired_premise: str,
    desired_retailer_channel: Sequence[str],
    offset_col: str = "offset_weeks",
    max_controls_per_test: int = 1,
    candidate_control_vpids: Optional[Sequence[str]] = None,
    data_end_cap: Optional[date] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], pd.DataFrame], Dict[Tuple[str, int], pd.DataFrame]]:
    if offset_col not in vpid_timing_df.columns:
        raise KeyError(
            f"Column '{offset_col}' not found in vpid_timing_df. "
            f"Columns: {list(vpid_timing_df.columns)}"
        )

    timing_df = vpid_timing_df.copy()
    timing_df["offset_weeks"] = timing_df[offset_col].astype(int)

    offsets = sorted(timing_df["offset_weeks"].unique().tolist())

    vol_pre_period_by_offset, vol_post_period_by_offset = build_variable_timing_caches(
        hv, hp, hr, hc,
        offsets=offsets,
        base_start=base_start, base_end=base_end,
        base_measure_start=base_measure_start, base_measure_end=base_measure_end,
        brand_cluster_code=brand_cluster_code,
        desired_premise=desired_premise,
        desired_retailer_channel=desired_retailer_channel,
        vpids_for_post_period=None,
        data_end_cap=data_end_cap,
    )

    all_test_vpids = set(timing_df["VPID"].tolist())

    tests_fixed_by_offset: Dict[int, pd.DataFrame] = {}
    tests_fixed_vpids_by_offset: Dict[int, set] = {}
    for k in offsets:
        df_pre_period_k = vol_pre_period_by_offset[k]
        req_tests_k = set(timing_df.loc[timing_df["offset_weeks"] == k, "VPID"].tolist())
        eligible = pd.unique(df_pre_period_k.loc[df_pre_period_k["VPID"].isin(req_tests_k), "VPID"])
        tests_fixed = pd.DataFrame({"VPID": sorted(eligible)})
        tests_fixed.insert(0, "Group", "Test")
        tests_fixed["offset_weeks"] = k
        tests_fixed_by_offset[k] = tests_fixed[["Group", "VPID", "offset_weeks"]]
        tests_fixed_vpids_by_offset[k] = set(tests_fixed["VPID"])

    matched_dfs: Dict[Tuple[str, int], pd.DataFrame] = {}
    for cfg in tqdm(
        match_configs,
        desc="Matching configs",
        unit="config"
    ):
        cfg_name = cfg["name"]


        used_controls_global: set = set(all_test_vpids)
        all_rounds_by_offset: Dict[int, List[pd.DataFrame]] = {k: [] for k in offsets}

        dist_global, test_ids, control_ids, test_offset_by_vpid = precompute_dist_global_across_offsets(
            vol_pre_period_by_offset=vol_pre_period_by_offset,
            tests_fixed_vpids_by_offset=tests_fixed_vpids_by_offset,
            all_test_vpids=all_test_vpids,
            outlier_cols=cfg["outlier_cols"],
            cols_to_standardize=cfg["cols_to_standardize"],
            blocking_factors=cfg["blocking_factors"],
            scaler_cls=cfg.get("scaler_type", MinMaxScaler),
            candidate_control_vpids=candidate_control_vpids,
        )

        if dist_global.size == 0 or not test_ids or not control_ids:
            continue

        for m in range(1, max_controls_per_test + 1):
            ctrl_mask = np.array([cid not in used_controls_global for cid in control_ids])
            if not ctrl_mask.any():
                break

            dist_round = dist_global.copy()
            dist_round[:, ~ctrl_mask] = np.inf

            matches = _stable_marriage_from_distance(test_ids, control_ids, dist_round)
            if not matches:
                break

            out_by_offset: Dict[int, List[dict]] = {k: [] for k in offsets}
            for test_vpid, ctrl_vpid, d in matches:
                k = test_offset_by_vpid[test_vpid]
                out_by_offset[k].append({
                    "Test_VPID": test_vpid,
                    "Control_VPID": ctrl_vpid,
                    "Distance": float(d),
                })
                used_controls_global.add(ctrl_vpid)

            matched_by_offset = {
                k: (pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Test_VPID", "Control_VPID", "Distance"]))
                for k, rows in out_by_offset.items()
            }

            any_match = False
            for k in offsets:
                df_k = matched_by_offset.get(k)
                if df_k is None or df_k.empty:
                    continue

                any_match = True
                df_k = df_k.copy()
                df_k["match_order"] = m
                all_rounds_by_offset[k].append(df_k)

            if not any_match:
                break

        for k in offsets:
            if all_rounds_by_offset[k]:
                matched_dfs[(cfg_name, k)] = pd.concat(all_rounds_by_offset[k], ignore_index=True)
            else:
                matched_dfs[(cfg_name, k)] = pd.DataFrame(
                    columns=["Test_VPID", "Control_VPID", "Distance", "match_order"]
                )


    configs = [cfg["name"] for cfg in match_configs]
    canonical_tests_by_offset: Dict[int, set] = {}
    for k in offsets:
        tests_sets: List[set] = []
        for cfg_name in configs:
            df_pairs = matched_dfs.get((cfg_name, k))
            if df_pairs is None or df_pairs.empty:
                continue
            tests_sets.append(set(df_pairs["Test_VPID"].unique()))

        if not tests_sets:
            canonical_tests_by_offset[k] = set()
        else:
            canonical_tests_by_offset[k] = set.intersection(*tests_sets)


    group_dfs: Dict[Tuple[str, int], pd.DataFrame] = {}
    config_rows: List[dict] = []

    for cfg in match_configs:
        cfg_name = cfg["name"]
        memberships_all: List[pd.DataFrame] = []

        for k in offsets:
            canonical_tests = canonical_tests_by_offset.get(k, set())
            if not canonical_tests:
                membership_df_k = pd.DataFrame(columns=["Group", "VPID", "offset_weeks"])
                group_dfs[(cfg_name, k)] = membership_df_k
                memberships_all.append(membership_df_k)
                continue

            df_pairs = matched_dfs.get((cfg_name, k))
            if df_pairs is None or df_pairs.empty:
                membership_df_k = pd.DataFrame(columns=["Group", "VPID", "offset_weeks"])
                group_dfs[(cfg_name, k)] = membership_df_k
                memberships_all.append(membership_df_k)
                continue

            df_pairs = df_pairs[df_pairs["Test_VPID"].isin(canonical_tests)].copy()
            if df_pairs.empty:
                membership_df_k = pd.DataFrame(columns=["Group", "VPID", "offset_weeks"])
                group_dfs[(cfg_name, k)] = membership_df_k
                memberships_all.append(membership_df_k)
                continue

            # Option: one control per test (lowest distance). If you want up to max_controls_per_test,
            # you can instead keep rows with match_order <= max_controls_per_test.
            df_pairs = df_pairs.sort_values(["Test_VPID", "Distance"])
            df_best = df_pairs.groupby("Test_VPID", as_index=False).first()

            test_members = pd.DataFrame({"VPID": sorted(canonical_tests)})
            test_members.insert(0, "Group", "Test")
            test_members["offset_weeks"] = k

            ctrl_members = (
                df_best[["Control_VPID"]]
                .drop_duplicates()
                .rename(columns={"Control_VPID": "VPID"})
            )
            ctrl_members.insert(0, "Group", "Control")
            ctrl_members["offset_weeks"] = k

            membership_df_k = pd.concat([test_members, ctrl_members], ignore_index=True)\
                                .drop_duplicates(["Group", "VPID"])

            group_dfs[(cfg_name, k)] = membership_df_k
            memberships_all.append(membership_df_k)

        if memberships_all:
            membership_df_all = pd.concat(memberships_all, ignore_index=True)
            row = aggregate_config_metrics_variable_timing(
                cfg_name=cfg_name,
                membership_df=membership_df_all,
                vol_pre_period_by_offset=vol_pre_period_by_offset,
                vol_post_period_by_offset=vol_post_period_by_offset,
            )
            config_rows.append(row)

    config_summary = pd.DataFrame(config_rows)

    if not config_summary.empty and "config" in config_summary.columns:
        config_summary = config_summary.sort_values("config")

    config_summary = config_summary.reset_index(drop=True)

    return config_summary, matched_dfs, group_dfs


def _build_distance_matrix_blocked(
    test_df: pd.DataFrame,
    control_df: pd.DataFrame,
    cols_to_standardize: list[str],
    blocking_factors: list[str],
) -> tuple[np.ndarray, list, list]:
    if test_df.empty or control_df.empty:
        return np.zeros((0, 0), dtype=float), [], []
    test_ids = sorted(pd.unique(test_df["VPID"]))
    control_ids = sorted(pd.unique(control_df["VPID"]))

    test_unique = test_df.drop_duplicates(subset=["VPID"]).set_index("VPID")
    control_unique = control_df.drop_duplicates(subset=["VPID"]).set_index("VPID")

    X_test = test_unique.loc[test_ids, cols_to_standardize].to_numpy(dtype=float)
    X_ctrl = control_unique.loc[control_ids, cols_to_standardize].to_numpy(dtype=float)

    n_test = len(test_ids)
    n_ctrl = len(control_ids)
    dist = np.full((n_test, n_ctrl), np.inf, dtype=np.float32)
    if not blocking_factors:
        diff = X_test[:, None, :] - X_ctrl[None, :, :]  # (T, C, F)
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        return dist, test_ids, control_ids

    test_keys = (
        test_unique.loc[test_ids, blocking_factors]
        .astype("object")
        .agg(tuple, axis=1)
        .to_numpy()
    )
    ctrl_keys = (
        control_unique.loc[control_ids, blocking_factors]
        .astype("object")
        .agg(tuple, axis=1)
        .to_numpy()
    )

    from collections import defaultdict
    test_idx_by_key: dict = defaultdict(list)
    ctrl_idx_by_key: dict = defaultdict(list)

    for i, k in enumerate(test_keys):
        test_idx_by_key[k].append(i)
    for j, k in enumerate(ctrl_keys):
        ctrl_idx_by_key[k].append(j)

    common_keys = set(test_idx_by_key.keys()) & set(ctrl_idx_by_key.keys())
    for k in common_keys:
        ti = np.asarray(test_idx_by_key[k], dtype=int)
        cj = np.asarray(ctrl_idx_by_key[k], dtype=int)
        Xt_block = X_test[ti] 
        Xc_block = X_ctrl[cj]  
        diff = Xt_block[:, None, :] - Xc_block[None, :, :] 
        block_dist = np.linalg.norm(diff, axis=2).astype(np.float32)

        dist[np.ix_(ti, cj)] = block_dist

    return dist, test_ids, control_ids

def precompute_dist_global_across_offsets(
    vol_pre_period_by_offset: Dict[int, pd.DataFrame],
    tests_fixed_vpids_by_offset: Dict[int, set],
    all_test_vpids: set,
    outlier_cols: list[str],
    cols_to_standardize: list[str],
    blocking_factors: list[str],
    scaler_cls=MinMaxScaler,
    control_volume_col: str = "CY_pre_period_VOL",
    control_volume_q_bounds: Tuple[float, float] = (0.05, 0.95),
    candidate_control_vpids: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list, list, Dict[str, int]]:

    control_ids = build_global_control_universe(
        vol_pre_period_by_offset=vol_pre_period_by_offset,
        all_test_vpids=all_test_vpids,
        control_volume_col=control_volume_col,
        control_volume_q_bounds=control_volume_q_bounds,
        candidate_control_vpids=candidate_control_vpids,
    )

    if not control_ids:
        return np.zeros((0, 0), dtype=np.float32), [], [], {}

    prepared_tables = prepare_offset_tables(
        vol_pre_period_by_offset=vol_pre_period_by_offset,
        tests_fixed_vpids_by_offset=tests_fixed_vpids_by_offset,
        control_ids=control_ids,
        outlier_cols=outlier_cols,
        cols_to_standardize=cols_to_standardize,
        scaler_cls=scaler_cls,
    )

    if not prepared_tables:
        return np.zeros((0, 0), dtype=np.float32), [], [], {}


    dist_global, test_ids, control_ids, test_offset_by_vpid = compute_global_distance_matrix(
            prepared_offset_tables=prepared_tables,
            tests_fixed_vpids_by_offset=tests_fixed_vpids_by_offset,
            blocking_factors=blocking_factors,
            cols_to_standardize=cols_to_standardize,
        )

    return dist_global, test_ids, control_ids, test_offset_by_vpid





def _stable_marriage_from_distance(
    test_ids: list,
    control_ids: list,
    dist: np.ndarray,
) -> list[tuple[str, str, float]]:

    n_test, n_ctrl = dist.shape
    if n_test == 0 or n_ctrl == 0:
        return []


    test_prefs: list[list[int]] = []
    for i in range(n_test):
        row = dist[i, :]
        valid = [(j, row[j]) for j in range(n_ctrl) if np.isfinite(row[j])]
        valid.sort(key=lambda x: x[1]) 
        test_prefs.append([j for j, _ in valid])


    control_prefs: list[list[int]] = []
    for j in range(n_ctrl):
        col = dist[:, j]
        valid = [(i, col[i]) for i in range(n_test) if np.isfinite(col[i])]
        valid.sort(key=lambda x: x[1])
        control_prefs.append([i for i, _ in valid])


    control_rank = np.full((n_ctrl, n_test), fill_value=n_test + 1, dtype=int)
    for j in range(n_ctrl):
        for rank, i in enumerate(control_prefs[j]):
            control_rank[j, i] = rank


    free_tests = {i for i in range(n_test) if len(test_prefs[i]) > 0}
    next_proposal_index = [0] * n_test
    current_partner = [None] * n_ctrl  

    while free_tests:
        t = free_tests.pop()
        prefs_t = test_prefs[t]

        while next_proposal_index[t] < len(prefs_t):
            c = prefs_t[next_proposal_index[t]]
            next_proposal_index[t] += 1

            if current_partner[c] is None:
                current_partner[c] = t
                break


            other = current_partner[c]
            if control_rank[c, t] < control_rank[c, other]:

                current_partner[c] = t

                if next_proposal_index[other] < len(test_prefs[other]):
                    free_tests.add(other)
                break
            else:

                continue

    matches: list[tuple[str, str, float]] = []
    for c_idx, t_idx in enumerate(current_partner):
        if t_idx is None:
            continue
        d = float(dist[t_idx, c_idx])
        matches.append((test_ids[t_idx], control_ids[c_idx], d))

    return matches


def run_matching_stable(
    control_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outlier_cols: list[str],
    cols_to_standardize: list[str],
    blocking_factors: list[str],
    scaler_cls = MinMaxScaler,
    candidate_control_vpids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:

    if "VPID" not in control_df.columns or "VPID" not in test_df.columns:
        raise ValueError("Both control_df and test_df must contain a 'VPID' column.")

    control_clean = _data_cleaning(remove_outliers_iqr(control_df, outlier_cols), cols_to_standardize)
    test_clean = _data_cleaning(remove_outliers_iqr(test_df, outlier_cols), cols_to_standardize)

    test_vpids = set(test_clean["VPID"])

    control_pool = control_clean[~control_clean["VPID"].isin(test_vpids)]

    if candidate_control_vpids is not None:
        allowed = set(candidate_control_vpids)
        control_pool = control_pool[control_pool["VPID"].isin(allowed)]

    if control_pool.empty or test_clean.empty:
        return pd.DataFrame()

    scaler = scaler_cls()
    scaler.fit(control_pool[cols_to_standardize])

    control_scaled = control_pool.copy()
    test_scaled = test_clean.copy()
    control_scaled[cols_to_standardize] = scaler.transform(control_pool[cols_to_standardize])
    test_scaled[cols_to_standardize] = scaler.transform(test_clean[cols_to_standardize])

    dist, test_ids, control_ids = _build_distance_matrix_blocked(
        test_df=test_scaled,
        control_df=control_scaled,
        cols_to_standardize=cols_to_standardize,
        blocking_factors=blocking_factors,
    )
    if dist.size == 0:
        return pd.DataFrame()

    matches = _stable_marriage_from_distance(test_ids, control_ids, dist)
    if not matches:
        return pd.DataFrame()

    control_scaled_idx = control_scaled.set_index("VPID")
    test_scaled_idx = test_scaled.set_index("VPID")

    rows: list[dict] = []
    for test_vpid, ctrl_vpid, d in matches:
        t_row = test_scaled_idx.loc[test_vpid]
        c_row = control_scaled_idx.loc[ctrl_vpid]

        entry: dict = {
            "Test_VPID": test_vpid,
            "Control_VPID": ctrl_vpid,
            "Distance": d,
        }

        for bf in blocking_factors:
            entry[bf] = t_row.get(bf, np.nan)

        for col in cols_to_standardize:
            entry[f"Test_{col}"] = t_row[col]
            entry[f"Control_{col}"] = c_row[col]

        rows.append(entry)

    return pd.DataFrame(rows)

def build_global_control_universe(
    vol_pre_period_by_offset: Dict[int, pd.DataFrame],
    all_test_vpids: set,
    control_volume_col: str = "CY_pre_period_VOL",
    control_volume_q_bounds: Tuple[float, float] = (0.05, 0.95),
    candidate_control_vpids: Optional[Sequence[str]] = None,
) -> list[str]:

    union_pre = pd.concat(
        vol_pre_period_by_offset.values(),
        ignore_index=True
    )

    all_controls = set(union_pre["VPID"].unique())
    all_controls -= all_test_vpids

    if candidate_control_vpids is not None:
        all_controls &= set(candidate_control_vpids)

    if control_volume_col in union_pre.columns:
        agg = (
            union_pre.groupby("VPID", as_index=False)[control_volume_col]
            .sum()
        )

        vals = pd.to_numeric(agg[control_volume_col], errors="coerce").dropna()

        if len(vals):
            q_low, q_high = vals.quantile(control_volume_q_bounds).tolist()
            band_ids = set(
                agg.loc[
                    (agg[control_volume_col] >= q_low)
                    & (agg[control_volume_col] <= q_high),
                    "VPID"
                ]
            )
            all_controls &= band_ids

    return sorted(all_controls)

def prepare_offset_tables(
    vol_pre_period_by_offset: Dict[int, pd.DataFrame],
    tests_fixed_vpids_by_offset: Dict[int, set],
    control_ids: list[str],
    outlier_cols: list[str],
    cols_to_standardize: list[str],
    scaler_cls,
):

    prepared = {}

    for k, df in vol_pre_period_by_offset.items():

        test_ids_k = tests_fixed_vpids_by_offset.get(k, set())
        if not test_ids_k:
            continue

        df_k = df[df["VPID"].isin(test_ids_k | set(control_ids))]
        if df_k.empty:
            continue

        test_df = df_k[df_k["VPID"].isin(test_ids_k)]
        control_df = df_k[df_k["VPID"].isin(control_ids)]

        if test_df.empty or control_df.empty:
            continue

        control_df = _data_cleaning(
            remove_outliers_iqr(control_df, outlier_cols),
            cols_to_standardize
        )
        test_df = _data_cleaning(
            remove_outliers_iqr(test_df, outlier_cols),
            cols_to_standardize
        )

        if control_df.empty or test_df.empty:
            continue

        scaler = scaler_cls()
        scaler.fit(control_df[cols_to_standardize])

        control_scaled = control_df.copy()
        test_scaled = test_df.copy()

        control_scaled[cols_to_standardize] = scaler.transform(
            control_df[cols_to_standardize]
        )
        test_scaled[cols_to_standardize] = scaler.transform(
            test_df[cols_to_standardize]
        )

        prepared[k] = (test_scaled, control_scaled)

    return prepared
def compute_global_distance_matrix(
    prepared_offset_tables: Dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    tests_fixed_vpids_by_offset: Dict[int, set],
    blocking_factors: list[str],
    cols_to_standardize: list[str],
):

    test_offset_by_vpid = {
        v: k
        for k, vpids in tests_fixed_vpids_by_offset.items()
        for v in vpids
    }

    test_ids = sorted(test_offset_by_vpid.keys())

    control_ids = sorted({
        v
        for _, (_, ctrl_df) in prepared_offset_tables.items()
        for v in ctrl_df["VPID"].unique()
    })

    if not test_ids or not control_ids:
        return np.zeros((0, 0), dtype=np.float32), [], [], {}

    test_index = {v: i for i, v in enumerate(test_ids)}
    ctrl_index = {v: j for j, v in enumerate(control_ids)}

    dist_global = np.full(
        (len(test_ids), len(control_ids)),
        np.inf,
        dtype=np.float32
    )

    for k, (test_df, control_df) in prepared_offset_tables.items():

        dist_k, test_ids_k, control_ids_k = _build_distance_matrix_blocked(
            test_df,
            control_df,
            cols_to_standardize,
            blocking_factors
        )

        if dist_k.size == 0:
            continue

        ctrl_positions = np.array(
            [ctrl_index[c] for c in control_ids_k],
            dtype=int
        )

        for i_local, t_id in enumerate(test_ids_k):
            gi = test_index[t_id]

            dist_global[gi, ctrl_positions] = np.minimum(
                dist_global[gi, ctrl_positions],
                dist_k[i_local]
            )

    return dist_global, test_ids, control_ids, test_offset_by_vpid
