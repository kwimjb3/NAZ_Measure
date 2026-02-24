
# def override_sys_breakpoint(frame=None):
#     from IPython.core.debugger import set_trace
#     set_trace(frame=frame)
# sys.breakpointhook = override_sys_breakpoint
# %reload_ext autoreload
# %autoreload 2
try:
    from pyspark.sql import Window
    from pyspark.sql import functions as F, SparkSession
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.functions import broadcast
    _PYSPARK_AVAILABLE = True
except Exception:
    Window = None
    F = None
    SparkSession = None
    from typing import Any as DataFrame
    broadcast = None
    _PYSPARK_AVAILABLE = False
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import scipy.stats as stats
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import broadcast
from typing import Optional, Sequence
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from tqdm.auto import tqdm



CAL_DATE = "CAL_DATE"
REGION_COL = "RETAILER_SALES_REGION_CODE"
FORMAT_COL = "FORMAT"


def get_hero_calendar(spark) -> DataFrame:
    hc = (
        spark.table("feature_store.edw.cal_dt")
        .select(
            F.col("cal_dt").alias("CAL_DATE"),
            F.col("prior_yr_cal_dt").alias("CAL_DATE_LY_ALIGNED"),

            # Week structure
            F.col("curr_wk_mon_dt").alias("WEEK_START_DATE"),
            F.col("curr_wk_sun_dt").alias("WEEK_END_DATE"),

            # Week identifiers
            F.col("cal_yr_wk_nbr").alias("YEAR_WEEK"),
            F.col("iso_yr_wk_nbr").alias("ISO_YEAR_WEEK"),

            # Prior year week identifiers
            F.col("prior_yr_cal_yr_wk_nbr").alias("LY_YEAR_WEEK"),
            F.col("prior_yr_iso_yr_wk_nbr").alias("LY_ISO_YEAR_WEEK"),
        )
    )

    hc.createOrReplaceTempView("HERO_CALENDAR")
    return hc

def get_hero_retailer(spark) -> DataFrame:
    hr = (
        spark.table("commercial.retailer.account")
        .select(
            F.col("vpid").cast("bigint").alias("VPID"),
            F.upper(F.col("on_off_premise_nm")).alias("PREMISE"),
            F.upper(F.col("channel_nm")).alias("RETAILER_CHANNEL"),
            F.col("sls_regn_cd").alias("SALES_REGION_CODE"),
            F.upper(F.col("parent_channel_nm")).alias("RETAILER_PARENT_CHANNEL"),
        )
    )

    hr.createOrReplaceTempView("HERO_RETAILER")
    return hr

def get_hero_product(spark) -> DataFrame:
    hp = (
        spark.table("commercial.product.product")
        .filter(F.col("brnd_flg").isin("ABI", "ADJ"))
        .select(
            F.col("pdcn_cd").alias("PDCN_CODE"),
            F.col("brnd_clstr_cd").alias("BRAND_CLUSTER_CODE"),
        )
    )

    hp.createOrReplaceTempView("HERO_PRODUCT")
    return hp

def get_hero_volume(spark) -> DataFrame:
    hv = (
        spark.table("vip.sales.vip_sls_str").alias("v")
        .join(
            spark.table("commercial.product.product").alias("p"),
            F.col("v.RSITEM") == F.col("p.PDCN_CD"),
            "inner",
        )
        .filter(F.col("p.brnd_flg").isin("ABI", "ADJ"))
        .groupBy(
            F.to_date(F.col("v.RSIDAT")).alias("CAL_DATE"),
            F.col("v.DISTID").alias("WHOLESALER_NUMBER"),
            F.coalesce(F.col("v.VPID"), F.lit(0)).cast("bigint").alias("VPID"),
            F.col("v.RSITEM").alias("PDCN_CODE"),
        )
        .agg(
            F.sum("v.dlvrd_bbl_eqv_qty")
            .cast("decimal(38,6)")
            .alias("ACTUAL_BARRELS")
        )
        .filter(F.length("WHOLESALER_NUMBER") <= 5)
    )

    hv.createOrReplaceTempView("HERO_VOLUME")
    return hv

def get_raw_data_tables(spark):
    hc = get_hero_calendar(spark)
    hp = get_hero_product(spark)
    hr = get_hero_retailer(spark)
    hv = get_hero_volume(spark)

    return hv, hp, hr, hc
# def get_raw_data_tables(spark):
#     # ----------------------------
#     # 1) FACT: VIP STR daily snapshot
#     # ----------------------------
#     vstr = spark.table("vip.sales.vip_sls_str").alias("vstr")

#     # Normalize to match your existing expectations:
#     # VPID, WHOLESALER_NUMBER, CAL_DATE, PDCN_CODE, ACTUAL_BARRELS
#     hv = (
#     vstr.select(
#         F.col("VPID").cast("string").alias("VPID"),
#         F.col("DISTID").alias("WHOLESALER_NUMBER"),
#         F.to_date(F.col("RSIDAT")).alias("CAL_DATE"),          
#         F.col("RSITEM").alias("PDCN_CODE"),
#         F.col("DLVRD_BBL_EQV_QTY").cast("double").alias("ACTUAL_BARRELS"),
#     )
#     .alias("hv"))


#     # ----------------------------
#     # 2) PRODUCT DIM: PDCN -> Brand Cluster
#     # ----------------------------
#     hp = (
#         spark.table("commercial.product.product")
#              .select(
#                  F.col("pdcn_cd").alias("PDCN_CODE"),
#                  F.col("brnd_clstr_cd").alias("BRAND_CLUSTER_CODE"),
#              )
#              .alias("hp")
#     )

#     # ----------------------------
#     # 3) CALENDAR: Day + prior year aligned
#     # ----------------------------
#     hc = (
#         spark.table("feature_store.edw.cal_dt")
#              .select(
#                  F.col("cal_dt").alias("CAL_DATE"),
#                  F.col("prior_yr_cal_dt").alias("CAL_DATE_LY_ALIGNED"),
#              )
#              .alias("hc")
#     )

#     # ----------------------------
#     # 4) RETAILER DIM: VPID -> premise + channel/macro_channel (+ region via wholesaler)
#     # ----------------------------
#     # ----------------------------
#     # 4) RETAILER DIM: VPID -> premise + vpcot (+ region)
#     # ----------------------------

#     vo = spark.table("vip.retailer.vip_outlt").alias("vo")
#     vc = spark.table("vip.retailer.vip_class").alias("vc")
#     sx = spark.table("commercial.wholesaler.region_state_xref").alias("sx")

#     # Map VPIDs to Sales Region via state code
#     hr = (
#     vo.join(
#         broadcast(vc),
#         vo.vpcot == vc.vpcotcd,
#         "left"
#     )
#     .join(
#         sx,
#         vo.vpstat == sx.st_cd,
#         "left"
#     )
#     .select(
#         F.col("vo.vpid").cast("string").alias("VPID"),
#         F.when(F.col("vo.vpprem") == "O", "ON PREMISE")
#          .when(F.col("vo.vpprem") == "F", "OFF PREMISE")
#          .otherwise(F.col("vo.vpprem"))
#          .alias("PREMISE"),
#         F.col("vc.vpcot").alias("RETAILER_CHANNEL"),
#         F.col("vo.vpchn").alias("RETAILER_PARENT_CHANNEL"),
#         F.col("vc.macro_channel").alias("FORMAT"),
#         F.col("sx.sls_regn_cd").alias("SALES_REGION_CODE")
#     )
#     .alias("hr")
#     )
#     return hv, hp, hr, hc

def shift_dates_by_weeks(base_start: date, base_end: date, weeks: int) -> Tuple[date, date]:
    return base_start + relativedelta(weeks=weeks), base_end + relativedelta(weeks=weeks)

def get_pre_period_data_df_for_range(
    hv: DataFrame,
    hp: DataFrame,
    hr: DataFrame,
    hc: DataFrame,
    start_dt: date,
    end_dt: date,
    brand_cluster_code: str,
    desired_premise: str,
    desired_retailer_channel: Sequence[str],
    data_end_cap: Optional[date] = None,
) -> DataFrame:
    cond = (F.col("CAL_DATE") >= F.lit(start_dt)) & (F.col("CAL_DATE") <= F.lit(end_dt))
    if data_end_cap is not None:
        cond = cond & (F.col("CAL_DATE") <= F.lit(data_end_cap))

    cal_filtered = (
        hc.filter(cond)
          .select("CAL_DATE", "CAL_DATE_LY_ALIGNED")
          .orderBy("CAL_DATE")
    )

    rows = cal_filtered.collect()
    cy_dates = [r["CAL_DATE"] for r in rows]
    ly_dates = [r["CAL_DATE_LY_ALIGNED"] for r in rows if r["CAL_DATE_LY_ALIGNED"] is not None]

    cy_df = (
        hv.join(hp, hv.PDCN_CODE == hp.PDCN_CODE)
          .join(hr, hv.VPID == hr.VPID)
          .filter(
              (hp.BRAND_CLUSTER_CODE == brand_cluster_code)
              & (F.col("CAL_DATE").isin(cy_dates))
              & (hr.PREMISE == desired_premise)
              & (hr.RETAILER_CHANNEL.isin(list(desired_retailer_channel)))
          )
          .groupBy(
              hv.VPID,
              hv.WHOLESALER_NUMBER,
              hr.RETAILER_CHANNEL,
              hr.SALES_REGION_CODE,
              hr.RETAILER_PARENT_CHANNEL,
          )
          .agg(F.sum(hv.ACTUAL_BARRELS).alias("CY_pre_period_VOL"))
    )

    ly_df = (
        hv.join(hp, hv.PDCN_CODE == hp.PDCN_CODE)
          .join(hr, hv.VPID == hr.VPID)
          .filter(
              (hp.BRAND_CLUSTER_CODE == brand_cluster_code)
              & (F.col("CAL_DATE").isin(ly_dates))
              & (hr.PREMISE == desired_premise)
              & (hr.RETAILER_CHANNEL.isin(list(desired_retailer_channel)))
          )
          .groupBy(hv.VPID)
          .agg(F.sum(hv.ACTUAL_BARRELS).alias("LY_pre_period_VOL"))
    )

    result = (
        cy_df.join(ly_df, cy_df.VPID == ly_df.VPID, "left")
             .select(
                 cy_df.VPID,
                 cy_df.WHOLESALER_NUMBER,
                 cy_df.RETAILER_CHANNEL,
                 hr.SALES_REGION_CODE.alias(REGION_COL),
                 hr.RETAILER_PARENT_CHANNEL.alias(FORMAT_COL),
                 F.round(cy_df.CY_pre_period_VOL, 2).alias("CY_pre_period_VOL"),
                 F.round(ly_df.LY_pre_period_VOL, 2).alias("LY_pre_period_VOL"),
             )
             .orderBy("VPID")
    )

    filtered = result.filter(
        (F.col("CY_pre_period_VOL").isNotNull())
        & (F.col("LY_pre_period_VOL").isNotNull())
        & (F.col("CY_pre_period_VOL") > 0)
        & (F.col("LY_pre_period_VOL") > 0)
    )

    return (
        filtered.select(
            "VPID",
            "WHOLESALER_NUMBER",
            "RETAILER_CHANNEL",
            REGION_COL,
            FORMAT_COL,
            F.round(F.col("CY_pre_period_VOL"), 2).alias("CY_pre_period_VOL"),
            F.round(F.col("LY_pre_period_VOL"), 2).alias("LY_pre_period_VOL"),
            F.round(F.col("CY_pre_period_VOL") / F.col("LY_pre_period_VOL") - F.lit(1.0), 4).alias("YOY_pre_period_TREND"),
            F.round(
                F.col("CY_pre_period_VOL")
                / (F.col("CY_pre_period_VOL") + F.col("LY_pre_period_VOL")),
                4,
            ).alias("CY_CONTRIBUTION"),
        )
        .orderBy("VPID")
    )



def get_post_period_data_df_for_range(
    hv: DataFrame,
    hp: DataFrame,
    hr: DataFrame,
    hc: DataFrame,
    measure_start_dt: date,
    measure_end_dt: date,
    brand_cluster_code: str,
    desired_premise: str,
    desired_retailer_channel: Sequence[str],
    vpids: Optional[Sequence[str]] = None,
    data_end_cap: Optional[date] = None, 
) -> DataFrame:
    cond = (F.col("CAL_DATE") >= F.lit(measure_start_dt)) & (F.col("CAL_DATE") <= F.lit(measure_end_dt))
    if data_end_cap is not None:
        cond = cond & (F.col("CAL_DATE") <= F.lit(data_end_cap))

    cal_filtered = (
        hc.filter(cond)
          .select("CAL_DATE", "CAL_DATE_LY_ALIGNED")
          .orderBy("CAL_DATE")
    )

    cal_pd = cal_filtered.toPandas()
    cy_dates = cal_pd["CAL_DATE"].tolist()
    ly_dates = cal_pd["CAL_DATE_LY_ALIGNED"].dropna().tolist()

    base = (
        hv.join(hp, hv.PDCN_CODE == hp.PDCN_CODE)
          .join(hr, hv.VPID == hr.VPID)
          .select(
              hv.VPID.alias("VPID"),
              hv.CAL_DATE.alias("CAL_DATE"),
              hv.ACTUAL_BARRELS.alias("ACTUAL_BARRELS"),
              hp.BRAND_CLUSTER_CODE.alias("BRAND_CLUSTER_CODE"),
              hr.PREMISE.alias("PREMISE"),
              hr.RETAILER_CHANNEL.alias("RETAILER_CHANNEL"),
          )
          .filter(
              (F.col("BRAND_CLUSTER_CODE") == brand_cluster_code)
              & (F.col("PREMISE") == desired_premise)
              & (F.col("RETAILER_CHANNEL").isin(list(desired_retailer_channel)))
          )
    )
    if vpids:
        base = base.filter(F.col("VPID").isin(list(vpids)))

    cy_df = (
        base.filter(F.col("CAL_DATE").isin(cy_dates))
            .groupBy("VPID")
            .agg(F.sum("ACTUAL_BARRELS").alias("CY_post_period_VOL"))
    )
    ly_df = (
        base.filter(F.col("CAL_DATE").isin(ly_dates))
            .groupBy("VPID")
            .agg(F.sum("ACTUAL_BARRELS").alias("LY_post_period_VOL"))
    )

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    if vpids:
        vpids_df = spark.createDataFrame([(v,) for v in vpids], ["VPID"])
    else:
        vpids_df = base.select("VPID").distinct()

    joined = vpids_df.join(cy_df, "VPID", "left").join(ly_df, "VPID", "left")
    filled = joined.select(
        "VPID",
        F.coalesce(F.col("CY_post_period_VOL"), F.lit(0.0)).cast("double").alias("CY_post_period_VOL"),
        F.coalesce(F.col("LY_post_period_VOL"), F.lit(0.0)).cast("double").alias("LY_post_period_VOL"),
    )

    return (
        filled.select(
            "VPID",
            F.round(F.col("CY_post_period_VOL"), 2).alias("CY_post_period_VOL"),
            F.round(F.col("LY_post_period_VOL"), 2).alias("LY_post_period_VOL"),
        )
        .orderBy("VPID")
    )

def build_variable_timing_caches(
    hv: DataFrame,
    hp: DataFrame,
    hr: DataFrame,
    hc: DataFrame,
    offsets: Iterable[int],
    base_start: date,
    base_end: date,
    base_measure_start: date,
    base_measure_end: date,
    brand_cluster_code: str,
    desired_premise: str,
    desired_retailer_channel: Sequence[str],
    vpids_for_post_period: Optional[Sequence[str]] = None,
    data_end_cap: Optional[date] = None,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    vol_pre_period_by_offset: Dict[int, pd.DataFrame] = {}
    vol_post_period_by_offset: Dict[int, pd.DataFrame] = {}

    for k in sorted(set(offsets)):
        s_k, e_k = shift_dates_by_weeks(base_start, base_end, k)
        ms_k, me_k = shift_dates_by_weeks(base_measure_start, base_measure_end, k)

        df_pre_period_k = get_pre_period_data_df_for_range(
            hv,
            hp,
            hr,
            hc,
            s_k,
            e_k,
            brand_cluster_code=brand_cluster_code,
            desired_premise=desired_premise,
            desired_retailer_channel=desired_retailer_channel,
            data_end_cap=data_end_cap,
        ).toPandas()
        vol_pre_period_by_offset[k] = df_pre_period_k

        df_post_period_k = get_post_period_data_df_for_range(
            hv,
            hp,
            hr,
            hc,
            ms_k,
            me_k,
            brand_cluster_code=brand_cluster_code,
            desired_premise=desired_premise,
            desired_retailer_channel=desired_retailer_channel,
            vpids=vpids_for_post_period,
            data_end_cap=data_end_cap,
        ).toPandas()
        vol_post_period_by_offset[k] = df_post_period_k

    return vol_pre_period_by_offset, vol_post_period_by_offset


def safe_trend(cy: float, ly: float) -> float:
    return np.nan if (pd.isna(ly) or ly == 0) else (cy / ly) - 1

# def aggregate_config_metrics(
#     cfg_name: str,
#     membership_df: pd.DataFrame,
#     result_pd_test: pd.DataFrame,
#     result_pd_control: pd.DataFrame,
#     hv,
#     hp,
#     hr,
#     hc,
#     vol_post_period_pd_cache,
# ) -> dict:
#     membership_df = membership_df.drop_duplicates(subset=["Group", "VPID"])

#     test_vpids = membership_df.loc[membership_df["Group"] == "Test", "VPID"].unique().tolist()
#     control_vpids = membership_df.loc[membership_df["Group"] == "Control", "VPID"].unique().tolist()
#     test_pre_period_by_vpid = (
#         result_pd_test.loc[result_pd_test["VPID"].isin(test_vpids), ["VPID", "CY_pre_period_VOL", "LY_pre_period_VOL"]]
#         .groupby("VPID", as_index=False)
#         .sum()
#     )
#     control_pre_period_by_vpid = (
#         result_pd_control.loc[result_pd_control["VPID"].isin(control_vpids), ["VPID", "CY_pre_period_VOL", "LY_pre_period_VOL"]]
#         .groupby("VPID", as_index=False)
#         .sum()
#     )

#     test_pre_period_cy = float(test_pre_period_by_vpid["CY_pre_period_VOL"].sum())
#     test_pre_period_ly = float(test_pre_period_by_vpid["LY_pre_period_VOL"].sum())
#     control_pre_period_cy = float(control_pre_period_by_vpid["CY_pre_period_VOL"].sum())
#     control_pre_period_ly = float(control_pre_period_by_vpid["LY_pre_period_VOL"].sum())

#     test_pre_period_trend = safe_trend(test_pre_period_cy, test_pre_period_ly)
#     control_pre_period_trend = safe_trend(control_pre_period_cy, control_pre_period_ly)

#     vpids_union = membership_df["VPID"].unique().tolist()
#     vol_post_period_df = get_post_period_data_df(hv, hp, hr, hc, vpids_union)
#     vol_post_period_pd = vol_post_period_df.toPandas()

#     vol_post_period_join = membership_df.merge(
#         vol_post_period_pd[["VPID", "CY_post_period_VOL", "LY_post_period_VOL"]],
#         on="VPID",
#         how="left",
#     )

#     test_post_period_cy = float(vol_post_period_join.loc[vol_post_period_join["Group"] == "Test", "CY_post_period_VOL"].sum())
#     test_post_period_ly = float(vol_post_period_join.loc[vol_post_period_join["Group"] == "Test", "LY_post_period_VOL"].sum())
#     control_post_period_cy = float(vol_post_period_join.loc[vol_post_period_join["Group"] == "Control", "CY_post_period_VOL"].sum())
#     control_post_period_ly = float(vol_post_period_join.loc[vol_post_period_join["Group"] == "Control", "LY_post_period_VOL"].sum())

#     test_post_period_trend = safe_trend(test_post_period_cy, test_post_period_ly)
#     control_post_period_trend = safe_trend(control_post_period_cy, control_post_period_ly)

#     return {
#         "config": cfg_name,

#         "test_pre_period_vol_cy": test_pre_period_cy,
#         "test_pre_period_vol_ly": test_pre_period_ly,
#         "test_pre_period_trend": test_pre_period_trend,
#         "control_pre_period_vol_cy": control_pre_period_cy,
#         "control_pre_period_vol_ly": control_pre_period_ly,
#         "control_pre_period_trend": control_pre_period_trend,

#         "test_post_period_vol_cy": test_post_period_cy,
#         "test_post_period_vol_ly": test_post_period_ly,
#         "test_post_period_trend": test_post_period_trend,
#         "control_post_period_vol_cy": control_post_period_cy,
#         "control_post_period_vol_ly": control_post_period_ly,
#         "control_post_period_trend": control_post_period_trend,

#     }
def aggregate_config_metrics_variable_timing(
    cfg_name: str,
    membership_df: pd.DataFrame,
    vol_pre_period_by_offset: Dict[int, pd.DataFrame],
    vol_post_period_by_offset: Dict[int, pd.DataFrame],
) -> dict:
    test_pre_period_cy = test_pre_period_ly = control_pre_period_cy = control_pre_period_ly = 0.0
    for k, g in membership_df.groupby("offset_weeks"):
        df6m = vol_pre_period_by_offset[int(k)]
        test_vpids = g.loc[g["Group"] == "Test", "VPID"]
        ctrl_vpids = g.loc[g["Group"] == "Control", "VPID"]
        t6 = df6m[df6m["VPID"].isin(test_vpids)][["CY_pre_period_VOL","LY_pre_period_VOL"]].sum()
        c6 = df6m[df6m["VPID"].isin(ctrl_vpids)][["CY_pre_period_VOL","LY_pre_period_VOL"]].sum()
        test_pre_period_cy += float(t6["CY_pre_period_VOL"]); test_pre_period_ly += float(t6["LY_pre_period_VOL"])
        control_pre_period_cy += float(c6["CY_pre_period_VOL"]); control_pre_period_ly += float(c6["LY_pre_period_VOL"])

    test_pre_period_trend = safe_trend(test_pre_period_cy, test_pre_period_ly)
    control_pre_period_trend = safe_trend(control_pre_period_cy, control_pre_period_ly)

    test_post_period_cy = test_post_period_ly = control_post_period_cy = control_post_period_ly = 0.0
    for k, g in membership_df.groupby("offset_weeks"):
        volpost_period = vol_post_period_by_offset[int(k)]
        joined = g.merge(volpost_period[["VPID","CY_post_period_VOL","LY_post_period_VOL"]], on="VPID", how="left")
        t3_cy = float(joined.loc[joined["Group"] == "Test","CY_post_period_VOL"].sum())
        t3_ly = float(joined.loc[joined["Group"] == "Test","LY_post_period_VOL"].sum())
        c3_cy = float(joined.loc[joined["Group"] == "Control","CY_post_period_VOL"].sum())
        c3_ly = float(joined.loc[joined["Group"] == "Control","LY_post_period_VOL"].sum())
        test_post_period_cy += t3_cy; test_post_period_ly += t3_ly
        control_post_period_cy += c3_cy; control_post_period_ly += c3_ly

    test_post_period_trend = safe_trend(test_post_period_cy, test_post_period_ly)
    control_post_period_trend = safe_trend(control_post_period_cy, control_post_period_ly)

    return {
        "config": cfg_name,
        "test_pre_period_vol_cy": test_pre_period_cy, "test_pre_period_vol_ly": test_pre_period_ly, "test_pre_period_trend": test_pre_period_trend,
        "control_pre_period_vol_cy": control_pre_period_cy, "control_pre_period_vol_ly": control_pre_period_ly, "control_pre_period_trend": control_pre_period_trend,
        "test_post_period_vol_cy": test_post_period_cy, "test_post_period_vol_ly": test_post_period_ly, "test_post_period_trend": test_post_period_trend,
        "control_post_period_vol_cy": control_post_period_cy, "control_post_period_vol_ly": control_post_period_ly, "control_post_period_trend": control_post_period_trend,
    }

