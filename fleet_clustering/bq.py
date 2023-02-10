from __future__ import print_function
from __future__ import division
from collections import defaultdict, OrderedDict, Counter
import calendar
import datetime
import dateutil.parser as dparser
from glob import glob
import hdbscan
import logging
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib_scalebar.scalebar import ScaleBar
import os
import numpy as np
import pandas as pd
import skimage
import skimage.io as skio
from skimage import draw
from sklearn.metrics import pairwise_distances
import subprocess
import sys
import time
import ujson as json


def load_raw_ais(
    vessel_types,
    start_yyyymmdd,
    end_yyyymmdd,
    min_km_from_shore=10,
    include_carriers=False,
    fishing_only=False,
    show_query=False,
    ssvid=(),
):
    if not isinstance(vessel_types, (tuple, list)):
        vessel_types = [vessel_types]
    if include_carriers:
        extra_condition = "OR (iscarrier)"
    elif len(ssvid):
        extra_condition = "or a.ssvid in ({})".format(
            ",".join('"{}"'.format(x) for x in ssvid)
        )
    else:
        extra_condition = ""
    if fishing_only:
        fishing_condition = "AND a.nnet_score > 0.5"
    else:
        fishing_condition = ""
    vessel_types_str = ", ".join(['"{}"'.format(x) for x in vessel_types])
    query = """
    WITH 
    base as (
        SELECT ssvid, 
               EXTRACT(YEAR FROM timestamp) year,
               EXTRACT(MONTH FROM timestamp) month,
               EXTRACT(DAY FROM timestamp) day,
               lon,
               lat,
               TIMESTAMP_TRUNC(timestamp, MINUTE) AS minute_stamp,
               distance_from_shore_m / 1000.0 AS distance_from_shore_km
        FROM 
        `world-fishing-827.pipe_production_v20201001.messages_scored_*`
        WHERE _TABLE_SUFFIX BETWEEN "{}" AND "{}"
        AND seg_id in (select seg_id from `gfw_research.pipe_v20201001_segs` where good_seg and not overlapping_and_short)
        AND distance_from_shore_m >= {}
    ),
    thinned as (
        SELECT ssvid, year, month, day, 
               APPROX_QUANTILES(lon, 2)[OFFSET(1)] AS lon,
               APPROX_QUANTILES(lat, 2)[OFFSET(1)] AS lat,
               APPROX_QUANTILES(distance_from_shore_km, 2)[OFFSET(1)] AS distance_from_shore_km,
               minute_stamp
        FROM 
        base
        GROUP BY 
        minute_stamp, ssvid, year, month, day
    )


    SELECT ssvid,
           year,
           month,
           day,
           lon,
           lat,
           iscarrier,
           distance_from_shore_km,
           ssvid IN (SELECT DISTINCT mmsi FROM
           # `world-fishing-827.scratch_jaeyoon.twn_foc_final_mmsis_flat`) AS is_foc
           `scratch_cylai.00_01_FOC_checked_veseel_FG_flat`) AS is_foc
    FROM (
        SELECT a.ssvid, 
               a.year,
               a.month,
               a.day,
               a.lon AS lon,
               a.lat AS lat,
               ROW_NUMBER() OVER(PARTITION BY a.ssvid,  TIMESTAMP_TRUNC(a.minute_stamp, DAY)
                                 ORDER BY ABS(TIMESTAMP_DIFF(a.minute_stamp , 
                                              TIMESTAMP_TRUNC(a.minute_stamp, DAY), minute) - 12 * 60 ) ASC) AS rk,
               c.is_carrier AND r.confidence = 3 AS iscarrier,
               distance_from_shore_km
        FROM 
        thinned a
            JOIN
        `world-fishing-827.gfw_research.vi_ssvid_v20230101` b
            ON a.ssvid = CAST(b.ssvid AS STRING)
            JOIN 
        `world-fishing-827.vessel_database.all_vessels_v20230101` c
            ON a.ssvid = CAST(c.identity.ssvid AS STRING),
            unnest (registry) as r
        WHERE ( (b.best.best_vessel_class in ({}) {}) {} )
    )
    WHERE rk = 1
    ORDER BY ssvid
    """.format(
        start_yyyymmdd,
        end_yyyymmdd,
        1000 * min_km_from_shore,
        vessel_types_str,
        fishing_condition,
        extra_condition,
    )
    if show_query:
        print(query)
    return pd.read_gbq(query, dialect="standard", project_id="world-fishing-827")


def load_ais_by_date(
    vessel_types,
    start_date,
    end_date,
    min_km_from_shore=10,
    include_carriers=False,
    fishing_only=False,
    show_queries=False,
    ssvid=(),
):
    dfs = []
    d0 = start_date
    while d0 < end_date:
        print(d0)
        d1 = min(d0 + datetime.timedelta(days=183), end_date)
        df = load_raw_ais(
            vessel_types,
            "{:%Y%m%d}".format(d0),
            "{:%Y%m%d}".format(d1),
            min_km_from_shore,
            include_carriers,
            fishing_only,
            show_query=show_queries,
            ssvid=ssvid,
        )
        dfs.append(df)
        d0 = d1 + datetime.timedelta(days=1)
    df = pd.concat(dfs)
    df.sort_values(by=["year", "month", "day"], inplace=True)
    df_by_date = {}
    d = start_date
    start_ndx = 0
    while d <= end_date:
        print(".", end=".", flush=True)
        for i, x in enumerate(df.iloc[start_ndx:].itertuples()):
            if datetime.date(x.year, x.month, x.day) >= d:
                break
            start_ndx += 1
        end_ndx = start_ndx
        for i, x in enumerate(df.iloc[end_ndx:].itertuples()):
            if datetime.date(x.year, x.month, x.day) > d:
                break
            end_ndx += 1
        if end_ndx > start_ndx:
            datestr = "{:4d}{:02d}{:02d}".format(d.year, d.month, d.day)
            df_by_date[datestr] = df.iloc[start_ndx:end_ndx].reset_index(drop=True)
        d += datetime.timedelta(days=1)
        start_ndx = end_ndx
    return df_by_date


def load_carriers(start_year, end_year):
    query = """
        SELECT b.ssvid as ssvid_1, 
               c.ssvid as ssvid_2,
               EXTRACT(YEAR FROM start_time) year,
               EXTRACT(MONTH FROM start_time) month,
               EXTRACT(DAY FROM start_time) day,
               mean_longitude AS lon,
               mean_latitude as lat, 
               1 AS iscarrier
        FROM 
        `world-fishing-827.pipe_production_v20201001.encounters` AS a
        JOIN
        `world-fishing-827.pipe_production_v20201001.vessel_info` AS b
        ON a.vessel_1_id = b.vessel_id
        JOIN
        `world-fishing-827.pipe_production_v20201001.vessel_info` AS c
        ON a.vessel_2_id = c.vessel_id
        WHERE 
        (
            b.ssvid in (
               SELECT CAST( identity.ssvid AS STRING) FROM
               `world-fishing-827.vessel_database.all_vessels_v20230101`,
               unnest (registry) as r
               WHERE  is_carrier AND r.confidence = 3
                  )
        OR
            c.ssvid in (
               SELECT CAST( identity.ssvid AS STRING) FROM
               `world-fishing-827.vessel_database.all_vessels_v20230101`,
               unnest (registry) as r
               WHERE  is_carrier AND r.confidence = 3
                  )
        )
        AND
             {} <= EXTRACT(YEAR FROM start_time) and EXTRACT(YEAR FROM start_time) <= {}
    """.format(
        start_year, end_year
    )
    return pd.read_gbq(query, dialect="standard", project_id="world-fishing-827")


def load_carriers_by_year(start_year, end_year):
    df = load_carriers(start_year, end_year)
    df_by_date = {}
    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime(end_year, 12, 31)
    d = start_date
    while d <= end_date:
        datestr = "{:4d}{:02d}{:02d}".format(d.year, d.month, d.day)
        mask = (df.year == d.year) & (df.month == d.month) & (df.day == d.day)
        if mask.sum():
            df_by_date[datestr] = df[mask]
        d += datetime.timedelta(days=1)
    return df_by_date
