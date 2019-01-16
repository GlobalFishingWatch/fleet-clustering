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

def load_raw_ais(vessel_types, start_yyyymmdd, end_yyyymmdd, min_km_from_shore=10, include_carriers=False, 
                fishing_only=False, show_query=False):
    if not isinstance(vessel_types, (tuple, list)):
        vessel_types = [vessel_types]
    if include_carriers:
        extra_condition = "OR (c.iscarriervessel AND c.confidence = 3)"
    else:
        extra_condition = ""
    if fishing_only:
        fishing_condition = "AND a.nnet_score > 0.5"
    else:
        fishing_condition = ""
    vessel_types_str = ', '.join(['"{}"'.format(x) for x in vessel_types])
    query = """
    SELECT ssvid,
           year,
           month,
           day,
           lon,
           lat,
           iscarrier
    FROM (
        SELECT a.ssvid, 
               EXTRACT(YEAR FROM timestamp) year,
               EXTRACT(MONTH FROM timestamp) month,
               EXTRACT(DAY FROM timestamp) day,
               a.lon AS lon,
               a.lat AS lat,
               ROW_NUMBER() OVER(PARTITION BY a.ssvid,  TIMESTAMP_TRUNC(a.timestamp, DAY)
                                 ORDER BY ABS(TIMESTAMP_DIFF(a.timestamp , 
                                              TIMESTAMP_TRUNC(a.timestamp, DAY), SECOND) - 12 * 60 * 60 ) ASC) AS rk,
               c.iscarriervessel AND c.confidence = 3 AS iscarrier
        FROM 
        `world-fishing-827.pipe_production_b.messages_scored_*` a
            JOIN
        `world-fishing-827.gfw_research.vessel_info_allyears_20181002` b
            ON a.ssvid = CAST(b.mmsi AS STRING)
            JOIN 
        `world-fishing-827.vessel_database.all_vessels_20190102` c
            ON a.ssvid = CAST(c.mmsi AS STRING)
        WHERE _TABLE_SUFFIX BETWEEN "{}" AND "{}"
        AND seg_id in (select seg_id from gfw_research.pipe_production_b_segs where good_seg)
        AND a.distance_from_shore_m > {}
        AND ( (b.best_label in ({}) {}) {} )
    )
    WHERE rk = 1
    ORDER BY ssvid
    """.format(start_yyyymmdd, end_yyyymmdd, 1000 * min_km_from_shore, vessel_types_str, fishing_condition, extra_condition)
    if show_query:
        print(query)
    return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')


def load_ais_by_date(vessel_types, start_date, end_date, min_km_from_shore=10, include_carriers=False, fishing_only=False):
    dfs = []
    d0 = start_date
    while d0 < end_date:
        print(d0)
        d1 = min(d0 + datetime.timedelta(days=366), end_date)
        df = load_raw_ais(vessel_types, "{:%Y%m%d}".format(d0), "{:%Y%m%d}".format(d1), 
                            min_km_from_shore, include_carriers, fishing_only)
        dfs.append(df)
        d0 = d1 + datetime.timedelta(days=1)
    df = pd.concat(dfs)
    df_by_date = {}
    d = start_date
    while d <= end_date:
        datestr = "{:4d}{:02d}{:02d}".format(d.year, d.month, d.day)
        mask = (df.year == d.year) & (df.month == d.month) & (df.day == d.day)
        if mask.sum():
            df_by_date[datestr] = df[mask]
        d += datetime.timedelta(days=1)
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
        `world-fishing-827.pipe_production_b.encounters` AS a
        JOIN
        `world-fishing-827.pipe_production_b.vessel_info` AS b
        ON a.vessel_1_id = b.vessel_id
        JOIN
        `world-fishing-827.pipe_production_b.vessel_info` AS c
        ON a.vessel_2_id = c.vessel_id
        WHERE 
        (
            b.ssvid in (
               SELECT CAST(mmsi AS STRING) FROM
               `world-fishing-827.vessel_database.all_vessels_20190102`
               WHERE  iscarriervessel AND confidence = 3
                  )
        OR
            c.ssvid in (
               SELECT CAST(mmsi AS STRING) FROM
               `world-fishing-827.vessel_database.all_vessels_20190102`
               WHERE  iscarriervessel AND confidence = 3
                  )
        )
        AND
             {} <= EXTRACT(YEAR FROM start_time) and EXTRACT(YEAR FROM start_time) <= {}
    """.format(start_year, end_year)
    return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')


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
