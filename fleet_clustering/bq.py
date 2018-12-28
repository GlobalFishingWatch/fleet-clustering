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


def load_raw_ais(vessel_types, start_yyyymmdd, end_yyyymmdd, min_km_from_shore=10):
    if not isinstance(vessel_types, (tuple, list)):
        vessel_types = [vessel_types]
    vessel_types_str = ', '.join(['"{}"'.format(x) for x in vessel_types])
    query = """
    SELECT ssvid,
           year,
           month,
           day,
           180 / ACOS(-1) * ATAN2(lonsin, loncos) AS lon,
           lat
           
    FROM (
        SELECT a.ssvid, 
               EXTRACT(YEAR FROM timestamp) year,
               EXTRACT(MONTH FROM timestamp) month,
               EXTRACT(DAY FROM timestamp) day,
               AVG(SIN(ACOS(-1) / 180 * a.lon)) AS lonsin, 
               AVG(COS(ACOS(-1) / 180 * a.lon)) AS loncos, 
               AVG(a.lat) AS lat 
        FROM 
        `world-fishing-827.pipe_production_b.position_messages_*` a
            JOIN
        `world-fishing-827.gfw_research.vessel_info_allyears_20181002` b
           ON a.ssvid = CAST(b.mmsi AS STRING)
        WHERE _TABLE_SUFFIX BETWEEN "{}" AND "{}"
        AND seg_id in (select seg_id from gfw_research.pipe_production_b_segs where good_seg)
        AND a.distance_from_shore_m > {}
        AND b.best_label in ({})
        GROUP BY a.ssvid, year, month, day
    )
    ORDER BY ssvid
    """.format(start_yyyymmdd, end_yyyymmdd, 1000 * min_km_from_shore, vessel_types_str)
    return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')


def load_ais_by_date(vessel_types, start_date, end_date, min_km_from_shore=10):
    df = load_raw_ais(vessel_types, "{:%Y%m%d}".format(start_date), "{:%Y%m%d}".format(end_date), min_km_from_shore)
    df_by_date = {}
    d = start_date
    while d <= end_date:
        datestr = "{:4d}{:02d}{:02d}".format(d.year, d.month, d.day)
        mask = (df.year == d.year) & (df.month == d.month) & (df.day == d.day)
        if mask.sum():
            df_by_date[datestr] = df[mask]
        d += datetime.timedelta(days=1)
    return df_by_date

