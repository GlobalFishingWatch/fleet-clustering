# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Fleet Clustering
#
# ### Tim Hochberg, 2019-01-16
#
# ## Longliner Edition
#
# We cluster vessel using HDBSCAN and a custom metric to derive fleets
# that are related in the sense that they spend a lot of time in the same
# location while at sea.
#
# ## See Also
#
# * Other notebooks in https://github.com/GlobalFishingWatch/fleet-clustering for 
# examples of clustering Squid Jiggers, etc.
# * This workspace that Nate put together: https://globalfishingwatch.org/map/workspace/udw-v2-85ff8c4f-fbfe-4126-b067-4d94cdd2b737

# +
from __future__ import print_function
from __future__ import division
from collections import Counter, namedtuple, OrderedDict
import datetime as dt
import hdbscan
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
from IPython.display import HTML
from fleet_clustering import bq
from fleet_clustering import filters
from fleet_clustering import animation

import pyseas.maps as psm
# -

# ## Load AIS Clustering Data
#
# Load the AIS data that we use for clustering. Note that it onlyu includes vessels away
# from shores so as to exclude clustering on ports

all_by_date = bq.load_ais_by_date('drifting_longlines', dt.date(2017, 1, 1),
                                  dt.date(2017, 12, 31),
                                 fishing_only=False, min_km_from_shore=0)    
pruned_by_date = {k : filters.remove_near_shore(10,
                            filters.remove_chinese_coast(v)) for (k, v) in all_by_date.items()}
# valid_ssvid = sorted(filters.find_valid_ssvid(pruned_by_date))

# ## Create Distance Metrics
#
# Following https://arxiv.org/pdf/1104.1990.pdf, we compute raw distance matrices for each
# date, then iteratively update the "actual" distance matrices using an EWMA.

# +
Metrics = namedtuple("Weights", ["id", "day", "dist"])

AVG_EARTH_RADIUS = 6371  # in km


def compute_raw_metrics(pos, day, min_clip=1, max_clip=np.inf):
    assert len(pos) == len(set(pos.ssvid))
    lat1 = np.deg2rad(pos.lat.values[:, None])
    lat2 = np.deg2rad(pos.lat.values[None, :])
    lon1 = np.deg2rad(pos.lon.values[:, None])
    lon2 = np.deg2rad(pos.lon.values[None, :])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    h = np.clip(h, min_clip, max_clip)
    days = np.empty(h.shape)
    days.fill(day)
    return Metrics(pos.id, days, h)


def remap_weights(weights, ids):
    new = np.empty([len(ids), len(ids)])
    new.fill(np.inf)
    id_map = {id_: j for (j, id_) in enumerate(ids)}
    mask = np.zeros([len(ids)], dtype=bool)
    days = np.zeros(new.shape, dtype=int)
    valid_ids = np.empty([len(ids)], dtype=object)
    valid_ids.fill(None)
    for i, id_ in enumerate(weights.id):
        j = id_map[id_]
        mask[j] = True
        valid_ids[j] = weights.id[i]
    for i, id_ in enumerate(weights.id):
        new[id_map[id_], mask] = weights.dist[i, :]
        new[mask, id_map[id_]] = weights.dist[:, i]  # Needed ?
        days[id_map[id_], mask] = weights.day[i, :]
        days[mask, id_map[id_]] = weights.day[:, i]

    return Metrics(valid_ids, days, new)


def compute_metrics(pos_by_date, start, days, gamma=2.0, scale=365):
    metrics = []
    # TODO: don't need date anymore
    ids = set()
    for i in range(days):
        date = start + dt.timedelta(days=i)
        pos = pos_by_date[f"{date:%Y%m%d}"]

        pos["id"] = pos["ssvid"]  # TODO, do on load
        ids |= set(pos_by_date[f"{date:%Y%m%d}"].id)
    ids = np.array(sorted(ids))
    for i in range(days):
        print(".", end="", flush=True)
        date = start + dt.timedelta(days=i)
        pos = pos_by_date[f"{date:%Y%m%d}"].reset_index(drop=True)
        raw = compute_raw_metrics(pos, i)
        metrics.append(remap_weights(raw, ids).dist)
    metrics = np.array(metrics)
    print("merging")
    metrics[np.isnan(metrics)] = np.inf
    metrics.sort(axis=0)
    # This is the weighting function that is in the current code.
    # It weights the distances from 1 for the closest to 0 for the
    # the furthest. The shape of the curve is controlled by gamma
    # and scale. Only the scale closest items have any weight
    weights = (
        np.maximum(np.linspace(1, 1 - metrics.shape[0] / scale, metrics.shape[0]), 0)
        ** gamma
    )
    weights = weights[:, None, None] + np.zeros_like(metrics)
    mask = np.isinf(metrics)
    metrics[mask] = 0
    weights[mask] = 0
    norm = weights.sum(axis=0)
    merged = (weights * metrics).sum(axis=0) / (norm + 1e-10)
    merged[norm == 0] = np.inf
    return Metrics(ids, None, merged)


# +
start_date = dt.date(2017, 1, 1)

metric = compute_metrics(pruned_by_date, start_date, 365)
clusterer = hdbscan.HDBSCAN(metric='precomputed', 
                            min_cluster_size=9,
                           )
clusterer.fit(metric.dist)


# +
def plot_most_common(clusters, n=10):
    clusterer, id_, pos = clusters
    labels = clusterer.labels_
    with psm.context(psm.styles.light):
        plt.figure(figsize=(18, 9))
        ax = psm.create_map()
        psm.add_land()
        overall_mask = False
        for ndx, n in Counter([x for x in labels if x != -1]).most_common(n):
            ids = set(id_[labels == ndx])
            mask = np.array([(x in ids) for x in pos.id])
            overall_mask |= mask
            cluster_pos = pos[mask]
            ax.plot(cluster_pos.lon, cluster_pos.lat, '.', transform=psm.identity)
        ax.plot(pos[~overall_mask].lon, pos[~overall_mask].lat, 'k.', 
                transform=psm.identity, markersize=1)
        ax.set_global()
    plt.show()
        
cluster = (clusterer, metric.id, pruned_by_date["20170615"])
plot_most_common(cluster)


# -

def create_fleet_mapping(labels, include_carriers=False):
    counts = []
    skip = []
    for i in range(max(labels) + 1):
        if i in skip:
            counts.append(0)
        else:
            counts.append((np.array(labels) == i).sum())

    fleet_ids = [x for x in np.argsort(counts)[::-1] if counts[x] > 0]
    fleet_ids_without_carriers = [x for x in fleet_ids if x != max(labels)]

    fleets = OrderedDict()
    n_hues = int(np.ceil(len(fleet_ids) / 4.0))
    used = set()
    for i, fid in enumerate(fleet_ids_without_carriers):
        b = (i // (2 * n_hues)) % 2
        c = (i // 2)% n_hues
        d = i  % 2
        symbol = 'H^'[d]
        assert (b, c, d) not in used, (i, b, c, d)
        used.add((b, c, d))
        sat = 1
        val = 1
        raw_hue = c / float(n_hues)
        # We remap the raw hue in order to avoid the 60 degree segment around blue
        hue = 5. / 6. * raw_hue
        if hue > 7. / 12.:
            hue += 1. / 6.
        assert 0 <= hue < 1, hue
        [[clr]] = color.hsv2rgb([[(hue, sat, val)]])
        fg = [[0.1511111111111111, 0.2, 0.3333333333333333], clr][b]
        bg = [clr, [0.1511111111111111, 0.2, 0.3333333333333333]][b]
        w = [1, 2][b]
        sz = [9, 7][b]
        fleets[fid] = (symbol, tuple(fg), tuple(bg), sz, w,  str(i + 1))
    if include_carriers:
        fleets[max(labels)] = ('1', 'k', 'k', 8, 2, 'Carrier Vessel')
    print(len(set([x for x in fleets if x != -1])), "fleets")
    return fleets


# +
fleets = create_fleet_mapping(clusterer.labels_)

anim = animation.make_anim(list(metric.id),
                           list(clusterer.labels_),
                           pruned_by_date, 
                           interval=5,
                           fleets=fleets, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
HTML(anim.to_html5_video())

# +


Writer = mpl_animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
anim.save('fleet_longlines_2017.mp4', writer=writer,  
          savefig_kwargs={'facecolor':'#222D4B'})
