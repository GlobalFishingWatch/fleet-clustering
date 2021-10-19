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

from __future__ import print_function
from __future__ import division
from collections import Counter, OrderedDict
import datetime as dt
import hdbscan
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import numpy as np
import pandas as pd
from skimage import color
from IPython.display import HTML
from fleet_clustering import bq
from fleet_clustering import filters
from fleet_clustering import distances
# from fleet_clustering import animation

# ## Load AIS Clustering Data
#
# Load the AIS data that we use for clustering. Note that it onlyu includes vessels away
# from shores so as to exclude clustering on ports

all_by_date = bq.load_ais_by_date('drifting_longlines', dt.date(2016, 1, 1),
                                  dt.date(2016, 1, 31)
                                  #dt.date(2018, 12, 31),
                                 fishing_only=False, min_km_from_shore=0)    
pruned_by_date = {k : filters.remove_near_shore(10,
                            filters.remove_chinese_coast(v)) for (k, v) in all_by_date.items()}
valid_ssvid = sorted(filters.find_valid_ssvid(pruned_by_date))

# ## Create Distance Metrics
#
# Create an array of distance metrics. The details are still evolving, but in general
# we want to deal with two things.  Days on which a boat is missing and days where the
# boat is away from the fleet.
#
# * Distances to/from a boat on days when it is missing are represented by $\infty$ in 
#   the distance matrix. HDBSCAN ignores these values.
# * Only the closest N days are kept for each boat pair, allowing boats to leave the fleet
#   for up to half the year without penalty.
#   
# In addition, distances have a floor of 1 km to prevent overclustering when boats tie up
# up together, etc.

dists_by_date = {}

for start_date, end_date in [('20160101', '20161231'),
                             ('20170101', '20171231'), 
                             ('20180101', '20181231')]:
    if start_date in dists_by_date:
        continue
    print("computing distance for", start_date, end_date)
    subset_by_date = {k : v for (k, v) in pruned_by_date.items() if start_date <= k <= end_date}
    C = distances.create_composite_lonlat_array(subset_by_date, valid_ssvid)
    dists = distances.compute_distances_4(C, gamma=2)
    dists_by_date[start_date] = dists

# ## Load Carrier Data

carriers_by_date = bq.load_carriers_by_year(2017, 2018)
pruned_carriers_by_date = {k : filters.remove_chinese_coast(v) for (k, v) in carriers_by_date.items()}
query = """
               SELECT CAST(mmsi AS STRING) FROM
               `world-fishing-827.vessel_database.all_vessels_20190102`
               WHERE  iscarriervessel AND confidence = 3
        """
valid_carrier_ssvid_df = pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
valid_carrier_ssvid = valid_carrier_ssvid_df.f0_
valid_carrier_ssvid_set = set(valid_carrier_ssvid)

# ## Load Encounters Data And Country Codes
#
# This is used to filter the carrier vessels down to only those
# that meet with target vessels and to add iso3 labels to outputs

encounters = bq.load_carriers(2017, 2017)

query = """
SELECT code, iso3 FROM `world-fishing-827.gfw_research.country_codes`"""
country_codes_df = pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
iso3_map = {x.code : x.iso3 for x in country_codes_df.itertuples()}

# ## Fit the Clusterer
#
# This is pretty straightforward -- all the complicated stuff is
# embedded in the matrix computations. Fleet size can be tweaked
# using `min_cluster_size` and `min_sample_size`.

raw_clusterers = {}
for start_date, dists in dists_by_date.items():
    clusterer = hdbscan.HDBSCAN(metric='precomputed', 
                                min_cluster_size=9,
                               )
    clusterer.fit(dists)
    raw_clusterers[start_date] = clusterer

# ## Create Psuedo Distance From Fleet Membership

pdists_by_date = {}
for date in ['20160101', '20170101', '20180101']:
    pdists = np.zeros_like(dists_by_date[date])
    raw_labels = np.asarray(raw_clusterers[date].labels_)
    SCALE = 1000
    UNKNOWN_FLEET_DIST = 1 * SCALE
    OTHER_FLEET_DIST = 2 * SCALE
    mask = (raw_labels == -1)
    for i, fid in enumerate(raw_labels):
        if fid == -1:
            pdists[i] = UNKNOWN_FLEET_DIST
        else:
            pdists[i] = OTHER_FLEET_DIST * (raw_labels != fid)
            pdists[i, mask] = UNKNOWN_FLEET_DIST
    pdists_by_date[date] = pdists


# ## Set up Fleets
#
# Set up the fleets for viewing.

# +
def to_rgb(string):
    string = string.strip('#')
    r = string[:2]
    g = string[2:4]
    b = string[4:]
    return [int(x, 16) / 225.0 for x in (r, g, b)]


def find_labels(dists):
    clusterer = hdbscan.HDBSCAN(metric='precomputed', 
                                min_cluster_size=9).fit(dists)
    
    all_fleet_ssvid_set = set([s for (s, f) in zip(valid_ssvid, clusterer.labels_) if f >= 0])
    valid_ssvid_set = set(valid_ssvid)
    all_fleet_reefer_ssvid_set = set()
    for x in encounters.itertuples():
        if x.ssvid_1 in all_fleet_ssvid_set and x.ssvid_2 in valid_carrier_ssvid_set:
            all_fleet_reefer_ssvid_set.add(x.ssvid_2)
        if x.ssvid_2 in all_fleet_ssvid_set and x.ssvid_1 in valid_carrier_ssvid_set:
            all_fleet_reefer_ssvid_set.add(x.ssvid_1)
    all_fleet_reefer_ssvid = sorted(all_fleet_reefer_ssvid_set)

    valid_ssvid_set = set(valid_ssvid)
    carrier_ids = [x for x in all_fleet_reefer_ssvid if x not in valid_ssvid_set]
    joint_ssvid = valid_ssvid + sorted(carrier_ids) 
    labels = list(clusterer.labels_) + [max(clusterer.labels_) + 1] * len(carrier_ids) 

    # Remove vessels that have no connection to other vessels
    for i, ssvid in enumerate(valid_ssvid):
        connections = (~np.isinf(dists[i])).sum()
        if connections == 0:
            labels[i] = -1
            
    return joint_ssvid, labels


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
    
    
def iou(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b)

def best_match(a, bs):
    ious = [iou(a, b) for b in bs]
    i = np.argmax(ious)
    if ious[i] == 0:
        return None
    return i

    
def adapt_fleet_mapping(base_fleets, base_ssvid, base_labels, new_ssivd, new_labels):
    new_labels = np.array(new_labels)
    ssvid_base = []
    base_fleet_ids = sorted(base_fleets)
    for fid in base_fleet_ids:
        mask = (base_labels == fid)
        ssvid_base.append(np.array(base_ssvid)[mask])

    ssvid_new = []
    new_fleet_ids = sorted(set([x for x in new_labels if x != -1]))
    for fid in new_fleet_ids:
        mask = (new_labels == fid)
        ssvid_new.append(np.array(new_ssivd)[mask])

    rev_mapping = {}
    for fid, ssvid_list in zip(new_fleet_ids, ssvid_new):
        i = best_match(ssvid_list, ssvid_base)
        if i is None:
            rev_mapping[fid] = None
        else:
            rev_mapping[fid] = base_fleet_ids[i]
            
    mapping = {}
    for k, v in rev_mapping.items():
        if v in mapping:
            mask = (new_labels == k)
            new_labels[mask] = mapping[v]
        else:
            mapping[v] = k
                         
    new_fleets = OrderedDict()
    for i, fid in enumerate(base_fleets):
        if fid in mapping and mapping[fid] is not None:
            k = mapping[fid]
            if k in new_fleets:
                print("Skipping", k, fid, "because of double match")
                new_fleets[i + max(base_fleets)] = base_fleets[fid]
            else:
                new_fleets[mapping[fid]] = base_fleets[fid]
        else:
            new_fleets[i + max(base_fleets)] = base_fleets[fid]
            
    return new_fleets, new_labels
                         


# +
joint_ssvid_2017, labels_2017 = find_labels(dists_by_date['20170101'])
fleets_2017 = create_fleet_mapping(labels_2017)
all_by_date_2017 = {k : v for (k, v) in all_by_date.items() if '20170101' <= k <= '20171231'}

anim = animation.make_anim(joint_ssvid_2017, 
                           labels_2017, 
                           all_by_date_2017, 
                           interval=100,
                           fleets=fleets_2017, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
HTML(anim.to_html5_video())

# +
joint_ssvid_2016, labels_2016 = find_labels(dists_by_date['20160101'] +                                
                                            pdists_by_date['20170101'])
fleets_2016, labels_2016 = adapt_fleet_mapping(fleets_2017, joint_ssvid_2017, labels_2017, 
                                  joint_ssvid_2016, labels_2016)
all_by_date_2016 = {k : v for (k, v) in all_by_date.items() if '20160101' <= k <= '20161231'}

anim = animation.make_anim(joint_ssvid_2016, 
                           labels_2016, 
                           all_by_date_2016, 
                           interval=100,
                           fleets=fleets_2016, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
HTML(anim.to_html5_video())

# +
joint_ssvid_2018, labels_2018 = find_labels(dists_by_date['20180101'] +                                
                                            pdists_by_date['20170101'])
fleets_2018, labels_2018 = adapt_fleet_mapping(fleets_2017, joint_ssvid_2017, labels_2017, 
                                  joint_ssvid_2018, labels_2018)
all_by_date_2018 = {k : v for (k, v) in all_by_date.items() if '20180101' <= k <= '20181231'}

anim = animation.make_anim(joint_ssvid_2018, 
                           labels_2018, 
                           all_by_date_2018, 
                           interval=100,
                           fleets=fleets_2018, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
HTML(anim.to_html5_video())
# -

anim = animation.make_anim(joint_ssvid_2017, 
                           labels_2017, 
                           all_by_date_2017, 
                           interval=1,
                           fleets=fleets_2017, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
Writer = mpl_animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
anim.save('fleet_longlines_2017.mp4', writer=writer,  
          savefig_kwargs={'facecolor':'#222D4B'})

anim = animation.make_anim(joint_ssvid_2018, 
                           labels_2018, 
                           all_by_date_2018, 
                           interval=1,
                           fleets=fleets_2018, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
Writer = mpl_animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
anim.save('fleet_longlines_2018.mp4', writer=writer,  
          savefig_kwargs={'facecolor':'#222D4B'})

anim = animation.make_anim(joint_ssvid_2016, 
                           labels_2016, 
                           all_by_date_2016, 
                           interval=1,
                           fleets=fleets_2016, 
                           show_ungrouped=True,
                           alpha=1,
                           legend_cols=12,
                           ungrouped_legend="Ungrouped")
Writer = mpl_animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
anim.save('fleet_longlines_2016.mp4', writer=writer,  
          savefig_kwargs={'facecolor':'#222D4B'})


# ## Print Out Typical Fleet Membership

def print_fleets(fleets, labels, joint_ssvid):
    for fid, v in fleets.items():
        label = v[-1]
        mask = (fid == np.array(labels))
        ssvids = np.array(joint_ssvid)[mask]
        mids = [x[:3] for x in ssvids]
        countries = [iso3_map.get(float(x), x) for x in mids]
        c = Counter(countries)
        print('Fleet: {} ({})'.format(label, fid), label)
        for country, count in c.most_common():
            print('\t', country, ':', count)


print_fleets(fleets_2017, labels_2017, joint_ssvid_2017)

# ## Look for labor violations

print(2016)
available = set(mmsi) & set(joint_ssvid_2016)
for x in available:
    mask = (np.array(joint_ssvid_2016) == x)
    [fid] = np.array(labels_2016)[mask]
    if fid in fleets_2016:
        label = fleets_2016[fid][-1]
        print(x, label, fid)

# +
text = "312422000,2015;312422000,2014;312000125,2015;312000125,2014;412420941,2014;412420941,2015;412201837,2015;412201837,2016;413270430,2017;413270430,2016;440801000,2013;440801000,2014;533000000,2017;567000421,2015;567000445,2014;567000445,2015;567025800,2015;567025800,2014;416202800,2014;416202800,2015;416003928,2014;416054500,2017;416054500,2016;416001769,2013;416001769,2014;367363390,2015;576678000,2015;576678000,2014"
pairs = text.strip().split(';')
# Ignore years for now
mmsi = [x.split(',')[0] for x in pairs]

print(2017)
available = set(mmsi) & set(joint_ssvid_2017)
for x in available:
    mask = (np.array(joint_ssvid_2017) == x)
    [fid] = np.array(labels_2017)[mask]
    if fid != -1:
        label = fleets_2017[fid][-1]
        print(x, label, fid)
# -

print(2018)
available = set(mmsi) & set(joint_ssvid_2018)
for x in available:
    mask = (np.array(joint_ssvid_2018) == x)
    [fid] = np.array(labels_2018)[mask]
    if fid in fleets_2018:
        label = fleets_2018[fid][-1]
        print(x, label, fid)

mask = (np.array(labels_2017) == 28)
np.array(joint_ssvid_2017)[mask]

assert len(joint_ssvid_2017) == len(labels_2017)
with open('fleet_longlines_2017.csv', 'w') as f:
    f.write('mmsi,fleet\n')
    for mmsi, fid in zip(joint_ssvid_2017, labels_2017):
        if fid != -1 and fid in fleets_2017:
            f.write("{},{}\n".format(mmsi, fleets_2017[fid][-1]))
