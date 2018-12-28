from __future__ import print_function
from __future__ import division
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
import numpy as np


def make_anim(ssvids, labels, df_by_date, interval=1, max_fleets=20):


    n_fleets = min(max_fleets, max(labels) + 1)

    fleet_map = defaultdict(list)
    for s, lbl in zip(ssvids, labels):
        if lbl != -1:
            fleet_map[lbl].append(s)

    fleet_ids = fleet_map.keys()
    fleet_lengths = [len(fleet_map[x]) for x in fleet_ids]
    indices = np.argsort(fleet_lengths)[::-1]

    fig, ax = plt.subplots(figsize = (14, 7))

    projection = Basemap(lon_0=-155, projection='eck4', resolution="l", ax=ax)
    projection.fillcontinents(color='#BBBBBB',lake_color='#BBBBBB')

    points, = plt.plot([], [], '.', alpha=1, markersize=4, color='#dddddd')
    point_sets = [points]
    cmap = plt.get_cmap("tab10")
    CYCLE = 9
    for i in range(n_fleets):
        color = cmap(i % CYCLE)
        marker = ['.', 'x', '+'][i // CYCLE]
        markersize = [12, 8, 8][i // CYCLE]
        fid = fleet_ids[indices[i]]
        cnts = Counter([x[:3] for x in fleet_map[fid]]).most_common()
        sigssvid = [s for (s, c) in cnts if c / len(fleet_map[fid]) > 0.1]
        label = ",".join(sigssvid)
        points, = plt.plot([], [], marker, alpha=1, color=color, markersize=markersize, label=label)
        point_sets.append(points)


    title = plt.title('DATE')
    legend = plt.legend(loc=2, framealpha=0.5, ncol=n_fleets//10 + 1)
    if legend is not None:
        for lh in legend.legendHandles: 
            lh._legmarker.set_alpha(1)
        legend.get_frame().set_linewidth(0.0)

    plt.tight_layout()

    def init():
        for i in range(n_fleets + 1):
            point_sets[i].set_data([], [])
            return point_sets
        
    ssvid_set = set(ssvids)

    def animate(i):
        datestr = sorted(df_by_date)[i*interval]
        df = df_by_date[datestr]
        lons, lats = projection(df.lon.values, df.lat.values)
        mask = [y in ssvid_set for y in df.ssvid]
        point_sets[0].set_data(lons[mask], lats[mask])
        for j, n in enumerate(indices[:n_fleets]):
            fid = fleet_ids[n]
            fleet_ssvids = set(fleet_map[fid])
            mask = [y in fleet_ssvids for y in df.ssvid]
            lons, lats = projection(df.lon[mask].values, df.lat[mask].values)
            point_sets[j+1].set_data(lons, lats)
        title.set_text(datestr)
        return point_sets

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(df_by_date)//interval, interval=500, 
                                   blit=True)