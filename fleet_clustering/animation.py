from __future__ import print_function
from __future__ import division
from collections import Counter, defaultdict, OrderedDict
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
import numpy as np



def make_fleet_map(ssvids, labels):
    fleet_map = defaultdict(list)
    for s, lbl in zip(ssvids, labels):
        if lbl != -1:
            fleet_map[lbl].append(s)
    fleet_ids = fleet_map.keys()
    indices = np.argsort([len(fleet_map[x]) for x in fleet_ids])[::-1]
    ofleet_map = OrderedDict([(fleet_ids[i], fleet_map[fleet_ids[i]]) for i in indices])
    return ofleet_map


def make_anim(ssvids, labels, df_by_date, interval=1, max_fleets=30, region=None, fleets=None, alpha=1.0,
                show_ungrouped=True, legend_cols=None, ungrouped_legend=None):

    fleet_map = make_fleet_map(ssvids, labels)

    if fleets is None:
        fleet_ids = fleet_map.keys()
    else:
        fleet_ids = fleets.keys()
    n_fleets = min(max_fleets, len(fleet_ids))

    fig, ax = plt.subplots(figsize = (14, 7))

    if legend_cols is None:
        legend_cols = n_fleets // 4 + 1


    if region and region.lower() == 'mediterranean':
        projection = Basemap(projection='merc', llcrnrlat=29, urcrnrlat=49,
                             llcrnrlon=-10, urcrnrlon=40, resolution='l', ax=ax)
    else:
        projection = Basemap(lon_0=-155, projection='kav7', resolution="l", ax=ax)
    projection.fillcontinents(color='#BBBBBB',lake_color='#BBBBBB')

    points, = plt.plot([], [], '.', alpha=1, markersize=2, color='#dddddd')
    point_sets = [points]
    cmap = plt.get_cmap("tab10")
    CYCLE = 10
    for i, fid in enumerate(fleet_ids):
        if fleets and fid in fleets:
            marker, color, bgcolor, markersize, markeredgewidth, label = fleets[fid]
        else:
            color = bgcolor = cmap(i % CYCLE)
            marker = ['o', '+', 'x', '1'][i // CYCLE]
            markersize = [3, 6, 6, 6][i // CYCLE]
            markeredgewidth = 2
            cnts = Counter([x[:3] for x in fleet_map[fid]]).most_common()
            sigssvid = [s for (s, c) in cnts if c / len(fleet_map[fid]) > 0.2]
            label = ",".join(sigssvid) + "({})".format(fid)
        if fleets and fid not in fleets:
            label = None
        points, = plt.plot([], [], marker, alpha=alpha, markerfacecolor=bgcolor, markeredgecolor=color, 
                        markersize=markersize, markeredgewidth=markeredgewidth)
        # points, = plt.plot([], [], marker, alpha=1, markerfacecolor=color, markeredgecolor='k', markersize=markersize)
        point_sets.append(points)
        points, = plt.plot([], [], marker, alpha=alpha, markerfacecolor=bgcolor, markeredgecolor=color, 
                        markersize=markersize, label=label, markeredgewidth=markeredgewidth)
        point_sets.append(points)
    if ungrouped_legend:
        plt.plot([], [], '.', alpha=1, markersize=2, color='#888888', label=ungrouped_legend)


    title = plt.title('DATE')
    legend = plt.legend(loc=8, framealpha=0.8, ncol=legend_cols)
    if legend is not None:
        for lh in legend.legendHandles: 
            lh._legmarker.set_alpha(1)
        legend.get_frame().set_linewidth(0.0)

    plt.tight_layout()

    def init():
        for i in range(2 * n_fleets + 1):
            point_sets[i].set_data([], [])
        return point_sets[::-1]
        
    ssvid_set = set(ssvids)

    def animate(i):
        datestr = sorted(df_by_date)[i*interval]
        df = df_by_date[datestr]
        if show_ungrouped:
            lons, lats = projection(df.lon.values, df.lat.values)
            mask = [y in ssvid_set for y in df.ssvid]
            point_sets[0].set_data(lons[mask], lats[mask])
        else:
            point_sets[0].set_data([], [])
        for j, fid in enumerate(fleet_ids):
            fid = fleet_ids[j]
            if fleets and fid not in fleets:
                continue
            fleet_ssvids = set(fleet_map[fid])
            mask = [(y.ssvid in fleet_ssvids and bool(y.iscarrier)) for y in df.itertuples()]
            if sum(mask):
                lons, lats = projection(df.lon[mask].values, df.lat[mask].values)
            else:
                lons = lats = []
            point_sets[2 * j + 1].set_data(lons, lats)
            mask = [(y.ssvid in fleet_ssvids and not y.iscarrier) for y in df.itertuples()]
            if sum(mask):
                lons, lats = projection(df.lon[mask].values, df.lat[mask].values)
            else:
                lons = lats = []
            point_sets[2 * j + 2].set_data(lons, lats)
        title.set_text(datestr)
        return point_sets[::-1]

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(df_by_date)//interval, interval=500, 
                                   blit=True)