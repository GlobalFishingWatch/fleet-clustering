from __future__ import print_function
from __future__ import division
from collections import Counter, defaultdict, OrderedDict
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from skimage import io
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
                show_ungrouped=True, legend_cols=None, ungrouped_legend=None, lon_0=-155,
                figsize=None, bottom_adjust=0.11, top_adjust=None, 
                logo_loc=(0.25, 0.11, .16, .16), text_color='white', plot_frame=None):

    fleet_map = make_fleet_map(ssvids, labels)

    if fleets is None:
        fleet_ids = fleet_map.keys()
    else:
        fleet_ids = fleets.keys()
    n_fleets = min(max_fleets, len(fleet_ids))

    if legend_cols is None:
        legend_cols = n_fleets // 20 + 1

    if region and region.lower() == 'mediterranean':
        if figsize is None:
            figsize = (20, 10)
        fig, ax = plt.subplots(figsize=figsize)  
        projection = Basemap(projection='merc', llcrnrlat=29, urcrnrlat=49,
                             llcrnrlon=-10, urcrnrlon=40, resolution='l', ax=ax)
    elif region and region.lower() == 'europe':
        if figsize is None:
            figsize = (10, 15)
        fig, ax = plt.subplots(figsize=figsize)
        projection = Basemap(projection='merc', llcrnrlat=15, urcrnrlat=70,
                             llcrnrlon=-20, urcrnrlon=40, resolution='l', ax=ax)
    else:
        if figsize is None:
            figsize = (16, 10)
        fig, ax = plt.subplots(figsize=figsize)
        projection = Basemap(lon_0=lon_0, projection='robin', resolution="l", ax=ax)
    projection.fillcontinents(color='#37496D', lake_color='#0A1738')
    projection.drawcountries(color='#222D4B')
    projection.drawmapboundary(fill_color='#0A1738', color='#222D4B')
    projection.drawmapboundary(fill_color='none', color='#222D4B', 
        zorder=1000, linewidth=4)

    points, = plt.plot([], [], '.', alpha=1, markersize=2, color='#777777')
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
        plt.plot([], [], '.', alpha=1, markersize=2, color='#777777', label=ungrouped_legend)


    title = plt.title('DATE', color=text_color)
    legend = plt.legend(bbox_to_anchor=(0.5, -0.03), loc="upper center", ncol=legend_cols, facecolor="none",
        framealpha=1.0, edgecolor='none')
    if legend is not None:
        for lh in legend.legendHandles: 
            lh._legmarker.set_alpha(1)
        for text in legend.get_texts():
            text.set_color(text_color)

    # this is another inset axes over the main axes
    a = plt.axes(logo_loc, facecolor='none')
    img = io.imread('GFW_logo_primary_RGB.png')
    plt.imshow(img)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_adjust, top=top_adjust)

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
            fleet_ssvids = set(fleet_map.get(fid, ()))
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

    if plot_frame is not None:
        init()
        animate(plot_frame)
        return fig
    else:
        return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(df_by_date)//interval, interval=250, 
                                   blit=True)