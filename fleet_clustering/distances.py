import numpy as np


def create_lonlat_array(df, valid_ssvid):
    """
    Unlabelled rows are 0 (either because vessel
    no broadcasting on day or not grouped)
    
    Other rows are +1 for cluster and -1 elsewhere
    """
    n_ssvid = len(valid_ssvid)
    ssvid_map = {ssvid : i for (i, ssvid) in enumerate(df.ssvid)}
    lonlats = df[['lon', 'lat']].values
    C = np.empty([n_ssvid, 2])
    C.fill(np.nan)
    for i, ssvid in enumerate(valid_ssvid):
        if ssvid not in ssvid_map:
            continue
        j = ssvid_map[ssvid]
        C[i, :] = lonlats[j]
    return C

def create_composite_lonlat_array(df_by_date, valid_ssvid):
    Cns = []
    for date in sorted(df_by_date):
        Cns.append(create_lonlat_array(df_by_date[date], valid_ssvid))
    return np.concatenate(Cns, axis=1)

def compute_distances(C, clip=np.inf):
    # Compute the distances, ignoring infs
    AVG_EARTH_RADIUS = 6371  # in km
    n = len(C)
    distances = np.zeros([n, n])
    Cr = np.radians(C)
    for i in range(n):
        d2s = []
        for j in range(0, Cr.shape[1], 2):
            lat1 = Cr[i, None, j + 1]
            lat2 = Cr[:, j + 1]
            lng1 = Cr[i, None, j]
            lng2 = Cr[:, j]
            lat = lat2 - lat1
            lng = lng2 - lng1
            d = np.sin(lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng / 2) ** 2
            h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
            h = np.minimum(h, clip)
            d2s.append(h ** 2)
        distances[i] = np.sqrt(np.nanmean(d2s, axis=0))
    distances[np.isnan(distances)] = np.inf
    return distances

def infclip(X, v):
    Y = np.minimum(X, v)
    Y[np.isinf(X)] = np.inf
    return Y
