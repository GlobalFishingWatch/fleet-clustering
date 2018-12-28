import pandas as pd


def remove_chinese_coast(df):
    rows = []
    for x in df.itertuples():
        if 105 <= x.lon <= 130 and 10 <= x.lat <= 40:
            continue
        rows.append(x)
    return pd.DataFrame(rows)


def is_valid_mmsi(mmsi):
    if len(mmsi) != 9:
        return False
    if not '2' <= mmsi[0] <= '7':
        return False
    return True


def find_all_ssvid(df_by_date):
    all_ssvid = set()
    for v in df_by_date.values():
        all_ssvid |= set(v.ssvid)
    return all_ssvid


def find_valid_ssvid(df_by_date):
    all_ssvid = find_all_ssvid(df_by_date)
    return {x for x in all_ssvid if is_valid_mmsi(x)}