import json
import os
import pandas as pd

from pickles import pickling


def load_df_pubs(json_path,pickles_path=None,refresh_pickle=False):
    if (not os.path.exists(pickles_path)) or refresh_pickle:
        with open(json_path, 'rb') as infile:
            raw_data = json.load(infile)
            names = raw_data.keys()
            pubs_dfs = {auth_name:pd.DataFrame(raw_data[auth_name]) for auth_name in names}
            pickling.pickle_object(pubs_dfs, pickles_path)
    else:
        pubs_dfs = pickling.unpickle_object(pickles_path)
    return pubs_dfs


def load_assignment(json_path,pickles_path=None,refresh_pickle=False):
    if (not os.path.exists(pickles_path)) or refresh_pickle:
        with open(json_path, 'rb') as infile:
            raw_data = json.load(infile)
            pickling.pickle_object(raw_data, pickles_path)
    else:
        raw_data = pickling.unpickle_object(pickles_path)
    return raw_data
