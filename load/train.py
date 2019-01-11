import json
import os
import pandas as pd

from pickles import pickling
from load import loader

DATA_DIRECTORY = 'data/'
PUBS_TRAIN_PATH = DATA_DIRECTORY + 'pubs_train.json'
ASSIGNMENT_TRAIN_PATH = DATA_DIRECTORY + 'assignment_train.json'


def load_df_pubs(refresh_pickle=False):
    return loader.load_df_pubs(json_path=PUBS_TRAIN_PATH,pickles_path=pickling.PUBS_TRAIN_DF,refresh_pickle=refresh_pickle)


def load_assignment(refresh_pickle=False):
    if (not os.path.exists(pickling.ASSIGNMENT_TRAIN_DF)) or refresh_pickle:
        with open(ASSIGNMENT_TRAIN_PATH, 'rb') as infile:
            raw_data = json.load(infile)
            df = pd.DataFrame.from_dict(raw_data)
            pickling.pickle_object(df, pickling.ASSIGNMENT_TRAIN_DF)
    else:
        df = pickling.unpickle_object(pickling.ASSIGNMENT_TRAIN_DF)
    return df
