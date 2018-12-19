import json
import os
import pandas as pd

from pickles import pickling

DATA_DIRECTORY = 'data/'
PUBS_TRAIN_PATH = DATA_DIRECTORY + 'pubs_train.json'
ASSIGNMENT_TRAIN_PATH = DATA_DIRECTORY + 'assignment_train.json'
PUBS_VALIDATE_PATH = DATA_DIRECTORY + 'pubs_validate.json'


def load_pubs_train(refresh_pickle=False):
    if (not os.path.exists(pickling.PUBS_TRAIN_DF)) or refresh_pickle:
        with open(PUBS_TRAIN_PATH, 'rb') as infile:
            raw_data = json.load(infile)
            df = pd.DataFrame.from_dict(raw_data)
            pickling.pickle_object(df, pickling.PUBS_TRAIN_DF)
    else:
        df = pickling.unpickle_object(pickling.PUBS_TRAIN_DF)
    return df


def load_assignment_train(refresh_pickle=False):
    if (not os.path.exists(pickling.ASSIGNMENT_TRAIN_DF)) or refresh_pickle:
        with open(ASSIGNMENT_TRAIN_PATH, 'rb') as infile:
            raw_data = json.load(infile)
            df = pd.DataFrame.from_dict(raw_data)
            pickling.pickle_object(df, pickling.ASSIGNMENT_TRAIN_DF)
    else:
        df = pickling.unpickle_object(pickling.ASSIGNMENT_TRAIN_DF)
    return df
