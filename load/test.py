import json
import os
import pandas as pd

from pickles import pickling

DATA_DIRECTORY = 'data/'
PUBS_VALIDATE_PATH = DATA_DIRECTORY + 'pubs_validate.json'


def load_pubs_validate(refresh_pickle=False):
    if (not os.path.exists(pickling.PUBS_VALIDATE_DF)) or refresh_pickle:
        with open(PUBS_VALIDATE_PATH, 'rb') as infile:
            raw_data = json.load(infile)
            df = pd.DataFrame.from_dict(raw_data)
            pickling.pickle_object(df, pickling.PUBS_VALIDATE_DF)
    else:
        df = pickling.unpickle_object(pickling.PUBS_VALIDATE_DF)
    return df
