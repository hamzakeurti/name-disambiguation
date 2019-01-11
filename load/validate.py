import json
import os
import pandas as pd

from pickles import pickling
from load import loader

DATA_DIRECTORY = 'data/'
PUBS_VALIDATE_PATH = DATA_DIRECTORY + 'pubs_validate.json'
ASSIGNMENT_VALIDATE_PATH = DATA_DIRECTORY + 'assignment_validate.json'


def load_df_pubs(refresh_pickle=False):
    return loader.load_df_pubs(json_path=PUBS_VALIDATE_PATH, pickles_path=pickling.PUBS_VALIDATE_DF,
                               refresh_pickle=refresh_pickle)


def load_assignment(refresh_pickle=False):
    return loader.load_assignment(json_path=ASSIGNMENT_VALIDATE_PATH, pickles_path=pickling.ASSIGNMENT_VALIDATE,
                                  refresh_pickle=refresh_pickle)
