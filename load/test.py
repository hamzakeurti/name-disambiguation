from pickles import pickling
from load import loader

DATA_DIRECTORY = 'data/'
PUBS_TEST_PATH = DATA_DIRECTORY + 'pubs_train.json'


def load_df_pubs(refresh_pickle=False):
    return loader.load_df_pubs(json_path=PUBS_TEST_PATH, pickles_path=pickling.PUBS_TEST_DF,
                               refresh_pickle=refresh_pickle)
