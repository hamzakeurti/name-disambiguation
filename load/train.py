from pickles import pickling
from load import loader

DATA_DIRECTORY = 'data/'
PUBS_TRAIN_PATH = DATA_DIRECTORY + 'pubs_train.json'
ASSIGNMENT_TRAIN_PATH = DATA_DIRECTORY + 'assignment_train.json'


def load_df_pubs(refresh_pickle=False):
    return loader.load_df_pubs(json_path=PUBS_TRAIN_PATH, pickles_path=pickling.PUBS_TRAIN_DF,
                               refresh_pickle=refresh_pickle)


def load_assignment(refresh_pickle=False):
    return loader.load_assignment(json_path=ASSIGNMENT_TRAIN_PATH, pickles_path=pickling.ASSIGNMENT_TRAIN,
                                  refresh_pickle=refresh_pickle)
