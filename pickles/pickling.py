import pickle

PUBS_TRAIN_DF = "pickles/train/pubs_train.p"
ASSIGNMENT_TRAIN_DF = "pickles/train/assignment_train.p"
PUBS_VALIDATE_DF = "pickles/validate/pubs_validate.p"


def pickle_object(obj, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile)


def unpickle_object(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)
