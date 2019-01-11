import pickle

PUBS_TRAIN_DF = "pickles/train/pubs_train.p"
ASSIGNMENT_TRAIN = "pickles/train/assignment_train.p"
PUBS_VALIDATE_DF = "pickles/validate/pubs_validate.p"
ASSIGNMENT_VALIDATE = "pickles/validate/assignment_validate.p"
PUBS_TEST_DF = "pickles/test/pubs_test.p"


def pickle_object(obj, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile)


def unpickle_object(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)
