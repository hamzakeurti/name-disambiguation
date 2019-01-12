import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer


AUTHORS = 'authors'
NAME = 'name'
KEYWORDS = 'keywords'


def get_authors_lists(single_name_df):
    return single_name_df[AUTHORS].apply(lambda x: np.array([elem[NAME] for elem in x])).values.copy()


def get_authors_features(single_name_df,sparse_output=True):
    authors_list = get_authors_lists(single_name_df)
    mlb_authors = MultiLabelBinarizer(sparse_output=sparse_output)
    authors_features = mlb_authors.fit_transform(authors_list)
    return TfidfTransformer().fit_transform(authors_features)


def get_keywords_features(single_name_df,sparse_output=True):
    keywords_list = single_name_df[KEYWORDS].values.copy()
    mlb_keywords = MultiLabelBinarizer(sparse_output=sparse_output)
    keywords_list = mlb_keywords.fit_transform(keywords_list)
    return TfidfTransformer().fit_transform(keywords_list)
