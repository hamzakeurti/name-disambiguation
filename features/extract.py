import numpy as np
import scipy as sp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer



FEATURE_ABSTRACT = 'abstract'
FEATURE_AUTHORS = 'authors'
FEATURE_KEYWORDS = 'keywords'
FEATURE_ORG = 'org'
FEATURE_TITLE = 'title'
FEATURE_VENUE = 'venue'
FEATURE_YEAR = 'year'



AUTHORS = 'authors'
NAME = 'name'
KEYWORDS = 'keywords'
ABSTRACT = 'abstract'
TITLE = 'title'
VENUE = 'venue'
ORG = 'org'

YEAR = 'year'
YEARm1 = 'year-1'
YEARm2 = 'year-2'
YEARp1 = 'year+1'
YEARp2 = 'year+2'



def get_authors_lists(single_name_df):
    return single_name_df[AUTHORS].apply(lambda x: np.array([elem[NAME] for elem in x])).values.copy()


def get_organizations_lists(single_name_df):
    return single_name_df[AUTHORS].apply(lambda x: np.array([elem[ORG] for elem in x])).values.copy()



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


def get_abstract_features(single_name_df):
    abstracts = single_name_df[ABSTRACT].values.copy().astype(str)
    vectorizer_abstracts = TfidfVectorizer(strip_accents='unicode',decode_error='ignore',stop_words='english')
    return vectorizer_abstracts.fit_transform(abstracts)


def get_title_features(single_name_df):
    titles = single_name_df[TITLE].values.copy().astype(str)
    vectorizer_titles = TfidfVectorizer(strip_accents='unicode',decode_error='ignore',stop_words='english')
    return vectorizer_titles.fit_transform(titles)


def get_venue_features(single_name_df):
    venues = single_name_df[VENUE].values.copy().astype(str)
    vectorizer_venues = TfidfVectorizer(strip_accents='unicode',decode_error='ignore',stop_words='english')
    return vectorizer_venues.fit_transform(venues)
 

def get_organization_features(single_name_df,sparse_output=True):
    orgs_list = get_organizations_lists(single_name_df)
    mlb_orgs = MultiLabelBinarizer(sparse_output=sparse_output)
    orgs_features = mlb_orgs.fit_transform(orgs_list)
    return TfidfTransformer().fit_transform(orgs_features)


def get_year_features(single_name_df,sparse_output=True):
    single_name_df[YEARm1] = single_name_df[YEAR]-1
    single_name_df[YEARm2] = single_name_df[YEAR]-2
    single_name_df[YEARp1] = single_name_df[YEAR]+1
    single_name_df[YEARp2] = single_name_df[YEAR]+2
    years_list = single_name_df[[YEARm2,YEARm1,YEAR,YEARp1,YEARp2]].values.copy()

    labels = np.arange(single_name_df[YEARm2].min(),single_name_df[YEARp2].max()+1)

    mlb_years = MultiLabelBinarizer(classes=labels,sparse_output=sparse_output)
    years_features = mlb_years.fit_transform(years_list)
    return TfidfTransformer().fit_transform(years_features)


def get_features(single_name_df,feature_name):
    if feature_name == FEATURE_AUTHORS:
        return get_authors_features(single_name_df)
    if feature_name == FEATURE_KEYWORDS:
        return get_keywords_features(single_name_df)
    if feature_name == FEATURE_TITLE:
        return get_title_features(single_name_df)
    if feature_name == FEATURE_YEAR:
        return get_year_features(single_name_df)
    if feature_name == FEATURE_VENUE:
        return get_venue_features(single_name_df)
    if feature_name == FEATURE_ORG:
        return get_organization_features(single_name_df)
    if feature_name == FEATURE_ABSTRACT:
        return get_abstract_features(single_name_df)
    return


def get_weighted_features(single_name_df,features_dict):
    features_matrices = []
    for feature in features_dict:
        weight = features_dict[feature]
        features_matrices.append((weight*get_features(single_name_df,feature)).toarray())
    features = sp.sparse.csr.csr_matrix(sp.concatenate(features_matrices,axis=1))
    return  TfidfTransformer().fit_transform(features)

