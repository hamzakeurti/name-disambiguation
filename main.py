import pandas as pd
import numpy as np
import scipy as sp
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN

from load import train,validate,test
from submit import utils
from features import extract


submission_path = 'data/submission/tresult.json'



pubs_test = test.load_df_pubs()
sub = {}
names_test = list(pubs_test.keys())
for name0 in names_test:
#     if name0 != names_test[0]:
#         continue
    print(name0)
    single_name_pubs = pubs_test[name0]
    
    dict_features_weights = {
        extract.FEATURE_AUTHORS:1, #eps= 0.6 0.64
        
#         extract.FEATURE_KEYWORDS:1, #eps = 0.8 really bad
        extract.FEATURE_TITLE:1,
        extract.FEATURE_YEAR:1,
        extract.FEATURE_VENUE:1,
        extract.FEATURE_ORG:1
        
    }
    
    features =  extract.get_weighted_features(single_name_df=single_name_pubs,features_dict=dict_features_weights)
    
    db = DBSCAN(eps=0.8,metric='cosine',min_samples=2).fit(features)
    
    single_name_sub = utils.single_name_cluster(cluster_labels=db.labels_,name=name0,pub_ids=single_name_pubs.id.values)
    
    sub.update(single_name_sub)
    
    
with open(submission_path,'w') as f:
    json.dump(sub,f)