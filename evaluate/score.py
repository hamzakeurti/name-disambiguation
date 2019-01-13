
# coding: utf-8

# In[9]:

import json
from zipfile import *
from os import listdir
from os.path import isfile, join
from itertools import combinations
import datetime
import time

def build_truth_pairs(truth):
    paper_set = set([])
    truth_pair = {}
    for item in truth:
        truth_pair[item] = []
        for i in range(len(truth[item])):
            truth_pair[item] += [set(p) for p in combinations(truth[item][i], 2)]
            for p in truth[item][i]:
                    paper_set.add(p)
    return truth_pair,paper_set


def build_truth_pairs_single_name(truth,author):
    tp = 0
    paper_set = set([])
    truth_pair = {}
    truth_pair[author] = []
    for i in range(len(truth[author])):
        set_list = [set(p) for p in combinations(truth[author][i], 2)]
        truth_pair[author] += set_list
        tp += len(set_list)
        for p in truth[author][i]:
                paper_set.add(p)
    return truth_pair,paper_set,tp

def build_sub_pairs_single_name(sub,author,paper_set):
    sub_pair = {}
    c = 0
    sub_pair[author] = []
    for i in range(len(sub[author])):
        set_list = [set(p) for p in combinations(sub[author][i], 2) if p[0] in paper_set and p[1] in paper_set]
        sub_pair[author] += set_list
        c += len(set_list)
    return sub_pair,c

def count_single_name(sub_pair,truth_pair):
    correct = 0
    for name in sub_pair:
        for item in sub_pair[name]:
            if item in truth_pair[name]:
                correct += 1
    return correct
    

def f_score_single_name(sub,truth,author):
    start = time.time()
    truth_pair,paper_set,tp = build_truth_pairs_single_name(truth=truth,author=author)
    t1 = time.time()
    print(f'Truth Pairs built in {t1-start}')
    
    start2 = time.time()
    sub_pair,c = build_sub_pairs_single_name(sub=sub,author=author,paper_set=paper_set)
    t2 = time.time()
    print(f'Sub Pairs built in {t2-start2}')
    
    start3 = time.time()
    correct = count_single_name(sub_pair,truth_pair)
    t3 = time.time()
    print(f'Counted correct pairs in {t3-start3}')


    p = correct / c if c != 0 else 0
    r = correct / tp
    if p + r == 0:
        score = 0
    else:
        score = 2 * p * r /(p + r)
    print(f'Precision : {p}')
    print(f'Recall : {r}')
    print(f'F-score : {score}')
    return score
            

