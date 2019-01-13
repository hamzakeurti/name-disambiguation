
# coding: utf-8

# In[9]:

import json
from zipfile import *
from os import listdir
from os.path import isfile, join
from itertools import combinations
import multiprocessing as mp
import datetime

output = mp.Queue()

# (TODO) Your file here, please change the path. #
sub_file = '............'

with open('assignment_validate.json') as f:
    truth = json.load(f)

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
        set_list = [set(p) for p in combinations(truth[item][i], 2)]
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
        for item in p[name]:
            if item in truth_pair[name]:
                correct += 1
    return correct
    

def f_score_single_name(sub,truth,author):
    truth_pair,paper_set,tp = build_truth_pairs_single_name(truth=truth,author=author)
    sub_pair,c = build_sub_pairs_single_name(sub=sub,author=author,paper_set=paper_set)
    correct = count_single_name(sub_pair,truth_pair)
    p = correct / c if c != 0 else 0
    r = correct / tp
    if p + r == 0:
        score = 0
    else:
        score = 2 * p * r /(p + r)
    return score
            

paper_set = set([])
truth_pair = {}
for item in truth:
    truth_pair[item] = []
    for i in range(len(truth[item])):
        truth_pair[item] += [set(p) for p in combinations(truth[item][i], 2)]
        for p in truth[item][i]:
                paper_set.add(p)


with open(sub_file) as f:
    sub = json.load(f)

sub_pair = {}
c = 0
for item in sub:
    sub_pair[item] = []
    for i in range(len(sub[item])):
        set_list = [set(p) for p in combinations(sub[item][i], 2) if p[0] in paper_set and p[1] in paper_set]
        sub_pair[item] += set_list
        c += len(set_list)

part = [{},{},{}]
for n, item in enumerate(sub_pair):
    if item == 't_suzuki':
        part[0][item] = sub_pair[item]
    elif item == 'q_liu':
        part[1][item] = sub_pair[item]
    else:
        part[2][item] = sub_pair[item]

len(part[2])

def count_part(p, output):
    correct = 0
    for name in p:
        for item in p[name]:
            if item in truth_pair[name]:
                correct += 1
    output.put(correct)
    return correct

processes = [mp.Process(target=count_part, args=(part[x], output)) for x in range(3)]

begin = datetime.datetime.now()
for p in processes:
    p.start()
    
for p in processes:
    p.join()
    
results = [output.get() for p in processes]
end = datetime.datetime.now()

correct = 0
for i in results:
    correct += i
    
p = correct / c if c != 0 else 0
r = correct / 424583

if p + r == 0:
    score = 0
else:
    score = 2 * p * r /(p + r)

print('score', score)





