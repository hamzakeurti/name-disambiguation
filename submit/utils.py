import numpy as np


def single_name_cluster(pub_ids, cluster_labels, name):
    d = {name: []}
    for i in np.unique(cluster_labels):
        if i == -1:
            continue
        d[name].append(pub_ids[cluster_labels == i].tolist())
    d[name] += pub_ids[cluster_labels == -1].reshape([-1, 1]).tolist()
    return d
