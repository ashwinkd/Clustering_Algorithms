import random
from itertools import combinations

import numpy as np

np.random.seed(0)


def get_data(num_examples=500):
    fptr = open('cluster_data.csv', 'r')
    data = []
    for line in fptr.readlines():
        line = line.strip().split(',')
        data.append([int(line[0]), int(line[1])])

    data = random.sample(data, num_examples)
    data = np.array(data)
    return data


def normalize(data):
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    data = (data - _mean) / _std
    return data


def create_distance_matrix(data):
    num_examples = data.shape[0]
    distance_matrix = np.zeros((num_examples, num_examples))
    for i, d1 in enumerate(data):
        for j, d2 in enumerate(data):
            distance_matrix[i][j] = np.linalg.norm(d1 - d2)
    return distance_matrix


def flatten(list_object, dtype=int):
    _list = []
    for elem in list_object:
        if isinstance(elem, list):
            elem = flatten(elem)
            _list += elem
        elif isinstance(elem, dtype):
            _list.append(elem)
    return _list


def distance_between_cluster(cluster_pair, distance_matrix, type_of_clustering):
    cluster1 = flatten(cluster_pair[0])
    cluster2 = flatten(cluster_pair[1])
    cross_cluster_distances = []
    for i in cluster1:
        for j in cluster2:
            cross_cluster_distances.append(distance_matrix[i][j])
    if type_of_clustering == 'single':
        return min(cross_cluster_distances)
    elif type_of_clustering == 'complete':
        return max(cross_cluster_distances)
    elif type_of_clustering == 'average':
        return np.mean(cross_cluster_distances)


def hierarchical_cluster(distance_matrix, type_of_clustering='single'):
    if type_of_clustering not in ['single', 'complete', 'average']:
        return []
    clusters = [[i] for i in range(len(distance_matrix))]
    while len(clusters) > 2:
        cluster_joins = [list(pair) for pair in combinations(clusters, 2)]
        cluster_distances = [distance_between_cluster(pair, distance_matrix, type_of_clustering)
                             for pair in cluster_joins]
        closest_pair_index = np.argmin(cluster_distances)
        closest_pair = cluster_joins[closest_pair_index]
        clusters = [cluster for cluster in clusters if cluster not in closest_pair]
        clusters.append(closest_pair)

    return clusters


def main():
    # data = get_data(10)
    # data = normalize(data)

    data = np.array([[1.5, 2.0],
                     [3.0, 1.0],
                     [3.5, 2.5],
                     [1.0, 0.5],
                     [2.5, 2.0]])

    distance_matrix = create_distance_matrix(data)
    clusters = hierarchical_cluster(distance_matrix, 'complete')
    print(clusters)


if __name__ == '__main__':
    main()
