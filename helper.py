from typing import List

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N = 30
D_size = 3
mu_limits = (-20, 20)
sigma_limits = (0, 15)
card_set_of_sets = 15
number_of_partie = 5

number_list = list(range(1, card_set_of_sets+1))


def distribution_creator(number_list, mu_limits, sigma_limits, N):
    dist_list = list(map(lambda x: [random.uniform(mu_limits[0], mu_limits[1]), \
                                    random.uniform(sigma_limits[0], sigma_limits[1]), \
                                N], number_list))
    dist_list_ = list(map(lambda x: np.random.normal(x[0], x[1], x[2]), dist_list))
    return dist_list_


def generate_vector(d_size, input_min, input_max):
    list_ = list(range(d_size))
    return list(map(lambda x: random.uniform(input_min, input_max), list_))


def generate_cov_matrix(D_size, input_min, input_max):
    cov_matrix = np.eye(D_size, dtype=int)
    sigma = random.uniform(input_min, input_max)
    cov_matrix = cov_matrix * sigma
    return cov_matrix


def generate_distribution(d_size, number_list, mu_limits, sigma_limits, N):
    """
    :param d_size: dimension size
    :param number_list: # fixme: number_list needs name clarification (and why don't pass the upper bound of the range?)
    :param mu_limits: array of min and max mean values
    :param sigma_limits:  array of min and max sigma values (positive sigma is considered)
    :param N: number of points to be generated
    :return:
    """
    # random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
    _dist_list = np.array(list(map(lambda x: [generate_vector(d_size, input_min=mu_limits[0], input_max=mu_limits[1]), \
                                              generate_cov_matrix(d_size, input_min=sigma_limits[0], input_max=sigma_limits[1]), \
                                              N], number_list)))
    _mu_list = _dist_list[:, 0]
    return _dist_list, _mu_list


def partie_grouper(dist_list, mu_list, number_of_partie, card_set_of_sets):
    """
    split params over partitions
    todo: consider generating params for partitions
    :param dist_list:
    :param mu_list:
    :param number_of_partie:
    :param card_set_of_sets:
    :return:
    """
    step = int(card_set_of_sets/number_of_partie)
    partitions = []
    i = 0
    while i+step <= card_set_of_sets:
        grouped_step = [(mu_list[j], dist_list[j]) for j in range(i, i+step)]
        partitions.append(grouped_step)
        i += step
    if i < card_set_of_sets < i+step:
        arr_len = min(len(dist_list), len(mu_list))
        grouped_step = [(mu_list[j], dist_list[j]) for j in range(i, arr_len)]
        partitions.append(grouped_step)
    return partitions


def distributions_plotter(pertie_dist_list, pertie_mu_list, card_set_of_sets, number_of_partie, projection='3d',
                          figsize=(8, 8)):

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    color_step = int(100/(number_of_partie+1))/100

    color = 0
    for j in list(range(number_of_partie)):
        color += color_step
        one_pertie_dist = pertie_dist_list[j]
        one_pertie_color = [[color, 1-color, np.random.uniform(0, 1)]]
        for i in list(range(len(one_pertie_dist))):

            X = np.random.multivariate_normal(one_pertie_dist[i][0], one_pertie_dist[i][1], one_pertie_dist[i][2]).T
            ax.scatter(X[0], X[1], X[2], c=one_pertie_color, alpha=0.3)  # s=10,  c=color_list[j])
            ax.scatter(pertie_mu_list[j][i][0], pertie_mu_list[j][i][1], pertie_mu_list[j][i][2], c=one_pertie_color,\
                       alpha=1, s=50, edgecolors='black')
    plt.show()


def distribution_saver(pertie_dist_list, pertie_mu_list, card_set_of_sets, number_of_partie, distribution_type = 'normal'):
    distributions_list = []
    color_list = []
    G = nx.Graph()
    color_step = int(100/(number_of_partie+1))/100
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # print('color_step', color_step)
    # print('number_of_partie', number_of_partie, len(pertie_dist_list))
    color = 0
    pos = {}
    dist_dict = {}
    color_map = []
    weight_map = []
    for j in list(range(number_of_partie)):
        color += color_step
        one_pertie_color = [[color, 1-color, np.random.uniform(0, 1)]]
        color_list.append(one_pertie_color)
        one_partie_dist_features = pertie_dist_list[j] # one_pertie_mu = pertie_mu_list[j]

        one_partie_mu_list = pertie_mu_list[j]

        for i in list(range(len(one_partie_dist_features))):
            if distribution_type == 'normal':
                X = np.random.multivariate_normal(one_partie_dist_features[i][0], \
                                              one_partie_dist_features[i][1], one_partie_dist_features[i][2]).T
                node = 'P'+str(j) +'_n'+str(i)
                node_coordinates = one_partie_mu_list[i]
                G.add_node(node) #, color=one_pertie_color[0])
                pos[node] = tuple(node_coordinates[:2]) # pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}
                color_map.append(one_pertie_color[0])

                # G.add_edge(1, 2, color='r', weight=3)
                dist_dict[node] = X

    parties_numbers = list(range(number_of_partie))
    for j in parties_numbers:
        print('j', j)
        for k in parties_numbers:
            if j != k:
                for m in list(range(len(pertie_dist_list[j]))):
                    for n in list(range(len(pertie_dist_list[k]))):
                        node_j = 'P'+str(j) +'_n'+str(m)
                        node_k = 'P' + str(k) + '_n' + str(n)
                        data_j = dist_dict[node_j]
                        data_k = dist_dict[node_k]
                        data_X = []
                        data_X.append(data_j)
                        data_X.append(data_k)
                        X = [[0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
                        y = [0, 1, 1, 1]
                        clf = svm.SVC()
                        print(clf.fit(X, y))
                        y_predict = clf.predict([[2, 2, 2]])
                        print(y_predict)
                        weight_map.append((node_j, node_k, int(j+k)))



    G.add_weighted_edges_from(weight_map)
    weights = nx.get_edge_attributes(G, 'weight').values()
    nx.draw(G, pos=pos, node_color=color_map, with_labels=True, width=list(weights), alpha=0.3)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    return


if __name__ == "__main__":
    dist_list, mu_list = generate_distribution(D_size, number_list, mu_limits, sigma_limits, N)
    chunked_params: list = partie_grouper(dist_list, mu_list, number_of_partie, card_set_of_sets)