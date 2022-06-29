import networkx as nx
import plotly.graph_objects as go
import pandas as pd


def plot_objects(links: list, number_of_partie):
    G = nx.Graph()
    color_step = int(100 / (number_of_partie + 1)) / 100
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    color = 0
    pos = {}
    dist_dict = {}
    color_map = []
    weight_map = []
    # As before we use networkx to determine node positions. We want to do the same spring layout but in 3D
    spring_3D = nx.spring_layout(G, dim=3, seed=18)

    for link in links:
        pos[link.v1.label] = link.v1.get_center()[0: 2]
        pos[link.v2.label] = link.v2.get_center()[0: 2]
        weight_map.append((link.v1.label, link.v2.label, link.meta['acc']))


if __name__ == '__main__':
    plot_objects([], 2)
