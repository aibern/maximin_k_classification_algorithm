import string
from statistics import median
from typing import Optional

import numpy as np
from networkx import Graph

from binary_classifiers import SVMBinaryClassifier, SVMRbfBinaryClassifier
from classifiers import SVMMultiClassifier
from dist_util import NormalDistributor, ESet, Link, LinkClassifier, BaseBinaryClassifier
from main import generate_distribution, partie_grouper

import plotly.graph_objects as go

import argparse
import json

import random

from datetime import datetime


def map_to_nx_weights(links: list) -> list:
    return [(link.v1.label, link.v2.label, link.meta['acc']) for link in links]


def extract_nodes_from(links: list) -> dict:
    nodes = {}
    for link in links:
        nodes[link.v1.label] = link.v1.get_center()
        nodes[link.v2.label] = link.v2.get_center()
    return nodes

def extract_sets_from(links: list) -> dict:
    sets = {}
    for link in links:
        sets[link.v1.label] = link.v1
        sets[link.v2.label] = link.v2
    return sets


class Nx3dLayout:

    def __init__(self, nx3d: dict) -> None:
        self.nx3d = nx3d

    def get_xyz_link_for_nodes(self, n0: str, n1: str):
        x_link = [self.nx3d[n0][0], self.nx3d[n1][0]]
        y_link = [self.nx3d[n0][1], self.nx3d[n1][1]]
        z_link = [self.nx3d[n0][2], self.nx3d[n1][2]]
        return x_link, y_link, z_link

    def extract_xyz_from_nx3d(self):
        # we need to separate the X,Y,Z coordinates for Plotly
        x_nodes = [self.nx3d[key][0] for key in self.nx3d.keys()]  # x-coordinates of nodes
        y_nodes = [self.nx3d[key][1] for key in self.nx3d.keys()]  # y-coordinates
        z_nodes = [self.nx3d[key][2] for key in self.nx3d.keys()]  # z-coordinates
        return x_nodes, y_nodes, z_nodes


def extract_xyz_from_nx3d(nx3d_layout: dict):
    # we need to separate the X,Y,Z coordinates for Plotly
    x_nodes = [nx3d_layout[key][0] for key in nx3d_layout.keys()]  # x-coordinates of nodes
    y_nodes = [nx3d_layout[key][1] for key in nx3d_layout.keys()]  # y-coordinates
    z_nodes = [nx3d_layout[key][2] for key in nx3d_layout.keys()]  # z-coordinates
    return x_nodes, y_nodes, z_nodes


def default_nx_layout(title: str = None):
    axis = dict(showbackground=True,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    return go.Layout(title=title or "Generated points in 3D",
                          width=650,
                          height=625,
                          showlegend=False,
                          scene=dict(xaxis=dict(axis),
                                     yaxis=dict(axis),
                                     zaxis=dict(axis),
                                     ),
                          margin=dict(t=100),
                          hovermode='closest')

def get_traces_from(links):
    traces = {}
    for link in links:
        trace_name = link.default_title()
        traces[trace_name] = go.Scatter3d(x=link.get_ax(0), y=link.get_ax(1), z=link.get_ax(2),
                                          mode='lines',
                                          line=dict(color='black', width=10*max(0, link.meta['acc'] - 0.91) + 3*max(0, link.meta['acc'] - 0.5) + 0.1),
                                          hoverinfo='all')
    return traces


def build_multi_classifier(links: list, clique: list) -> SVMMultiClassifier:
    clique_links = [link for link in links if link.v1.label in clique and link.v2.label in clique]
    clique_nodes = extract_sets_from(clique_links)
    classifier = SVMMultiClassifier()
    classifier.build_for(list(clique_nodes.values()))
    return classifier

def plot_clique(links: list, clique: list):
    """
    draw all the centers
    :param links:
    :param clique:
    :return:
    """
    # build clique links
    # filter links by checking if the node name belongs to clique list
    clique_links = [link for link in links if link.v1.label in clique and link.v2.label in clique]
    traces = get_traces_from(clique_links)
    nx_layout = default_nx_layout(title="Clique edges")
    data = list(traces.values())
    fig = go.Figure(data=data, layout=nx_layout)
    fig.show()


def plot_objects_3d(links: list, clique: list = None):
    G: Graph = nx.Graph()
    color_list = ['blue', 'yellow', 'purple', 'red',  'orange', 'y', 'k', 'w']
    weighted_edges: list = map_to_nx_weights(links)

    G.add_weighted_edges_from(weighted_edges)

    traces = get_traces_from(links)

    nodes = {}
    for link in links:
        nodes[link.v1.label] = link.v1
        nodes[link.v2.label] = link.v2

    def prepare_trace_nodes(_nodes):
        nodes_list = list(_nodes.values())
        centers = [node.get_center() for node in nodes_list]
        x_nodes = [c[0] for c in centers]
        y_nodes = [c[1] for c in centers]
        z_nodes = [c[2] for c in centers]
        color = [color_list[node.c] for node in nodes_list]
        trace_nodes = go.Scatter3d(x=x_nodes,
                                   y=y_nodes,
                                   z=z_nodes,
                                   mode='markers',
                                   marker=dict(symbol='circle',
                                               size=7,
                                               color=color,  # color the nodes according to their community
                                               # colorscale=['lightgreen', 'magenta'],  # either green or mageneta
                                               line=dict(color='blue', width=0.5)),
                                   text=[],
                                   hoverinfo='text')
        return trace_nodes

    trace_samples = {}

    for node in nodes.values():
        color = color_list[node.c]
        trace_samples[node.label] = go.Scatter3d(x=node.data[:, 0], y=node.data[:, 1], z=node.data[:, 2],
                                                 mode='markers',
                                                 marker=dict(symbol='circle', size=2,
                                                             color=color,  # color the nodes according to their community
                                                             colorscale=['lightgreen', 'magenta'],  # either green or mageneta
                                                            line=dict(color='blue', width=0.1)), text=[], hoverinfo='text')
    if clique is not None:
        clique_nodes = {name: node for name, node in nodes.items() if name in clique}
        print(clique_nodes)

    # we need to set the axis for the plot
    axis = dict(showbackground=True,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    nx_layout = go.Layout(title="Generated center points in 3D",
                          width=650,
                          height=625,
                          showlegend=False,
                          scene=dict(xaxis=dict(axis),
                                     yaxis=dict(axis),
                                     zaxis=dict(axis),
                                     ),
                          margin=dict(t=100),
                          hovermode='closest')

    # Include the traces we want to plot and create a figure
    data = [prepare_trace_nodes(nodes)]
    data += list(trace_samples.values())
    fig = go.Figure(data=data, layout=nx_layout)
    fig.show()

    edges_layout = go.Layout(title="Data centers connected with edges marked with accuracy value",
                          width=650,
                          height=625,
                          showlegend=False,
                          scene=dict(xaxis=dict(axis),
                                     yaxis=dict(axis),
                                     zaxis=dict(axis),
                                     ),
                          margin=dict(t=100),
                          hovermode='closest')
    data = list(traces.values())
    fig = go.Figure(data=data, layout=edges_layout)
    fig.show()


def gen_chunk_sets(chunk, class_idx):
    chunk_sets = []
    sets_counter = 0
    for item in chunk:
        distributor = NormalDistributor(mean=item[0], cov=item[1][1], size=3)
        samples_size = item[1][2]
        data = distributor.generate(samples_size)
        test_data = distributor.generate(samples_size)
        _set = ESet(data=data, test_data=test_data, label=f'c_{class_idx}_n_{sets_counter}', c=class_idx).center(v=item[0])
        chunk_sets.append(_set)
        sets_counter += 1
    return chunk_sets


def extract_ser_links_from(links: dict) -> dict:
    ser_links: dict = {}
    for name, link in links.items():
        ser_links[name] = link.to_json()
    return ser_links


def extract_points_from(links: dict) -> dict:
    points: dict = {}
    for link in links.values():
        points[link.v1.label] = link.v1.to_json()
        points[link.v2.label] = link.v2.to_json()
    return points


class AGraph:

    def __init__(self, links: dict) -> None:
        super().__init__()
        self.links: dict = links

    def apply_accuracy_threshold(self, threshold: float):
        filter_links = {}
        for k, v in self.links.items():
            if v.meta['acc'] >= threshold:
                filter_links[k] = v
        self.links = filter_links
        return self

    def with_accuracy_threshold(self, threshold: float):
        filter_links = {}
        for k, v in self.links.items():
            if v.meta['acc'] >= threshold:
                filter_links[k] = v
        return AGraph(filter_links)

    def find_cliques_with_acc(self, acc: float):
        subgraph = self.with_accuracy_threshold(threshold=acc)
        return nx.find_cliques(subgraph.build_nx_graph())

    def build_nx_graph(self):
        nx_graph: Graph = nx.Graph()
        nx_graph.add_weighted_edges_from(map_to_nx_weights(list(self.links.values())))
        return nx_graph


class CGraph:

    def __init__(self, initial_links: dict, classes: dict = None) -> None:
        super().__init__()
        self.links: dict = initial_links
        self.classes: dict = classes
        self.parts = {}

    def accept(self, link):
        pass

    def init(self):
        for k, v in self.classes.items():
            self.parts[k] = {}
            for item in v:
                self.parts[k][item.label] = {'degree': 0}
        for name, link in self.links.items():
            c1: int = link.v1.c
            c2: int = link.v2.c
            self.parts[c1][link.v1.label]['degree'] += 1
            self.parts[c2][link.v2.label]['degree'] += 1

    def has_clique(self) -> bool:
        nx_graph: Graph = nx.Graph()
        nx_graph.add_weighted_edges_from(map_to_nx_weights(list(self.links.values())))
        v = nx.find_cliques(nx_graph)
        for item in v:
            print(item)
        return False


def find_cliques(links: dict, acc: float):
    a_graph = AGraph(links=links)\
        .with_accuracy_threshold(threshold=acc)
    return nx.find_cliques(a_graph.build_nx_graph())


class TaskParams:
    pass

class TaskHandler:

    def __init__(self, link_classifier: LinkClassifier, moments_gen_fun, card_set_of_sets: int,
                 num_of_classes: int, classes_group_fun) -> None:
        super().__init__()
        self.link_classifier = link_classifier
        self.moments_gen_fun = moments_gen_fun
        self.card_set_of_sets = card_set_of_sets
        self.num_of_classes: int = num_of_classes
        self.classes_group_fun = classes_group_fun

        self.dist_list: Optional[np.ndarray] = None
        self.mu_list: Optional[np.ndarray] = None
        self.classes: Optional[dict] = None
        self.links: Optional[dict] = None
        self.chunked_params: Optional[list] = None
        self.links_data: Optional[dict] = None

    def process(self, result_file: str, do_plot: bool = False):
        clique, min_part_accuracy = self.td_linear_find_high_accuracy_clique(parts_size=len(self.classes), step=0.01)
        clique.sort()
        links = list(self.links.values())
        classifier = build_multi_classifier(links=links, clique=clique)
        if do_plot:
            plot_clique(links, clique)
        result: dict = {"clique": clique, "k_acc": classifier.acc, "bin_acc": min_part_accuracy, "id": ".".join(clique)}
        print("result: ", result)
        self.save_json(result, result_file)

        if do_plot:
            plot_objects_3d(links=list(self.links.values()), clique=clique)

    def get_max_cliques(self, min_acc: float = 0):
        a_graph = AGraph(links=self.links)
        for clique in a_graph.find_cliques_with_acc(acc=min_acc):
            if len(clique) == self.num_of_classes:
                yield clique

    def build_multi_classifier_for(self, clique: list) -> SVMMultiClassifier:
        clique_links = [link for link in self.links.values() if link.v1.label in clique and link.v2.label in clique]
        clique_nodes = extract_sets_from(clique_links)
        classifier = SVMMultiClassifier()
        classifier.build_for(list(clique_nodes.values()))
        return classifier

    def count_sum_binary_accuracy(self, clique: list) -> float:
        return sum([v.meta['acc'] for v in self.get_clique_links(clique).values()])

    def td_linear_find_high_accuracy_clique(self, parts_size: int, min_acc: float = 0.5, step: float = 0.05):
        a_graph = AGraph(links=self.links)
        current_clique = None
        current_acc: float = 1 - step
        while current_acc >= min_acc:
            cliques = a_graph.find_cliques_with_acc(acc=current_acc)
            max_cliques = [clique for clique in cliques if len(clique) == parts_size]
            if len(max_cliques) == 0:
                current_acc -= step
            else:
                current_clique = max_cliques[0]
                return current_clique, current_acc
        return current_clique, current_acc

    def init_classes(self) -> None:
        self.dist_list, self.mu_list = self.moments_gen_fun()
        self.chunked_params: list = self.classes_group_fun(dist_list=self.dist_list, mu_list=self.mu_list,
                                                           number_of_partie=self.num_of_classes,
                                                           card_set_of_sets=self.card_set_of_sets)
        self.classes = self.create_chunk_classes(self.chunked_params)

    def init_classify_links(self, _file_path: Optional[str]):
        self.init_classes()
        # create list of pairs of sets to compare (to build binary classifier)
        self.links: dict = self.gen_cross_classes_links(self.classes, self.num_of_classes)
        self.count_accuracy_for(self.links, self.link_classifier)
        if _file_path:

            self.save_links(_file_path)

    def handle(self, file_path: str = 'data.json', result_path: str = "r1.json", do_plot: bool = False):
        self.init_classify_links(_file_path=file_path)
        self.process(result_file=result_path, do_plot=do_plot)

    def save_links(self, file_path: str = 'data.json'):
        self.links_data: dict = self.pre_save_data(self.links)
        self.save_json(self.links_data, file_path)

    @staticmethod
    def save_json(data: dict, file_path: str):
        file = open(file_path, 'w')
        file.write(json.dumps(data))
        file.close()

    def count_accuracy_for(self, links: dict, link_classifier: LinkClassifier):
        for link in links.values():
            link.meta['acc'] = link_classifier.apply(link)
            link.meta['points'] = [link.v1.get_center(), link.v2.get_center()]

    def pre_save_data(self, links: dict) -> dict:
        return {'points': extract_points_from(links), 'links': extract_ser_links_from(links)}

    def create_chunk_classes(self, chunked_params: list):
        classes = {}
        class_idx = 0
        for chunk in chunked_params:
            chunk_sets = gen_chunk_sets(chunk, class_idx)
            classes[class_idx] = chunk_sets
            class_idx += 1
        return classes

    def init_cross_classes_links(self) -> dict:
        return self.gen_cross_classes_links(classes=self.classes, num_of_classes=self.num_of_classes)

    def gen_cross_classes_links(self, classes: dict, num_of_classes: int) -> dict:
        # create list of pairs of sets to compare (to build binary classifier)
        self.links = {}
        for i in range(0, num_of_classes - 1):
            for j in range(i + 1, num_of_classes):
                for set_i in classes[i]:
                    for set_j in classes[j]:
                        link_ij = Link(v1=set_i, v2=set_j)
                        link_ij.meta['c1'] = i
                        link_ij.meta['c2'] = j
                        link_name = link_ij.default_title()
                        self.links[link_name] = link_ij
        return self.links

    def get_clique_links(self, clique: list) -> dict:
        return {name: link for name, link in self.links.items() if link.v1.label in clique and link.v2.label in clique}


def prepare_draw_save_data():
    link_classifier = LinkClassifier(classifier=SVMBinaryClassifier())
    card_set_of_sets = 15
    num_of_classes = 5
    number_list = list(range(1, card_set_of_sets + 1))

    def moments_gen_fun():
        return generate_distribution(d_size=3, number_list=number_list,
                                     mu_limits=(-2, 2), sigma_limits=(0, 1), N=30)

    task = TaskHandler(link_classifier=link_classifier, moments_gen_fun=moments_gen_fun,
                       card_set_of_sets=card_set_of_sets, num_of_classes=num_of_classes,
                       classes_group_fun=partie_grouper)
    task.handle(do_plot=True)


classifiers: dict = {
    "linear": SVMRbfBinaryClassifier,
    "rbf": SVMRbfBinaryClassifier
}


def classifier_by_type(cl_type: str) -> BaseBinaryClassifier:
    return classifiers[cl_type]()


def run_svm_rbf_experiment(params: dict):
    link_classifier = LinkClassifier(classifier=classifier_by_type(params["svm_type"]))
    card_set_of_sets = params["card_set_of_sets"]
    num_of_classes = params["num_classes"]
    number_list = list(range(1, card_set_of_sets + 1))
    mu_limits = (params["mu_limits"][0], params["mu_limits"][1])
    sigma_limits = (params["sigma_limits"][0], params["sigma_limits"][1])
    N = params["N"]

    def moments_gen_fun():
        return generate_distribution(d_size=3, number_list=number_list,
                                     mu_limits=mu_limits, sigma_limits=sigma_limits, N=N)

    task = TaskHandler(link_classifier=link_classifier, moments_gen_fun=moments_gen_fun,
                       card_set_of_sets=card_set_of_sets, num_of_classes=num_of_classes,
                       classes_group_fun=partie_grouper)
    task.init_classes()
    task.init_cross_classes_links()

    task.count_accuracy_for(task.links, link_classifier)
    result = {}

    clique_results = {}

    clique_counter = 0
    max_sum_binary_acc = 0
    max_sum_clique = []

    print("Start task.process", datetime.now().time())
    task.process(result_file=params["result_file"], do_plot=False)
    print("End task.process", datetime.now().time())

    for clique in task.get_max_cliques():
        clique.sort()
        sum_binary_acc = task.count_sum_binary_accuracy(clique)
        if sum_binary_acc > max_sum_binary_acc:
            max_sum_binary_acc = sum_binary_acc
            max_sum_clique = clique
        if clique_counter % 10000 == 0:
            mclf = build_multi_classifier(links=list(task.links.values()), clique=max_sum_clique)
            print(clique_counter, max_sum_clique, max_sum_binary_acc, "value: " + str(mclf.acc))
        clique_counter += 1
    mclf = build_multi_classifier(links=list(task.links.values()), clique=max_sum_clique)

    calc_all_cliques: bool = False
    if calc_all_cliques:
        for clique in task.get_max_cliques():
            clique.sort()
            mclf = build_multi_classifier(links=list(task.links.values()), clique=clique)
            clique_results['.'.join(clique)] = {'k_acc': mclf.acc}
        max_k_acc: float = max([v['k_acc'] for v in clique_results.values()])
        max_accuracy_clique = {k: v for k, v in clique_results.items() if v['k_acc'] == max_k_acc}
        result['ma_clique'] = max_accuracy_clique
        print('ma_clique', max_accuracy_clique)
        print("Minimum accuracy", min(list(v['k_acc'] for v in clique_results.values())))
        print("Median accuracy", median(list(v['k_acc'] for v in clique_results.values())))
    result['clique_results'] = clique_results


    random_id = "".join(random.choices(string.ascii_lowercase, k=8))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Graph generation and processing.')
    parser.add_argument('--num_classes', help='number of classes', type=int)
    parser.add_argument('-N', help='number of points', type=int)
    parser.add_argument('-T', help='SVM kernel type', type=str)
    r = parser.parse_args()
    print(r.N)


if __name__ == '__main__':
    print("start", datetime.now().time())
    file_path = "tasks/2.json"
    task: Optional[dict] = None
    with open(file_path) as f:
        task = json.load(f)
    if task:
        print("Task parameters", task)
        run_svm_rbf_experiment(params=task['params'])
        print("end", datetime.now().time())
