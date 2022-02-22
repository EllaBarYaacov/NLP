from nltk.corpus import dependency_treebank
import os
import pickle
import numpy as np
from nltk import DependencyGraph
from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence
# from collections import namedtuple


def min_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = minimum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result

WORD_KEY = 'word'
POS_KEY = 'tag'
WORD_PKL_PATH = "word_dict.pkl"
POS_PKL_PATH = "pos_dict.pkl"


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_dicts():
    word_i = 0
    word_dict = dict()
    pos_i = 0
    pos_dict = dict()
    sents = dependency_treebank.parsed_sents()

    for sent in sents:
        for node in sent.nodes.values():
            if node[WORD_KEY] not in word_dict:
                word_dict[node[WORD_KEY]] = word_i
                word_i += 1
            if node[POS_KEY] not in pos_dict:
                pos_dict[node[POS_KEY]] = pos_i
                pos_i += 1
    return word_dict, pos_dict

def create_or_load_dicts():
    if not os.path.exists(WORD_PKL_PATH) or not os.path.exists(POS_PKL_PATH):
        print("Creating word/pos dicts")
        word_dict, pos_dict = create_dicts()
        save_pickle(word_dict, WORD_PKL_PATH)
        save_pickle(pos_dict, POS_PKL_PATH)
        return word_dict, pos_dict
    else:
        print("Loading word/pos dicts")
        return load_pickle(WORD_PKL_PATH), load_pickle(POS_PKL_PATH)

Arc = namedtuple('Arc', 'head tail weight')

class MST_parser:
    def __init__(self):
        self.words_dict, self.tags_dict = create_or_load_dicts()
        self.weight_vec = dict()
        self.averaged_theta = dict()
        self.run_count = 0

    def feature_function(self, node1, node2, sentence = None):
        words_num, tags_num = len(self.words_dict), len(self.tags_dict)

        word1 = self.words_dict[node1[WORD_KEY]]
        word2 = self.words_dict[node2[WORD_KEY]]
        words_edge_id = word1*words_num +word2

        try:
            tag1 = self.tags_dict[node1[POS_KEY]]
            tag2 = self.tags_dict[node2[POS_KEY]]
            tags_edge_id = (tag1*tags_num) + tag2 + (words_num ** 2)
            return words_edge_id, tags_edge_id 
        except:
            print(node2[POS_KEY])

    def get_edge_score(self, node1, node2, sentence=None):
        words_edge_id, tags_edge_id = self.feature_function(node1, node2, sentence)
        return self.weight_vec.get(words_edge_id, 0) + self.weight_vec.get(tags_edge_id, 0)

    def get_all_negative_arcs(self, dep_graph: DependencyGraph):
        # weight is in minus for the MST func
        arc_set = list()
        for node1 in dep_graph.nodes.values():
            for node2 in dep_graph.nodes.values():
                if node1 != node2:
                    weight = self.get_edge_score(node1, node2)
                    arc_set.append(Arc(node1['address'], node2['address'], -weight))
        return arc_set

    def predict(self, dep_graph):
        arcs = self.get_all_negative_arcs(dep_graph)
        mst_arcs = min_spanning_arborescence_nx(arcs, None)
        return {(arc.head, arc.tail) for arc in mst_arcs.values()}

    def get_tree_edges(self, dep_graph: DependencyGraph):
        edge_list = set()
        for node in dep_graph.nodes.values():
            address = node['address']
            for child_index in node['deps'].get('\'\'', list()):
                edge_list.add((address, child_index))
        return edge_list

    def get_feature_dict(self, dep_graph: DependencyGraph, arc_set):
        feature_dict = dict()
        for arc in arc_set:
            head = dep_graph.nodes[arc[0]]
            tail = dep_graph.nodes[arc[1]]
            for index in self.feature_function(head, tail):
                feature_dict[index] = feature_dict.get(index, 0) + 1
        return feature_dict

    def get_dict_diff(self, dict1: dict, dict2: dict):
        diff_dict = dict()
        for index in dict1.keys():
            diff_dict[index] = dict1[index]
        for index in dict2.keys():
            diff_dict[index] = diff_dict.get(index, 0) - dict2[index]
        return diff_dict

    def update_averaged_theta(self):
        for index, val in self.averaged_theta.items():
            self.averaged_theta[index] *= self.run_count

        for index, val in self.weight_vec.items():
            self.averaged_theta[index] = self.averaged_theta.get(index, 0) + val

        for index, val in self.averaged_theta.items():
            self.averaged_theta[index] /= float(self.run_count + 1)

    def train_on_sentence(self, dep_graph: DependencyGraph):
        T_prediction = self.predict(dep_graph)
        T_ans = self.get_tree_edges(dep_graph)

        pred_features = self.get_feature_dict(dep_graph, T_prediction)
        ans_features = self.get_feature_dict(dep_graph, T_ans)
        difference = self.get_dict_diff(ans_features, pred_features)

        # new theta
        for index, diff in difference.items():
            self.weight_vec[index] = self.weight_vec.get(index, 0) + diff

        # update average with new theta:
        self.update_averaged_theta()
        self.run_count += 1

    def train(self, training_set, n_epochs):
        self.run_count = 0
        self.weight_vec = dict()
        self.averaged_theta = dict()

        for x in range(n_epochs):
            for i, dep_graph in enumerate(training_set):
                if not i % 100:
                    print(i)
                self.train_on_sentence(dep_graph)
        self.weight_vec = self.averaged_theta

if __name__ == '__main__':
    sents = dependency_treebank.parsed_sents()
    mst = MST_parser()
    mst.train(sents, 2)
    print(mst.weight_vec)