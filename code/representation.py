# -*- coding: utf-8 -*-
# @Time : 2024/4/10 10:05 下午
# @Author : Tony
# @File : representation.py
# @Project : contagion_dynamics

import copy
import itertools
import numpy as np
import networkx as nx
from hypergraph import Hypergraph
from hypergraph import WeightedHypergraph


class Representation:

    def get_aggregated_weighted_hypergraph(self, org_hg):
        """
        get the aggregated hypergraph of an original hypergraph
        :return: aggregated_HG (the aggregated hyperedge-weighted hypergraph)
        """

        nd_hpe_dict = org_hg.get_nd_hpe_dict()
        hpe_nd_dict = org_hg.get_hpe_nd_dict()
        nd_hpe_cp_dict = copy.deepcopy(nd_hpe_dict)
        hpe_nd_cp_dict = copy.deepcopy(hpe_nd_dict)

        total_set_list = []
        hpe_nds_hpe_dict = {}
        hpe_weight_dict = {}

        for hpe, nds in hpe_nd_dict.items():
            if list(sorted(nds)) not in total_set_list:
                total_set_list.append(list(sorted(nds)))
                hpe_nds_hpe_dict[tuple(list(sorted(nds)))] = hpe
                hpe_weight_dict[hpe] = 1

            else:
                hpe_idx = hpe_nds_hpe_dict[tuple(list(sorted(nds)))]
                hpe_weight_dict[hpe_idx] += 1
                del hpe_nd_cp_dict[hpe]
                for nd in nds:
                    if hpe in nd_hpe_cp_dict[nd]:
                        nd_hpe_cp_dict[nd].remove(hpe)

        # re-order hpe_idx
        new_idx_lst = list(np.arange(len(list(hpe_nd_cp_dict.keys()))))
        hpe_nd_cp_dict_keys = list(hpe_nd_cp_dict.keys())
        hpe_nd_cp_dict_values = list(hpe_nd_cp_dict.values())
        reordered_hpe_nd_dict = {new_idx: rlt_nds for new_idx, rlt_nds in zip(new_idx_lst, hpe_nd_cp_dict_values)}
        originial_idx_to_new_idx = {origin_hpe_idx: new_idx
                                    for new_idx, origin_hpe_idx in zip(new_idx_lst, hpe_nd_cp_dict_keys)}
        reordered_hpe_weight_dict = {originial_idx_to_new_idx[origin_hpe_idx]: hpe_weight
                                     for origin_hpe_idx, hpe_weight in hpe_weight_dict.items()}
        reordered_nd_hpe_dict = nd_hpe_cp_dict
        aggregated_HG = WeightedHypergraph(len(reordered_nd_hpe_dict), len(reordered_hpe_nd_dict),
                                           reordered_nd_hpe_dict, reordered_hpe_nd_dict, reordered_hpe_weight_dict)
        return aggregated_HG

    def get_restricted_hypergraph(self, org_hg, restricted_num):
        """
        get the restricted hypergraph of an original hypergraph
        :return: restricted_HG (the (restricted_num)-degree restricted hypergraph)
        """

        nd_hpe_dict = org_hg.get_nd_hpe_dict()
        hpe_nd_dict = org_hg.get_hpe_nd_dict()
        nd_hpe_cp_dict = copy.deepcopy(nd_hpe_dict)
        hpe_nd_cp_dict = copy.deepcopy(hpe_nd_dict)
        max_hpe_idx = max(list(hpe_nd_dict.keys()))

        total_set_list = []

        for hpe, nds in hpe_nd_dict.items():

            if len(nds) > restricted_num:
                restricted_hpes = itertools.combinations(set(nds), restricted_num)
                for restricted_hpe in restricted_hpes:
                    if set(restricted_hpe) not in total_set_list:
                        total_set_list.append(set(restricted_hpe))
                        hpe_nd_cp_dict[max_hpe_idx] = list(restricted_hpe)
                        # 加入关联
                        for nd in list(restricted_hpe):
                            nd_hpe_cp_dict[nd].append(max_hpe_idx)
                        max_hpe_idx += 1
                del hpe_nd_cp_dict[hpe]
                for nd in nds:
                    if hpe in nd_hpe_cp_dict[nd]:
                        nd_hpe_cp_dict[nd].remove(hpe)
            else:
                if set(nds) not in total_set_list:
                    total_set_list.append(set(nds))
                else:
                    del hpe_nd_cp_dict[hpe]
                    for nd in nds:
                        if hpe in nd_hpe_cp_dict[nd]:
                            nd_hpe_cp_dict[nd].remove(hpe)

        cnt = 0
        for hpe, nds in hpe_nd_cp_dict.items():
            if len(nds) >= restricted_num:
                cnt += 1

        restricted_HG = Hypergraph(len(nd_hpe_cp_dict), len(hpe_nd_cp_dict),
                                           nd_hpe_cp_dict, hpe_nd_cp_dict)
        return restricted_HG

    def get_projected_network(self, org_hg):
        """
        get the clique expansion of an original hypergraph
        :return: G (the projected network)
        """

        dict_node = org_hg.get_nd_hpe_dict()
        dict_edge = org_hg.get_hpe_nd_dict()

        adj_dict = {}
        for nd, hpes in dict_node.items():
            adj_dict[nd] = list(set([x for hpe in hpes for x in dict_edge[hpe] if x != nd]))

        G = nx.Graph()
        G.add_nodes_from(list(adj_dict.keys()))

        for from_node in adj_dict:
            node_list = adj_dict[from_node]
            for to_node in node_list:
                G.add_edge(from_node, to_node)

        return G

    def get_multi_edge_graph_from_hypergraph(self, org_hg):
        """
        get the multi-edge graph of an original hypergraph
        :return: MG (the multi-edge graph)
        """

        dict_node = org_hg.get_nd_hpe_dict()
        dict_edge = org_hg.get_hpe_nd_dict()

        MG = nx.MultiGraph()
        MG.add_nodes_from(list(dict_node.keys()))

        for hpe, hpe_nds in dict_edge.items():
            items = hpe_nds
            combination_size = 2
            combinations = itertools.combinations(items, combination_size)
            for from_node, to_node in combinations:
                MG.add_edge(from_node, to_node)

        return MG