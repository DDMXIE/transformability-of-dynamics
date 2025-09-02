# -*- coding: utf-8 -*-
# @Time : 2024/5/13 09:54 上午
# @Author : Tony
# @File : hypergraph_generator.py
# @Project : contagion_dynamics

import random
import numpy as np
from itertools import combinations


class HypergraphGenerator():
    """
    Null model of a hypergraph
    === UPDATING ===
    """

    def HyperCL(self, hpe_size_dtb, nd_deg_dtb, nd_num, hpe_num):
        """
        HyperCL generator: Generating a hypergraph with the distributions of both the hyperedge size and node degree
        :param hpe_size_dtb: the distribution of hyperedge size
        :param nd_deg_dtb: the distribution of node hyper-degree
        :param nd_num: the number of nodes
        :param hpe_num: the number of hyperedges
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """
        nd_lst = np.arange(nd_num)
        nd_dict = {nd_idx: [] for nd_idx in range(nd_num)}
        hpe_dict = {}
        for idx, hpe_idx in enumerate(range(hpe_num)):
            hpe_nd_set = []
            while len(hpe_nd_set) < hpe_size_dtb[hpe_idx]:
                prob = np.array(nd_deg_dtb) / sum(nd_deg_dtb)
                v = np.random.choice(nd_lst, p=prob)
                if v not in hpe_nd_set:
                    hpe_nd_set.append(v)
                    if hpe_idx not in nd_dict[v]:
                        nd_dict[v].append(hpe_idx)
            hpe_dict[hpe_idx] = hpe_nd_set
        return nd_dict, hpe_dict

    def complete_hypergraph(self, nd_num):
        """
        Complete hypergraph generator: Generating a hypergraph that each k nodes contain a hyperedge in k-order
        :param nd_num: the number of nodes in a hypergraph
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """
        N = nd_num
        # 定义元素列表
        elements = list(range(N))  # 生成0到9的列表

        hpe_idx = 0
        nd_hpe_dict = {key: [] for key in list(np.arange(N))}
        hpe_nd_dict = {}
        # 计算任意抽取2个到9个不重复组合方式
        for r in range(2, N + 1):
            # 使用combinations生成r个元素的所有组合
            combos = list(combinations(elements, r))
            for combo in combos:
                hpe_nd_dict[hpe_idx] = list(combo)
                for item in combo:
                    if hpe_idx not in nd_hpe_dict[item]:
                        nd_hpe_dict[item].append(hpe_idx)
                hpe_idx += 1
        return nd_hpe_dict, hpe_nd_dict

    def erdos_renyi_uniform_hypergraph(self, N, M, k):
        """
        Erdos_renyi_uniform_hypergraph generator: Generating a uniform Erdos-renyi hypergraph
        :param M: the maximal size of a hyperedge in a hypergraph
        :param k: the uniform order of k-uniform hypergraph
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """

        edge_count = 0
        edge_set_list = []

        nodes_dict = {}
        for i in range(N):
            nodes_dict[i] = []

        edges_dict = {}
        for i in range(M):
            edges_dict[i] = []

        while 1:
            if len(edge_set_list) == M:
                break
            selected_nodes = np.random.choice(np.arange(N), k, replace=False)
            selected_nodes_set = set(list(selected_nodes))
            if selected_nodes_set in edge_set_list:
                continue
            else:
                edges_dict[edge_count] = list(selected_nodes)
                for node in selected_nodes:
                    nodes_dict[node].append(edge_count)
                edge_set_list.append(selected_nodes_set)
                edge_count = edge_count + 1

        return nodes_dict, edges_dict

    def barabasi_albert_uniform_hypergraph(self, N, M, k, mu):
        """
        Erdos_renyi_uniform_hypergraph generator: Generating a uniform Erdos-renyi hypergraph
        :param N: the number of nodes in a hypergraph
        :param M: the maximal size of a hyperedge in a hypergraph
        :param k: the uniform order of k-uniform hypergraph
        :param mu: the power-law exponent in node degree distribution (negative value)
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """

        edge_count = 0
        edge_set_list = []

        nodes_dict = {}
        for i in range(N):
            nodes_dict[i] = []

        edges_dict = {}
        for i in range(M):
            edges_dict[i] = []

        node_weight_initial_list = np.power(np.arange(1, N + 1), -mu)
        node_weight_list = node_weight_initial_list / np.sum(node_weight_initial_list)
        print(node_weight_list)

        while 1:
            if len(edge_set_list) == M:
                break
            selected_nodes = np.random.choice(N, size=k, replace=False,
                                              p=node_weight_list)
            selected_nodes_set = set(list(np.array(selected_nodes)))
            if selected_nodes_set in edge_set_list:
                continue
            else:
                edges_dict[edge_count] = list(selected_nodes)
                for node in selected_nodes:
                    nodes_dict[node].append(edge_count)
                edge_set_list.append(selected_nodes_set)
                edge_count = edge_count + 1

        return nodes_dict, edges_dict

    def watts_strogatz_uniform_hypergraph(self, N, k, p):
        """
        Watts_strogatz_uniform_hypergraph generator: Generating a uniform Watts_strogatz hypergraph
        :param N: the number of nodes in a hypergraph
        :param k: the k-order of adjacency neighbors
        :param p: the probability of rewising hyperedges
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """
        node_list = list(np.arange(N))
        # generate k-order adjacency hypergraph
        hpe_list = []
        for i in range(N):

            hpe_right = [i]
            for idx in range(int(k / 2)):
                if i + idx + 1 <= N - 1:
                    hpe_right.append(i + idx + 1)
                else:
                    hpe_right.append(i + idx + 1 - N)
            hpe_list.append(hpe_right)

        # random rewised hyperedges
        for hpe in hpe_list:
            if random.random() < p:
                new_hpe = [random.sample(hpe, 1)[0]]
                hpe_list.remove(hpe)
                new_hpe.extend(random.sample(node_list, int(k / 2)))
                hpe_list.append(new_hpe)
        print(hpe_list)

        # re-adjust the nd_hpe_lst & hpe_nd_lst
        nd_hpe_dict, hpe_nd_dict = {}, {}
        hpe_idx = 0
        nd_hpe_dict = {nd: [] for nd in range(N)}
        for nd_in_hpes in hpe_list:
            hpe_nd_dict[hpe_idx] = nd_in_hpes
            for nd in nd_in_hpes:
                nd_hpe_dict[nd].append(hpe_idx)
            hpe_idx += 1
        return nd_hpe_dict, hpe_nd_dict


    def k_nearest_neighbors_uniform_hypergraph(self, N, k):
        """
        k_nearest_neighbors_uniform_hypergraph generator: Generating a uniform Watts_strogatz hypergraph
        :param N: the number of nodes in a hypergraph
        :param k: the k-order of adjacency neighbors
        :return: node-to-hyperedge & hyperedge-to-node adjacency relationships
        """
        # generate k-order adjacency hypergraph
        hpe_list = []
        for i in range(N):

            hpe_right = [i]
            for idx in range(int(k / 2)):
                if i + idx + 1 <= N - 1:
                    hpe_right.append(i + idx + 1)
                else:
                    hpe_right.append(i + idx + 1 - N)
            hpe_list.append(hpe_right)

        # re-adjust the nd_hpe_lst & hpe_nd_lst
        nd_hpe_dict, hpe_nd_dict = {}, {}
        hpe_idx = 0
        nd_hpe_dict = {nd: [] for nd in range(N)}
        for nd_in_hpes in hpe_list:
            hpe_nd_dict[hpe_idx] = nd_in_hpes
            for nd in nd_in_hpes:
                nd_hpe_dict[nd].append(hpe_idx)
            hpe_idx += 1
        return nd_hpe_dict, hpe_nd_dict