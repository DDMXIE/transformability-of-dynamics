# -*- coding: utf-8 -*-
# @Time : 2024/2/21 12:33 下午
# @Author : Tony
# @File : hypergraph.py
# @Project : targetIM

import copy
import math
import warnings
import networkx as nx
import hypernetx as hnx


class Hypergraph:

    def __init__(self, N, M, nd_hpe_dict, hpe_nd_dict):
        """
        initial method
        :param N: the number of nodes
        :param M: the number of hyperedges
        :param nd_hpe_dict: the relationship from node to hyperedge
        :param hpe_nd_dict: the relationship from hyperedge to node
        """
        self.__N = N
        self.__M = M
        self.__nd_hpe_dict = nd_hpe_dict
        self.__hpe_nd_dict = hpe_nd_dict

    def get_nd_hpe_dict(self):
        """
        getters
        :return: self.__nd_hpe_dict (the relationship from node to hyperedge)
        """
        return self.__nd_hpe_dict

    def get_hpe_nd_dict(self):
        """
        getters
        :return: self.__hpe_nd_dict (the relationship from hyperedge to node)
        """
        return self.__hpe_nd_dict

    def get_adjacent_node(self):
        """
        compute the adjacent nodes of a node
        :return: adj_dict (key: node, value: the adjacent nodes of the node)
        """
        dict_node = self.__nd_hpe_dict
        dict_edge = self.__hpe_nd_dict

        adj_dict = {}
        for nd, hpes in dict_node.items():
            adj_dict[nd] = list(set([x for hpe in hpes for x in dict_edge[hpe] if x != nd]))

        return adj_dict

    def get_hyper_degree(self, type):
        """
        compute the degree of nodes/hyperedges
        type：node/edge
        """
        res_dict = copy.deepcopy(self.__nd_hpe_dict)
        if type == 'edge':
            res_dict = copy.deepcopy(self.__hpe_nd_dict)
        for each in list(res_dict.keys()):
            res_dict[each] = len(res_dict[each])
        return res_dict


class WeightedHypergraph:

    def __init__(self, N, M, nd_hpe_dict, hpe_nd_dict, hpe_weight_dict):
        """
        initial method
        :param N: the number of nodes
        :param M: the number of hyperedges
        :param nd_hpe_dict: the relationship from node to hyperedge
        :param hpe_nd_dict: the relationship from hyperedge to node
        """
        self.__N = N
        self.__M = M
        self.__nd_hpe_dict = nd_hpe_dict
        self.__hpe_nd_dict = hpe_nd_dict
        self.__hpe_weight_dict = hpe_weight_dict

    def get_nd_hpe_dict(self):
        """
        getters
        :return: self.__nd_hpe_dict (the relationship from node to hyperedge)
        """
        return self.__nd_hpe_dict

    def get_hpe_nd_dict(self):
        """
        getters
        :return: self.__hpe_nd_dict (the relationship from hyperedge to node)
        """
        return self.__hpe_nd_dict

    def get_hpe_weight_dict(self):
        """
        getters
        :return: self.__hpe_weight_dict (the relationship from hyperedge to node)
        """
        return self.__hpe_weight_dict

    def get_adjacent_node(self):
        """
        compute the adjacent nodes of a node
        :return: adj_dict (key: node, value: the adjacent nodes of the node)
        """
        dict_node = self.__nd_hpe_dict
        dict_edge = self.__hpe_nd_dict

        adj_dict = {}
        for nd, hpes in dict_node.items():
            adj_dict[nd] = list(set([x for hpe in hpes for x in dict_edge[hpe] if x != nd]))

        return adj_dict

    def get_hyper_degree(self, type):
        """
        compute the degree of nodes/hyperedges
        type：node/edge
        """
        res_dict = copy.deepcopy(self.__nd_hpe_dict)
        if type == 'edge':
            res_dict = copy.deepcopy(self.__hpe_nd_dict)
        for each in list(res_dict.keys()):
            res_dict[each] = len(res_dict[each])
        return res_dict