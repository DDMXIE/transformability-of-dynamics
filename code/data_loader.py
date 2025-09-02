# -*- coding: utf-8 -*-
# @Time : 2024/4/10 4:07 下午
# @Author : Tony
# @File : data_loader.py
# @Project : contagion_dynamics

import numpy as np
import pandas as pd


class DataPreManage:

    def __init__(self, params):
        """
        initial method
        :param node_dict: key: value
                        key: the index of node in real data,
                        value: the index of node in managed data
        :param node_arr: the sorted index array of node in real data
        :param node_list: the original index list of node in real data
        :param arr: the formatted array for the original .txt file of the real data
                    e.g., [['1 2']
                          ['3 4 5']
                          ['6 7 8']
                          ...
                          ['723 917 1160 1161']
                          ['197 632 633 993 994']
                          ['408 409 410']]
        """
        self.__node_dict, self.__node_arr, self.__node_list, self.__arr = params

    def load_data(self, path):
        """
        load and initialize the data of a hypergraph
        :param path: the file path of the dataset
        :return: node_dict, node_arr, node_list, arr
        """
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        node_list = []
        for each in arr:
            node_list.extend(list(map(int, each[0].split(" "))))
        node_arr = np.unique(np.array(node_list))
        node_dict = {node_arr[i]: i for i in range(0, len(node_arr))}
        self.__init__([node_dict, node_arr, node_list, arr])
        return node_dict, node_arr, node_list, arr

    def get_edge_dict(self, path):
        """
        get the hpe-node relationship from the loaded data
        :param path:
        :return: hpe_dict (the relationship from node to hyperedge)
        """
        node_dict, node_arr, node_list, arr = self.__node_dict, self.__node_arr, self.__node_list, self.__arr
        hpe_dict = {}
        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values
        i = 0
        for each in arr:
            nodes_index_list = list(map(int, each[0].split(" ")))
            hpe_dict[i] = [node_dict[idx] for idx in nodes_index_list]
            i += 1
        return hpe_dict

    def get_nodes_dict(self, path):
        """
        get the node-hpe relationship from the loaded data
        :param path:
        :return: nodes_dict (the relationship from hyperedge to node)
        """
        node_dict, node_arr, node_list, arr = self.__node_dict, self.__node_arr, self.__node_list, self.__arr
        total = len(node_dict.values())

        nodes_dict = {i: [] for i in range(0, total)}

        df = pd.read_csv(path, index_col=False, header=None)
        arr = df.values

        i = 0
        for each in arr:
            nodes_index_list = list(map(int, each[0].split(" ")))
            for index in nodes_index_list:
                nodes_dict[node_dict[index]].append(i)
            i += 1
        return nodes_dict
