import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

import pandas as pd
from clyent import color
from holoviews.plotting.bokeh.styles import font_size
from tqdm import tqdm

from hypergraph_generator import HypergraphGenerator
from hypergraph import Hypergraph
from representation import Representation
from matplotlib.colors import LinearSegmentedColormap


def opinion_dynamics_in_graph(G, states, c, steps):
    variances = [np.var(states)]
    edge_lst = list(G.edges())
    for _ in range(steps):
        new_states = states.copy()
        s_nd, t_nd = random.choices(edge_lst)[0]

        d = (new_states[s_nd] - new_states[t_nd]) ** 2
        if d < c:
            new_states[s_nd], new_states[t_nd] = 1 / 2 * (new_states[s_nd] + new_states[t_nd]), 1 / 2 * (
                        new_states[s_nd] + new_states[t_nd])

        variance = np.var(new_states)
        variances.append(variance)

        states = new_states

    return states, variances


def opinion_dynamics_in_multi_graph(MG, states, c, steps):
    variances = [np.var(states)]
    edge_lst = list(MG.edges())
    for _ in range(steps):
        new_states = states.copy()

        s_nd, t_nd = random.choices(edge_lst)[0]
        edge_weight = MG.number_of_edges(s_nd, t_nd)

        d = (1 / edge_weight) * ((new_states[s_nd] - new_states[t_nd]) ** 2)
        if d < c:
            new_states[s_nd], new_states[t_nd] = 1 / 2 * (new_states[s_nd] + new_states[t_nd]), 1 / 2 * (
                        new_states[s_nd] + new_states[t_nd])


        variance = np.var(new_states)
        variances.append(variance)

        states = new_states

    return states, variances


def opinion_dynamics_in_hypergraph(HG, states, c, steps, alpha):
    variances = [np.var(states)]
    MG = r.get_multi_edge_graph_from_hypergraph(HG)
    edge_lst = list(MG.edges())
    hpe_nd_lst = HG.get_hpe_nd_dict().values()
    states_mtx = []
    for _ in range(steps):
        new_states = states.copy()
        s_nd, t_nd = random.choices(edge_lst)[0]

        rlt_hpe_size_lst = []
        for hpe_nds in hpe_nd_lst:
            if s_nd in hpe_nds and t_nd in hpe_nds:
                rlt_hpe_size_lst.append(len(hpe_nds))
        ksi = 1 / np.sum(np.array([x ** alpha for x in rlt_hpe_size_lst]))
        d = ksi * ((new_states[s_nd] - new_states[t_nd]) ** 2)

        if d < c:
            new_states[s_nd], new_states[t_nd] = 1 / 2 * (new_states[s_nd] + new_states[t_nd]), 1 / 2 * (
                    new_states[s_nd] + new_states[t_nd])
        states_mtx.append(new_states)

        variance = np.var(new_states)
        variances.append(variance)

        states = new_states

    return states, variances, states_mtx


def avg_opinion_dynamics_in_graph(G, states_init, c, iterations, steps):
    variances_mtx = []
    for _ in enumerate(tqdm(range(iterations), desc='loading...')):
        states, variances = opinion_dynamics_in_graph(G, copy.deepcopy(states_init), c, steps)
        variances_mtx.append(variances)
    return pd.DataFrame(variances_mtx).mean(axis=0)


def avg_opinion_dynamics_in_multi_graph(MG, states_init, c, iterations, steps):
    variances_mtx = []
    for _ in enumerate(tqdm(range(iterations), desc='loading...')):
        states, variances = opinion_dynamics_in_multi_graph(MG, copy.deepcopy(states_init), c, steps)
        variances_mtx.append(variances)
    return pd.DataFrame(variances_mtx).mean(axis=0)


def avg_opinion_dynamics_in_hypergraph(HG, states_init, c, iterations, steps, alpha):
    variances_mtx = []
    for _ in enumerate(tqdm(range(iterations), desc='loading...')):
        states, variances, states_mtx = opinion_dynamics_in_hypergraph(HG, copy.deepcopy(states_init), c, steps, alpha)
        print(len(states_mtx))
        df_states = pd.DataFrame(states_mtx)
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=300)
        for i in range(nd_num):
            opinion_lst = df_states[i]
            plt.plot(np.arange(len(opinion_lst)), opinion_lst, color=cmap_1(i * 1 / nd_num), alpha=0.8)

        plt.subplots_adjust(left=0.2, bottom=0.23)
        plt.xlabel(r'$t$', fontsize=19)
        plt.ylabel(r'$\mathcal{O}(t)$', fontsize=19)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.tick_params(axis='both', direction='in', length=5, width=1.5, pad=5)
        ax.spines['top'].set_linewidth(1.5)  # 下边框粗细
        ax.spines['bottom'].set_linewidth(1.5)  # 下边框粗细
        ax.spines['left'].set_linewidth(1.5)  # 左边框粗细
        ax.spines['right'].set_linewidth(1.5)  # 右边框粗细
        plt.show()
        variances_mtx.append(variances)
    return pd.DataFrame(variances_mtx).mean(axis=0)


def plot_variances(iterations, variances):
    plt.plot(np.arange(iterations + 1), variances)
    plt.show()


# generate power-law array
def rndm(a, b, g, size=1):
    """
    Power-law gen for pdf(x) \propto x^{g-1} for a<=x<=b
    """
    r = np.random.random(size=size)
    ag, bg = a ** g, b ** g
    return (ag + (bg - ag) * r) ** (1. / g)


# Initialization
max_order = 10  # the number of nodes in each hyperedge
hpe_num = 100  # the number of hyperedges
iterations = 1000
steps = 1000
c = 0.1

colors_bar_1 = [ "white",  "#e04051"]
cmap_1 = LinearSegmentedColormap.from_list('my_cmap_1', colors_bar_1)
hpe_size_dtb = np.random.randint(2, max_order, size=hpe_num)  # the distribution of hyperedge size

nd_num = 100  # the number of nodes
nd_max_deg = 10  # the maximum hyper-degree of node in a hypergraph
nd_deg_dtb = rndm(2, nd_max_deg, g=-2, size=nd_num)  # the distribution of node hyper-degree

# Generate hypergraph
hg_grt = HypergraphGenerator()
nd_to_hpe_dict, hpe_to_node_dict = hg_grt.HyperCL(hpe_size_dtb, nd_deg_dtb, nd_num, hpe_num)

# Initialize the generated hypergraph
HG = Hypergraph(len(nd_to_hpe_dict.keys()), len(hpe_to_node_dict.keys()), nd_to_hpe_dict, hpe_to_node_dict)
nd_hpe_dict = HG.get_nd_hpe_dict()
hpe_nd_dict = HG.get_hpe_nd_dict()
N, M = len(list(nd_hpe_dict.keys())), len(list(hpe_nd_dict.keys()))

r = Representation()
G = r.get_projected_network(HG)
MG = r.get_multi_edge_graph_from_hypergraph(HG)

states = np.random.rand(N)
states_graph = copy.deepcopy(states)
states_multi = copy.deepcopy(states)
states_hyper = copy.deepcopy(states)
variances = avg_opinion_dynamics_in_graph(MG, states_graph, c, iterations, steps)
variances_multi = avg_opinion_dynamics_in_multi_graph(MG, states_multi, c, iterations, steps)
variances_hyper = avg_opinion_dynamics_in_hypergraph(HG, states_hyper, c, iterations, steps, alpha=0)
variances_hyper_sec = avg_opinion_dynamics_in_hypergraph(HG, states_hyper, c, iterations, steps, alpha=1/2)
