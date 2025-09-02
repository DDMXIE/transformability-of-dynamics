# -*- coding: utf-8 -*-
# @Time : 2024/4/10 5:20 下午
# @Author : Tony
# @File : contagion_dynamics_model.py
# @Project : contagion_dynamics
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class Contagions:

    """
    Toy models for contagion dynamics
    """

    def si_simple_contagion_model(self, G, times, beta, R, init_seed_num):
        nd_lst = list(G.nodes())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_simple_contagion_model: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                rlt_adj_nds = []
                for i_nd in list(i_nd_set):
                    rlt_adj_nds.extend(list(G.adj[i_nd]))
                prob_arr = np.random.random(len(rlt_adj_nds))
                new_i_nd_candidate_set = set(list(np.array(rlt_adj_nds)[np.where(prob_arr < beta)[0]]))
                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict

        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_simplicial_contagion_model(self, res_HG, times, beta_pairwise, beta_triangle, R, init_seed_num):
        nd_hpe_dict = res_HG.get_nd_hpe_dict()
        hpe_nd_dict = res_HG.get_hpe_nd_dict()
        nd_lst = list(nd_hpe_dict.keys())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_simplicial_contagion_model: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                new_i_nd_candidate_set = set({})
                for nd in nd_lst:
                    if nd in i_nd_set:
                        continue
                    rlt_pairwise_hpe_lst, rlt_triangle_hpe_lst = [], []
                    for hpe, nds in hpe_nd_dict.items():
                        if nd not in set(nds):
                            continue
                        hpe_nds_set = set(nds) - set(list([nd]))
                        if len(nds) == 3 and len((hpe_nds_set & i_nd_set)) == 2:
                            rlt_triangle_hpe_lst.append(hpe)
                            rlt_pairwise_hpe_lst.extend([hpe, hpe])
                        if len(nds) == 2 and len((hpe_nds_set & i_nd_set)) == 1:
                            rlt_pairwise_hpe_lst.append(hpe)

                    tri_prob_arr = np.random.random(len(rlt_triangle_hpe_lst))
                    new_tri_prob_set = set(list(np.array(np.where(tri_prob_arr < beta_triangle)[0])))
                    pair_prob_arr = np.random.random(len(rlt_pairwise_hpe_lst))
                    new_pair_prob_set = set(list(np.array(np.where(pair_prob_arr < beta_pairwise)[0])))
                    if len(new_tri_prob_set | new_pair_prob_set) >= 1:
                        new_i_nd_candidate_set.add(nd)

                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict
        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_simplicial_contagion_improved_model(self, res_HG, times, beta_pairwise, beta_triangle, R, init_seed_num):
        nd_hpe_dict = res_HG.get_nd_hpe_dict()
        hpe_nd_dict = res_HG.get_hpe_nd_dict()
        nd_lst = list(nd_hpe_dict.keys())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_simplicial_contagion_improved_model: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                new_i_nd_candidate_set = set({})
                for nd in nd_lst:
                    if nd in i_nd_set:
                        continue
                    rlt_pairwise_hpe_lst, rlt_triangle_hpe_lst = [], []
                    for hpe, nds in hpe_nd_dict.items():
                        if nd not in set(nds):
                            continue
                        hpe_nds_set = set(nds) - set(list([nd]))
                        if len(nds) == 3 and len((hpe_nds_set & i_nd_set)) == 2:
                            rlt_triangle_hpe_lst.append(hpe)
                            rlt_pairwise_hpe_lst.extend([hpe, hpe])
                        if len(nds) == 3 and len((hpe_nds_set & i_nd_set)) == 1:
                            rlt_pairwise_hpe_lst.append(hpe)
                        if len(nds) == 2 and len((hpe_nds_set & i_nd_set)) == 1:
                            rlt_pairwise_hpe_lst.append(hpe)

                    tri_prob_arr = np.random.random(len(rlt_triangle_hpe_lst))
                    new_tri_prob_set = set(list(np.array(np.where(tri_prob_arr < beta_triangle)[0])))
                    pair_prob_arr = np.random.random(len(rlt_pairwise_hpe_lst))
                    new_pair_prob_set = set(list(np.array(np.where(pair_prob_arr < beta_pairwise)[0])))
                    if len(new_tri_prob_set | new_pair_prob_set) >= 1:
                        new_i_nd_candidate_set.add(nd)

                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict
        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_non_linear_hypergraph_model(self, agr_HG, times, lmd, nu, R, init_seed_num):
        nd_hpe_dict = agr_HG.get_nd_hpe_dict()
        hpe_nd_dict = agr_HG.get_hpe_nd_dict()
        nd_lst = list(nd_hpe_dict.keys())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_non_linear_hypergraph_model: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                new_i_nd_candidate_set = set({})
                for hpe, hpe_nds in hpe_nd_dict.items():
                    n = len(set(hpe_nds) & i_nd_set)
                    hpe_candidate_nd = set(hpe_nds) - i_nd_set
                    if n < 1:
                        continue
                    if len(hpe_candidate_nd) >= 1:
                        prob_arr = np.random.random(len(hpe_candidate_nd))
                        cof = lmd * (n ** nu)
                        hpe_nd_candidate_set = set(list(np.array(list(hpe_candidate_nd))[np.where(prob_arr < cof)[0]]))
                        new_i_nd_candidate_set = new_i_nd_candidate_set | hpe_nd_candidate_set
                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict
        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_simple_contagion_model_in_multi_edge_graph(self, MG, times, beta, R, init_seed_num):
        nd_lst = list(MG.nodes())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_simple_contagion_model_in_multi_edge_graph: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                rlt_adj_nds = []
                for i_nd in list(i_nd_set):
                    rlt_adj_nds.extend([y for x, x_item in MG.adj[i_nd].items() for y in [x] * len(x_item)])
                prob_arr = np.random.random(len(rlt_adj_nds))
                new_i_nd_candidate_set = set(list(np.array(rlt_adj_nds)[np.where(prob_arr < beta)[0]]))
                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict

        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_non_linear_hypergraph_model_in_restricted_hypergraph(self, res_HG, times, lmd, nu, R, init_seed_num):
        nd_hpe_dict = res_HG.get_nd_hpe_dict()
        hpe_nd_dict = res_HG.get_hpe_nd_dict()
        nd_lst = list(nd_hpe_dict.keys())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_non_linear_hypergraph_model_in_restricted_hypergraph: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                new_i_nd_candidate_set = set({})
                for hpe, hpe_nds in hpe_nd_dict.items():
                    n = len(set(hpe_nds) & i_nd_set)
                    hpe_candidate_nd = set(hpe_nds) - i_nd_set
                    if n < 1:
                        continue
                    if len(hpe_candidate_nd) >= 1:
                        prob_arr = np.random.random(len(hpe_candidate_nd))
                        cof = lmd * (n ** nu)
                        hpe_nd_candidate_set = set(list(np.array(list(hpe_candidate_nd))[np.where(prob_arr < cof)[0]]))
                        new_i_nd_candidate_set = new_i_nd_candidate_set | hpe_nd_candidate_set
                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict
        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict

    def si_non_linear_hypergraph_model_in_restricted_hypergraph_with_different_cases(self, res_HG, times, lmd_fst, lmd_sec, lmd_trd, nu, R, init_seed_num):
        nd_hpe_dict = res_HG.get_nd_hpe_dict()
        hpe_nd_dict = res_HG.get_hpe_nd_dict()
        nd_lst = list(nd_hpe_dict.keys())
        i_num_mtx = []
        i_time_total_dict = {}
        for r in tqdm(range(R), desc='si_non_linear_hypergraph_model_in_restricted_hypergraph_with_different_cases: loading average times...'):
            initial_i_nd_num = init_seed_num
            i_nd_lst = random.sample(nd_lst, initial_i_nd_num)
            i_nd_set = set(i_nd_lst)
            i_num_lst = [len(list(i_nd_set)) / len(nd_lst)]
            i_time_dict = {}
            for t in range(times):
                new_i_nd_candidate_set = set({})
                for hpe, hpe_nds in hpe_nd_dict.items():
                    n = len(set(hpe_nds) & i_nd_set)
                    hpe_candidate_nd = set(hpe_nds) - i_nd_set

                    if n < 1:
                        continue

                    len_s_nd_num = len(hpe_candidate_nd)
                    if len_s_nd_num >= 1:
                        lmd = 0
                        if len_s_nd_num == 1 and n == 1:   # m=2 case1
                            lmd = lmd_fst
                        elif len_s_nd_num == 1 and n == 2:     # m=3 case1
                            lmd = lmd_sec
                        elif len_s_nd_num == 2 and n == 1:     # m=3 case2
                            lmd = lmd_trd
                        prob_arr = np.random.random(len(hpe_candidate_nd))
                        cof = lmd * (n ** nu)
                        hpe_nd_candidate_set = set(list(np.array(list(hpe_candidate_nd))[np.where(prob_arr < cof)[0]]))
                        new_i_nd_candidate_set = new_i_nd_candidate_set | hpe_nd_candidate_set

                new_i_nd_set = new_i_nd_candidate_set - i_nd_set
                i_nd_set = i_nd_set | new_i_nd_set
                i_num_lst.append(len(list(i_nd_set)) / len(nd_lst))
                i_time_dict[t] = list(new_i_nd_set)
            i_num_mtx.append(i_num_lst)
            i_time_total_dict[r] = i_time_dict
        return pd.DataFrame(i_num_mtx).mean(axis=0), 1 - pd.DataFrame(i_num_mtx).mean(axis=0), i_time_total_dict