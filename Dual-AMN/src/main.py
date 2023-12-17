import numpy as np
import os
import torch
import math
import tqdm
import sys
from model import Encoder_Model
import warnings
import difflib
import argparse
import time
from anchor import anchor_tabular
from collections import defaultdict
from preprocessing import DBpDataset
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import torch.nn as nn
from count import read_list, read_tri, read_link
import torch.nn.functional as F
import time
from tqdm import trange
import copy
from eval import Evaluate
from itertools import combinations
import random
from sklearn import linear_model
import pandas as pd
from LORE import util
from LORE.prepare_dataset import prepare_EA_dataset
from LORE import lore
from LORE.neighbor_generator import genetic_neighborhood

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def comb(n, m):
    return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))
def sigmoid(x):
    return 1 / (math.exp(-x) + 1)
def seed_torch(seed=1029):
    print('set seed')
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

class Proxy:
    def __init__(self, embed1, embed2):
        self.embed1 = embed1.cuda() 
        self.embed2 = embed2.cuda()


    def aggregation(self, tri1, tri2):
        # print(tri1, tri2)
        return self.embed1[tri1].mean(dim = 0), self.embed2[tri2].mean(dim = 0)
    
    def all_aggregation(self):
        return self.embed1.mean(dim = 0), self.embed2.mean(dim = 0)
    
    def sim(self, tri1, tri2):
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        return F.cosine_similarity(self.embed1[tri1].mean(dim = 0), self.embed2[tri2].mean(dim = 0), dim=0)

    def mask_sim(self, mask, split):

        return F.cosine_similarity((self.embed1 * mask[:split].unsqueeze(1)).mean(dim = 0), (self.embed2 * mask[split:].unsqueeze(1)).mean(dim = 0), dim=0)







class Shapley_Value:
    def __init__(self, model, e1, e2, num_players, players, split, p, e_dict, r_dict, dataset):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.dataset = dataset

    def value(self, tri1, tri2, method='ori'):
        if method == 'ori':
            # ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
            # print(node_size,adj_list, self.model.ent_embedding.weight.shape )
            # proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            try:
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
                # print(node_size,adj_list, self.model.ent_embedding.weight.shape )
                proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
            except RuntimeError as exception:
                # print('out_memory')
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception
        else:
            v_suf = self.model.sim(tri1, tri2)
        return v_suf

    def compute_coal_value(self, gid1, gid2, c):
        c = torch.Tensor(c)
        neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        neigh1 = F.normalize(neigh1, dim = 0)
        neigh2 = self.embed[c.t()[1].long()].mean(dim = 0)
        neigh2 = F.normalize(neigh2, dim = 0)
        return F.cosine_similarity(neigh1, neigh2, dim=0)


    def compute_shapley_value(self, num_simulations, method='ori',pair=[]):
        if self.num_players > 5:
            return self.MTC(num_simulations, method,pair)

        all_coal = [list(coal) for i in range(self.num_players) for coal in combinations(pair, i + 1)]
        shapely_value = []
        for p in pair:
            # print(e)
            p_coal = []
            no_p_coal = []
            
            for c in copy.deepcopy(all_coal):
                if p in c:
                    c_t = torch.Tensor(c)
                    value = self.value(c_t.t()[0].long(), c_t.t()[1].long(), method)
                    l = len(c)
                    p_coal.append((l, value))
                    c.remove(p)
                    if len(c) == 0:
                        no_p_coal.append((0, 0))
                    else:
                        c_t = torch.Tensor(c)
                        value = self.value(c_t.t()[0].long(), c_t.t()[1].long(), method)
                        no_p_coal.append((l - 1, value))
            # print('e联盟: ', e_coal)
            # print('noe联盟: ', no_e_coal)
            shapelyvalue = 0
            for i in range(len(p_coal)):
                s = p_coal[i][0]
                p_payoff = p_coal[i][1] - no_p_coal[i][1]
                p_weight = math.factorial(s-1)*math.factorial(self.num_players-s)/math.factorial(self.num_players)
                shapelyvalue += p_payoff * p_weight
            shapely_value.append((p, shapelyvalue))
        # print('夏普利值：',shapely_value)
        # print(len(shapely_value))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        new_p = []
        for cur in shapely_value:
            new_p.append([cur[0][0], cur[0][1]])
        return new_p       

    def MTC(self, num_simulations, method='ori',pair=[]):
        shapley_values = np.zeros(self.num_players)
        for _ in range(num_simulations):
        # 生成随机排列的玩家列表
            players = np.random.permutation( self.num_players)
            # print(players, split)
            # 初始化联盟价值和玩家计数器
            coalition_value = 0
            player_count = 0
            if method == 'ori':
                tri1 = []
                tri2 = []
                for player in self.players:
                    # 计算当前联盟中添加玩家后的价值差异
                    if player < self.split:
                        tri1.append(self.p[player])
                    else:
                        tri2.append(self.p[player])

                    coalition_value_with_player = self.value(tri1, tri2, method)
                    # print(tri1, tri2)
                    # print(coalition_value_with_player)
                    marginal_contribution = coalition_value_with_player - coalition_value

                    # 计算当前玩家的 Shapley 值
                    shapley_values[player] += marginal_contribution / num_simulations

                    # 更新联盟价值和玩家计数器
                    coalition_value = coalition_value_with_player
            else:
                tri1 = []
                tri2 = []
                for player in players:
                    # 计算当前联盟中添加玩家后的价值差异
                    tri1.append(pair[player][0])
                    tri2.append(pair[player][1])

                    coalition_value_with_player = self.value(tri1, tri2, method)
                    # print(tri1, tri2)
                    # print(coalition_value_with_player)
                    marginal_contribution = coalition_value_with_player - coalition_value

                    # 计算当前玩家的 Shapley 值
                    shapley_values[player] += marginal_contribution / num_simulations

                    # 更新联盟价值和玩家计数器
                    coalition_value = coalition_value_with_player
                # player_count += 1
        res = torch.Tensor(shapley_values).argsort(descending=True)
        if method == 'ori':
            return shapley_values
        else:
            new_p = []
            for cur in res:
                new_p.append(pair[cur])
            return new_p

class Shapley_Value_two:
    def __init__(self, model, e1, e2, num_players, players, split, p, dataset):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.dataset = dataset

    def value(self, tri1, tri2, method='ori'):
        if method == 'ori':
            try:
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e1, self.e2, tri1, tri2, True)
                proxy_e1, proxy_e2 = self.model.get_embeddings([self.e1], [self.e2], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception
        else:
            v_suf = self.model.sim(tri1, tri2)
        return v_suf

    def compute_coal_value(self, gid1, gid2, c):
        c = torch.Tensor(c)
        neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        neigh1 = F.normalize(neigh1, dim = 0)
        neigh2 = self.embed[c.t()[1].long()].mean(dim = 0)
        neigh2 = F.normalize(neigh2, dim = 0)
        return F.cosine_similarity(neigh1, neigh2, dim=0)


    def compute_shapley_value(self, num_simulations, method='ori',pair=[]):
        if self.num_players > 5:
            return self.MTC(num_simulations, method,pair)

        all_coal = [list(coal) for i in range(self.num_players) for coal in combinations(pair, i + 1)]
        shapely_value = []
        for p in pair:
            # print(e)
            p_coal = []
            no_p_coal = []
            
            for c in copy.deepcopy(all_coal):
                if p in c:
                    c_t = torch.Tensor(c)
                    value = self.value(c_t.t()[0].long(), c_t.t()[1].long(), method)
                    l = len(c)
                    p_coal.append((l, value))
                    c.remove(p)
                    if len(c) == 0:
                        no_p_coal.append((0, 0))
                    else:
                        c_t = torch.Tensor(c)
                        value = self.value(c_t.t()[0].long(), c_t.t()[1].long(), method)
                        no_p_coal.append((l - 1, value))
            # print('e联盟: ', e_coal)
            # print('noe联盟: ', no_e_coal)
            shapelyvalue = 0
            for i in range(len(p_coal)):
                s = p_coal[i][0]
                p_payoff = p_coal[i][1] - no_p_coal[i][1]
                p_weight = math.factorial(s-1)*math.factorial(self.num_players-s)/math.factorial(self.num_players)
                shapelyvalue += p_payoff * p_weight
            shapely_value.append((p, shapelyvalue))
        # print('夏普利值：',shapely_value)
        # print(len(shapely_value))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        new_p = []
        for cur in shapely_value:
            new_p.append([cur[0][0], cur[0][1]])
        return new_p       

    def MTC(self, num_simulations, method='ori',pair=[]):
        shapley_values = np.zeros(self.num_players)
        for _ in range(num_simulations):
        # 生成随机排列的玩家列表
            players = np.random.permutation(self.num_players)
            # print(players, split)
            # 初始化联盟价值和玩家计数器
            coalition_value = 0
            player_count = 0
            if method == 'ori':
                tri1 = []
                tri2 = []
                for player in players:
                    # 计算当前联盟中添加玩家后的价值差异
                    if player < self.split:
                        tri1.append(self.p[player])
                    else:
                        tri2.append(self.p[player])

                    coalition_value_with_player = self.value(tri1, tri2, method)
                    # print(tri1, tri2)
                    # print(coalition_value_with_player)
                    marginal_contribution = coalition_value_with_player - coalition_value

                    # 计算当前玩家的 Shapley 值
                    shapley_values[player] += marginal_contribution / num_simulations

                    # 更新联盟价值和玩家计数器
                    coalition_value = coalition_value_with_player
            else:
                tri1 = []
                tri2 = []
                for player in players:
                    # 计算当前联盟中添加玩家后的价值差异
                    tri1.append(pair[player][0])
                    tri2.append(pair[player][1])

                    coalition_value_with_player = self.value(tri1, tri2, method)
                    # print(tri1, tri2)
                    # print(coalition_value_with_player)
                    marginal_contribution = coalition_value_with_player - coalition_value

                    # 计算当前玩家的 Shapley 值
                    shapley_values[player] += marginal_contribution / num_simulations

                    # 更新联盟价值和玩家计数器
                    coalition_value = coalition_value_with_player
                # player_count += 1
        res = torch.Tensor(shapley_values).argsort(descending=True)
        if method == 'ori':
            return shapley_values
        else:
            new_p = []
            for cur in res:
                new_p.append(pair[cur])
            return new_p

class LIME:
    def __init__(self, model, e1, e2, num_players, players, split, p, e_dict, r_dict, dataset, embed):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.dataset = dataset
        self.embed = embed

    def value(self, tri1, tri2):
        try:
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
            # print(node_size,adj_list, model.ent_embedding.weight.shape )
            proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            v_suf = -2

        return v_suf, proxy_e1, proxy_e2
    def sim_kernel(self, proxy_e1, proxy_e2):
        try:
            
            sim1 = F.cosine_similarity(proxy_e1[0], self.embed[e1], dim=0)
            sim2 = F.cosine_similarity(proxy_e2[0], self.embed[e2], dim=0)
            sim = (sim1 + sim2) / 2
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            return 0

        return sim
    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)
            if len(players) == self.num_players or len(players) == 0:
                continue
            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(self.p[player])
                else:
                    tri2.append(self.p[player])
            v,proxy_e1, proxy_e2 = self.value(tri1, tri2)
            sim = self.sim_kernel(proxy_e1, proxy_e2)
            
            if v == -2:
                continue
            pi.append(sim)
            mask.append(cur_mask)
            Y.append(v)
                    
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        I = torch.eye(Z.shape[1])
        # print(pi)
        # print(Z,Y,I)
        # exit(0)
        # pi = torch.Tensor(pi)
        # pi = torch.diag(pi)
        # print(np.array(pi) + 1)
        # print(Z)
        # print(Y)
        reg = LinearRegression().fit(Z,Y,np.array(pi) + 1)
        # print(reg.coef_)
        # exit(0)
        # pi = I
        res = reg.coef_
        # res = torch.mm(torch.inverse(torch.mm(torch.mm(Z.t(),pi),Z) + I), torch.mm(torch.mm(Z.t(),pi), Y.unsqueeze(1)))
        # res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        return res

class LLM:
    def __init__(self, model, e1, e2, num_players, players, split, p, e_dict, r_dict, dataset, embed):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.dataset = dataset
        self.embed = embed

    def value(self, tri1, tri2):
        try:
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
            # print(node_size,adj_list, model.ent_embedding.weight.shape )
            proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            v_suf = -2

        return v_suf, proxy_e1, proxy_e2
    def sim_kernel(self, proxy_e1, proxy_e2):
        try:
            
            sim1 = F.cosine_similarity(proxy_e1[0], self.embed[e1], dim=0)
            sim2 = F.cosine_similarity(proxy_e2[0], self.embed[e2], dim=0)
            sim = (sim1 + sim2) / 2
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            return 0

        return sim
    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)
            if len(players) == self.num_players or len(players) == 0:
                continue
            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(self.p[player])
                else:
                    tri2.append(self.p[player])
            v,proxy_e1, proxy_e2 = self.value(tri1, tri2)
            sim = self.sim_kernel(proxy_e1, proxy_e2)
            
            if v == -2:
                continue
            pi.append(sim)
            mask.append(cur_mask)
            Y.append(v)
                    
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        return Z, Y



class BLACK:
    def __init__(self, model, split, alpha):
        self.model = model
        self.split = split
        self.alpha = alpha
    
    def predict(self, X):
        tri = []
        # print(X)
        for x in X:
            tri1 = []
            tri2 = []
            for i in range(len(x)):
                if x[i] == 1:
                    if i < self.split:
                        tri1.append(i)
                    else:
                        tri2.append(i - self.split)
            tri.append((tri1, tri2))
        res = []
        # print(tri)
        for tri1, tri2 in tri:
            # print(tri1, tri2)
            sim = self.model.sim(tri1, tri2)
            if sim >= self.alpha:
                c = 1
            else:
                c = 0
            res.append(c)
        return np.array(res)
        
class LORE:
    def __init__(self, model, num_players, players, split, e1, e2, embed, lang):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed
        self.lang = lang

    def compute(self, sample_nums, alpha):
        data = []
        Y = []
        pi = []
        pred = []
        
        data.append([1] * (self.num_players + 1))
        X2E = []
        X2E.append([1] * (self.num_players))
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players + 1)
            cur_x = [0] * (self.num_players)
            for player in players:
                cur_mask[player] = 1
                cur_x[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.model.sim(tri1, tri2)
            if sim >= alpha:
                c = 1
            else:
                c = 0
            Y.append(sim)
            pred.append(c)
            cur_mask[self.num_players] = c
            data.append(cur_mask)
            X2E.append(cur_x)
        X2E = np.array(X2E)
        # print(pred)
        if max(pred) == 0 or min(pred) == 1:
            alpha = sum(Y) / len(Y)
            for j in range(len(Y)):
                if Y[j] > alpha:
                    data[j + 1][-1] = 1
                else:
                    data[j + 1][-1] = 0
        # print(data)
        feature = [str(i) for i in range(self.num_players)]
        feature.append('class')
        df = pd.DataFrame(data, columns=feature)
        dataset = prepare_EA_dataset(df, self.lang)
        blackbox = BLACK(self.model, self.split, alpha)
        explanation, infos = lore.explain(0, X2E, dataset, blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=True,
                                      continuous_function_estimation=False,
                                      returns_infos=True,
                                      path='', sep=';', log=False)
        if explanation == None:
            return None, None
        print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
        return explanation[0][1], explanation[0][0]['class']

class Anchor:
    def __init__(self, model, num_players, players, split, e1, e2, embed, lang):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed
        self.lang = lang

    def compute(self, sample_nums, alpha):
        mask = []
        Y = []
        pi = []
        pred = {}
        def encoder_fn(X):
            tri = []
            # print(X)
            for x in X:
                tri1 = []
                tri2 = []
                for i in range(len(x)):
                    if x[i] == 1:
                        if i < self.split:
                            tri1.append(i)
                        else:
                            tri2.append(i - self.split)
                tri.append((tri1, tri2))
            return tri
        def predict(tri):
            res = []
            # print(tri)
            for tri1, tri2 in tri:
                # print(tri1, tri2)
                sim = self.model.sim(tri1, tri2)
                if sim >= alpha:
                    c = 1
                else:
                    c = 0
                res.append(c)
            return np.array(res)
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.model.sim(tri1, tri2)
            if sim >= alpha:
                c = 1
            else:
                c = 0
            pred[tuple(cur_mask)] = c
            
            mask.append(cur_mask)
            Y.append(c)
                    
        Z = np.array(mask)
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=[0,1],
            feature_names=list(range(self.num_players)),
            train_data=Z,
            categorical_names={},
            encoder_fn=encoder_fn
        )
        x = np.array([[1] * (self.num_players)])
        
        explanation = self.explainer.explain_instance(x, predict)

        # 打印Anchor规则
        # print(explanation.names())
        return explanation.names()
class KernelSHAP:
    def __init__(self, model, e1, e2, num_players, players, split, p, e_dict, r_dict, dataset, embed):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.dataset = dataset
        self.embed = embed

    def value(self, tri1, tri2):
        try:
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
            # print(node_size,adj_list, model.ent_embedding.weight.shape )
            proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            v_suf = -2

        return v_suf, proxy_e1, proxy_e2
    def sim_kernel(self, tri1,tri2):
        if len(tri1) + len(tri2) == self.num_players:
            return 1
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        z = len(tri1) + len(tri2)
        # print(z, self.num_players)
        sim = (self.num_players - 1) / (comb(self.num_players, z) * z * (self.num_players - z))

        return sim

    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)
            if len(players) == self.num_players or len(players) == 0:
                continue
            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(self.p[player])
                else:
                    tri2.append(self.p[player])
            v, proxy_e1, proxy_e2 = self.value(tri1, tri2)
            sim = self.sim_kernel(tri1, tri2)
            
            if v == -2:
                continue
            pi.append(sim)
            mask.append(cur_mask)
            Y.append(v)
                    
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        I = torch.eye(Z.shape[1])
        # print(pi)
        # print(Z,Y,I)
        # exit(0)
        # pi = torch.Tensor(pi)
        # pi = torch.diag(pi)
        # print(np.array(pi) + 1)
        # print(Z)
        # print(Y)
        reg = LinearRegression().fit(Z,Y,np.array(pi) + 1)
        # print(reg.coef_)
        # exit(0)
        # pi = I
        res = reg.coef_
        # res = torch.mm(torch.inverse(torch.mm(torch.mm(Z.t(),pi),Z) + I), torch.mm(torch.mm(Z.t(),pi), Y.unsqueeze(1)))
        # res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        return res

class EAExplainer(torch.nn.Module):
    def __init__(self, model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, lang='zh', split=19388, splitr=0):
        super(EAExplainer, self).__init__()
        self.model_name = model_name
        self.split = split
        self.splitr = splitr
        self.G_dataset = G_dataset
        self.lang = lang
        self.conflict_r_pair = G_dataset.conflict_r_pair
        self.conflict_id = G_dataset.conflict_id
        self.dist = nn.PairwiseDistance(p=2)
        if lang == 'zh':
            self.embed = torch.load('../saved_model/embed.pt').cuda()
        elif lang == 'zh2':
            self.embed = torch.load('../saved_model/embed_zh2.pt').cuda()
        elif lang == 'ja':
            self.embed = torch.load('../saved_model/embed_ja.pt').cuda()
        elif lang == 'fr':
            self.embed = torch.load('../saved_model/embed_fr.pt').cuda()
        elif lang == 'w':
            self.embed = torch.load('../saved_model/embed_w.pt').cuda()
        elif lang == 'w2':
            self.embed = torch.load('../saved_model/embed_w2.pt').cuda()
        elif lang == 'y':
            self.embed = torch.load('../saved_model/embed_y.pt').cuda()
        self.embed.requires_grad = False
        self.r_embed = model.rel_embedding.weight
        self.get_r_map(lang)
        self.e_embed = model.ent_embedding.weight
        # print(self.r_embed.shape)
        self.e_sim =self.cosine_matrix(self.embed[:self.split], self.embed[self.split:])
        # self.r_embed.requires_grad = False
        # self.embed = torch.load('../saved_model/embed.pt').cuda()
        self.L = self.embed[list(range(split))]
        self.R = self.embed[list(range(split, self.embed.shape[0]))]
        # if model != None:
        self.base_model = model
        # self.base_model.eval()
        # for name, param in self.base_model.named_parameters():
            # param.requires_grad = False
        
        self.Lvec = Lvec
        self.Rvec = Rvec
        self.Lvec.requires_grad = False
        self.Rvec.requires_grad = False
        self.test_indices = test_indices
        self.test_pair = G_dataset.test_pair
        self.train_pair = G_dataset.train_pair
        self.model_pair = G_dataset.model_pair
        self.model_link = G_dataset.model_link
        self.train_link = G_dataset.train_link
        self.test_link = G_dataset.test_link
        # self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.evaluator = evaluator

    def proxy(self):
        ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size = self.G_dataset.reconstruct_test(self.G_dataset.kgs)
        adj = torch.sparse_coo_tensor(indices=adj_list, values=torch.ones_like(adj_list[0, :], dtype=torch.float),
                                      size=[node_size, node_size])
        adj = torch.sparse.softmax(adj, dim=1)
        res_embed = torch.sparse.mm(adj, self.embed)
        kg1_test_entities = self.G_dataset.test_pair[:, 0]
        kg2_test_entities = self.G_dataset.test_pair[:, 1]
        Lvec = res_embed[kg1_test_entities].cpu()
        Rvec = res_embed[kg2_test_entities].cpu()
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        self.evaluator.test_rank(Lvec, Rvec)

    def get_r_map(self, lang):
        '''
        self.r_sim_l =self.cosine_matrix(self.r_embed[:self.splitr], self.r_embed[self.splitr:])
        self.r_sim_r =self.cosine_matrix(self.r_embed[self.splitr:], self.r_embed[:self.splitr])
        rankl = (-self.r_sim_l).argsort()
        rankr = (-self.r_sim_r).argsort()
        self.r_map1 = {}
        self.r_map2 = {}
        for i in range(rankl.shape[0]):
            self.r_map1[i] = rankl[i][0] + self.splitr
            print(self.G_dataset.r_dict[i], self.G_dataset.r_dict[int(rankl[i][0] + self.splitr)])
        for i in range(rankr.shape[0]):
            self.r_map2[i + self.splitr] = rankr[i][0]
            print(self.G_dataset.r_dict[i + self.splitr], self.G_dataset.r_dict[int(rankr[i][0])])
        exit(0)
        
        self.r_map1 = {}
        self.r_map2 = {}
        
        
        for i in range(self.splitr):
            cur1 = self.G_dataset.r_dict[i]
            for j in range(self.splitr, len(self.G_dataset.r_dict)):
                cur2 = self.G_dataset.r_dict[j]
                if cur1.split('/')[-1] == cur2.split('/')[-1]:
                    self.r_map1[self.G_dataset.id_r[cur1]] = self.G_dataset.id_r[cur2]
                    self.r_map2[self.G_dataset.id_r[cur2]] = self.G_dataset.id_r[cur1]
                    # print(cur1, cur2)
        '''
        if lang == 'de':
            self.r_map1 = {}
            self.r_map2 = {}
            for i in range(self.splitr):
                self.r_map1[i] = i
                self.r_map2[i] = i
        elif lang == 'y':
            self.r_map1 = {}
            self.r_map2 = {}
            
            '''
            pair1 = set()
            pair2 = set()
            
            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]
                cur_sim = 0
                ans = None
                for j in range(self.splitr, len(self.G_dataset.r_dict)):
                    cur2 = self.G_dataset.r_dict[j]
                    # if cur1.split('/')[-1] == cur2:
                    if difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio() > cur_sim:
                        cur_sim = difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio()
                        ans = cur2
                        # self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        # self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                pair1.add((cur1, ans))
            
            for i in range(self.splitr, len(self.G_dataset.r_dict)):
                cur2 = self.G_dataset.r_dict[i]
                cur_sim = 0
                ans = None
                for j in range(self.splitr):
                    cur1 = self.G_dataset.r_dict[j]
                    # if cur1.split('/')[-1] == cur2:
                    if difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio() > cur_sim:
                        cur_sim = difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio()
                        ans = cur1
                        # self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        # self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                pair2.add((ans, cur2))

            pair = pair1 & pair2

            for p in pair:
                cur1 = p[0]
                cur2 = p[1]
                self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
            '''

            with open('../datasets/y-en_f/rel_links') as f:
                lines = f.readlines()
                for line in lines:
                    cur = line.strip().split(' ')
                    self.r_map1[int(cur[0])] = int(cur[1])
                    self.r_map2[int(cur[1])] = int(cur[0])
            

            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]            
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None
        elif lang == 'w':
            self.r_map1 = {}
            self.r_map2 = {}
            with open('../datasets/w-en_f/rel_links') as f:
                lines = f.readlines()
                for line in lines:
                    cur = line.strip().split('\t')
                    self.r_map1[int(cur[0])] = int(cur[1])
                    self.r_map2[int(cur[1])] = int(cur[0])
            

            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]            
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None
        else:
            self.r_map1 = {}
            self.r_map2 = {}
            
            
            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]
                for j in range(self.splitr, len(self.G_dataset.r_dict)):
                    cur2 = self.G_dataset.r_dict[j]
                    if cur1.split('/')[-1] == cur2.split('/')[-1]:
                        self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None


    def explain_EA(self, method, thred, num,  version = '', lang='zh'):
        # print(version)
        if method == 'EG':
            with open('../datasets/' + lang + '-en_f/base/exp_ours' + str(version), 'w') as f:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    # self.explain_ours2(gid1, gid2)
                    # if version == 1:
                    tri1, tri2, pair = self.explain_ours4(gid1, gid2)
                    if version == 2:
                        exp_tri1, exp_tri2 = self.explain_ours_two(gid1, gid2)
                        tri1 = set(tri1 + exp_tri1)
                        tri2 = set(tri2 + exp_tri2)
                    
                    for cur in tri1[:3]:
                        f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    for cur in tri2[:3]:
                        f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    f.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\n')
            if version == 1:
                self.get_test_file_mask('../datasets/' + lang + '-en_f/base/exp_ours' + str(version), str(version))
            else:
                self.get_test_file_mask_two('../datasets/' + lang + '-en_f/base/exp_ours' + str(version), str(version))
        
        elif method == 'shapley':
            with open('../datasets/' + lang + '-en_f/base/exp_shapley', 'w') as f:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    if version == 1:
                        tri = self.explain_shapely(gid1, gid2)
                    else:
                        tri = self.explain_shapely_two(gid1, gid2)
                    for cur in tri:
                        f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            if version == 1:
                self.get_test_file_mask('../datasets/' + lang + '-en_f/base/exp_shapley', str(version), method)
            else:
                self.get_test_file_mask_two('../datasets/' + lang + '-en_f/base/exp_shapley', str(version), method)
        
        elif method == 'lime':
            with open('../datasets/' + lang + '-en_f/base/exp_lime', 'w') as f:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    # tri1, tri2 = self.explain_lime(gid1, gid2)
                    if version == 1:
                        tri = self.explain_lime(gid1, gid2)
                    else:
                        tri = self.explain_lime_two(gid1, gid2)
                    for cur in tri:
                        f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            if version == 1:
                self.get_test_file_mask('../datasets/' + lang + '-en_f/base/exp_lime', str(version), method)
            else:
                self.get_test_file_mask_two('../datasets/' + lang + '-en_f/base/exp_lime', str(version), method)
        elif method == 'llm':
            with open('../datasets/' + lang + '-en_f/base/exp_llm_per', 'w') as f, open('../datasets/' + lang + '-en_f/base/exp_llm_per_feature', 'w') as f1:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    # tri1, tri2 = self.explain_lime(gid1, gid2)
                    Z, Y, p1, p2 = self.explain_llm(gid1, gid2)
                    feature_name =''
                    for i in range(len(p1) + len(p2) - 1):
                        feature_name += 'f' + str(i) +','
                    feature_name += 'f' + str(len(p1) + len(p2) - 1)
                    context = 'We have a machine learning model that predicts based on {} features: [{}]. The model has been trained on a dataset and has made the following predictions\n'.format(len(p1) + len(p2), feature_name)
                    dataset = 'Dataset:\n'
                    for i in range(len(Z)):
                        Input = 'Input:'
                        for j in range(len(Z[i])):
                            Input += 'f' + str(j) + '=' + str(int(Z[i][j])) +', '
                        Output = 'Output: {}'.format(Y[i])
                        dataset += Input + '\n'
                        dataset += Output + '\n'
                    Qestions = 'Based on the model\'s predictions and the given dataset, please rank features according to their importance in determining the model\'s prediction?\n'
                    # Instructions = 'Instructions: Think about the question. After explaining your reasoning, provide your answer as features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line.\n'
                    Instructions = 'Instructions: Only output the ranking feature names.\n'
                    prompt = context + dataset + Qestions + Instructions + '-------------------------\n'
                    f.write(prompt + '\n')                
                    for i in range(len(p1)):
                        f1.write(str(p1[i][0]) + ',' + str(p1[i][1]) + ',' + str(p1[i][2]) + '\t')
                    for i in range(len(p2)):
                        f1.write(str(p2[i][0]) + ',' + str(p2[i][1]) + ',' + str(p2[i][2]) + '\t')
                    f1.write('\n')
            
        elif method == 'anchor':
            with open('../datasets/' + lang + '-en_f/base/exp_anchor', 'w') as f:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    # tri1, tri2 = self.explain_lime(gid1, gid2)
                    if version == 1:
                        rule = self.explain_anchor(gid1, gid2)
                    else:
                        rule = self.explain_anchor_two(gid1, gid2)
    
                    f.write(str(rule) + '\n')
            print('Save rules to exp_anchor in the corresponding dataset directory')
        elif method == 'lore':
            with open('../datasets/' + lang + '-en_f/base/exp_lore', 'w') as f:
                for i in trange(len(self.test_indices)):
                    gid1, gid2 = self.test_indices[i]
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    # tri1, tri2 = self.explain_lime(gid1, gid2)
                    if version == 1:
                        rule = self.explain_lore(gid1, gid2)
                    else:
                        rule = self.explain_lore_two(gid1, gid2)
                    f.write(str(rule) + '\n')
            print('Save rules to exp_lore in the corresponding dataset directory')
        elif method == 'repair':
            K = 100
            r1_func, r1_func_r, r2_func, r2_func_r = self.get_r_func()
            node = {}
            ground = {}
            cur_link = {}
            cur_pair = set()
            for p in self.model_pair:
                cur_link[int(p[0])] = int(p[1])
                cur_pair.add((int(p[0]), int(p[1])))
            for cur in self.test_pair:
                ground[str(cur[0])] = str(cur[1])
                ground[str(cur[1])] = str(cur[0])
            node_set = defaultdict(float)
            
            kg2 = set()
            all_kg1 = set()
            for p in self.test_pair:
                kg2.add(p[1])
                all_kg1.add(p[0])
            ans_pair = set()
            for cur in self.test_link:
                ans_pair.add((int(cur), int(self.test_link[str(cur)])))

            for cur in cur_link:
                gid1 = cur
                gid2 = cur_link[cur]
                pair, score = self.get_pair_score(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                if len(pair) > 0:
                    node[str(gid1) + '\t' + str(gid2)] = pair
                    node_set[str(gid1) + '\t' + str(gid2)] = score
            # T1 = time.clock()
            c_set, new_model_pair, count1 = self.conflict_count(self.model_pair)
            kg1, _, cur_pair = self.conflict_solve(c_set, node_set, new_model_pair, kg2, count1, ground)
            last_len = 0
            while(len(kg1) > 0 and len(kg1) != last_len):
                last_len = len(kg1)
                cur_link = {}
                cur_link_r = {}
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    cur_link_r[int(p[1])] = int(p[0])
                new_cur_pair, kg1 = self.adjust(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K)
                # print(len(cur_pair))
                print('current acc: {}'.format(len(cur_pair & ans_pair) / len(ans_pair)))
            
            # find low confidence conflict
            print('start low confidence conflict solving')
            last_len1 = None
            while True:
                cur_link = {}
                cur_link_r = {}
                kg1 = set()
                # print(len(cur_pair & ans_pair) / len(ans_pair))
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    kg1.add(int(p[0]))
                    cur_link_r[int(p[1])] = int(p[0])
                kg1 = all_kg1 - kg1
                for cur in cur_link:
                    gid1 = cur
                    gid2 = cur_link[cur]
                    _, score = self.find_low_confidence(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    if score == 0:
                        kg1.add(gid1)
                        cur_pair.remove((gid1, gid2))
                if last_len1 != None and len(kg1) >= last_len1:
                    break
                else:
                    last_len1 = len(kg1)
                # print(len(cur_pair & ans_pair) / len(ans_pair))
                while(len(kg1) > 0):
                    cur_link = {}
                    cur_link_r = {}
                    # print(len(cur_pair))
                    # print(len(kg1))
                    for p in cur_pair:
                        cur_link[int(p[0])] = int(p[1])
                        cur_link_r[int(p[1])] = int(p[0])
                    last_len = len(kg1)
                    # _, kg1 = self.adjust_conflict(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100, conflict_link)
                    _, kg1 = self.adjust_no_explain(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, 1)
                    if len(kg1) >= last_len:
                        break
                    # print(len(cur_pair))
                    # print(len(delete_pair & ans_pair), len(delete_pair) )
                    print('current acc: {}'.format(len(cur_pair & ans_pair) / len(ans_pair)))
            
            print('After low confidence conflict solving :', len(cur_pair & ans_pair) / len(ans_pair))
            
            solve_kg1 = set()
            solve_kg2 = set()
            for p in cur_pair:
                solve_kg1.add(p[0])
                solve_kg2.add(p[1])
            # print(len(solve_kg1), len(solve_kg2))
            left_kg1 = all_kg1 - solve_kg1
            left_kg2 = kg2 - solve_kg2
            # print(len(left_kg1), len(left_kg2))
            new_pair, _ = self.re_align(left_kg1, left_kg2)
            cur_pair |= new_pair
            print('final acc: ', len(cur_pair & ans_pair) / len(ans_pair))
            with open('repair_pair', 'w') as f:
                for p in cur_pair:
                    f.write(str(p[0]) + '\t' + str(p[1]) + '\n')
            # T2 = time.clock()
            # print(T2 - T1)
           
    def conflict_count(self, cur_model_pair):
        count = 0
        count1 = 0
        conflict_set = defaultdict(set)
        new_model_pair = {}
        for pair in cur_model_pair:
            conflict_set[pair[1]].add(pair[0])
            new_model_pair[pair[0]] = pair[1]
        
        for ent in conflict_set:
            if len(conflict_set[ent]) > 1:
                judge = 0
                # print(self.G_dataset.ent_dict[ent])
                # for cur in self.G_dataset.gid[ent]:
                    # self.read_triple_name(cur)
                # print('************************')
                count += 1
                for e in conflict_set[ent]:
                    new_model_pair[e] = None
                    if self.test_link[str(e)] == str(ent):
                        count1 += 1
                        judge = 1
        # print('conflict_num:', count, count1)
        # exit(0)
        return conflict_set, new_model_pair, count1

    def conflict_solve(self, c_set, node_set, new_model_pair, kg2, count1, ground):
        count = 0
        kg1 = set()
        cur_kg2 = set()
        cur_kg1 = set()
        cur_pair = set()
        for ent in c_set:
            cur_kg2.add(ent)
            if len(c_set[ent]) > 1:
                tmp = 0
                judge = 0
                score = 0
                max_e = None

                for e in c_set[ent]:
                    cur_score = node_set[str(e) + '\t' + str(ent)] # +  0.5 * self.e_sim[e, ent - self.split]
                    # cur_score =    self.e_sim[e, ent - self.split]
                    if cur_score >= score:
                        score = cur_score
                        max_e = e
                new_model_pair[max_e] = ent
                cur_pair.add((max_e, ent))
                for e in c_set[ent]:
                    if e != max_e:
                        kg1.add(e)
                    cur_kg1.add(e)
                if max_e == int(ground[str(ent)]):
                    count += 1
            else:
                for e in c_set[ent]:
                    cur_pair.add((e, ent))

        new_kg2 = kg2 - cur_kg2
        
        # print(len(kg1), len(new_kg2), len(cur_kg1), len(cur_kg2), count - count1)
        return kg1, new_kg2, cur_pair

    def re_align(self, kg1, kg2):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        # print(kg1_embed.shape, kg1)
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        rank = (-sim).argsort()
        new_pair = set()
        ans_pair = set()
        for cur in kg1:
            ans_pair.add((int(cur), int(self.test_link[str(cur)])))
        for i in range(rank.shape[0]):
            new_pair.add((kg1[i], kg2[rank[i][0]]))
        # print(len(new_pair & set(ans_pair)))
        # print(len(new_pair & set(ans_pair)) / len(ans_pair))
        return new_pair, set(kg2)


    def adjust_no_explain(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_type):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        

        for i in range(len(kg1)):
            candidate = self.candidate_ent(kg1[i], cur_link)
            # kg1_embed = self.embed[[kg1[i]]]
            # kg2_embed = self.embed[candidate]
            # cur_sim = self.cosine_matrix(kg1_embed, kg2_embed)
            # rank = (-cur_sim).argsort()
            # max_e = None
            score = []
            for ent in candidate:
                _, cur_score = self.get_pair_score5(kg1[i], ent, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                cur_score += 0.5 * self.e_sim[kg1[i], ent - self.split]
                score.append(cur_score)
            values, rank = torch.Tensor(score).sort(descending=True)
            for j in range(min(K, len(rank))):
                # print(len(candidate), rank[0][j], len(rank[0]))
                # print(candidate[int(rank[0][j])])
                e2 = int(candidate[int(rank[j])])
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 in kg2:
                    if e2 not in cur_link_r:   
                        cur_pair.add((kg1[i], e2))
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
                    else:
                        if rule_type == 0:
                            _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        elif rule_type == 1:
                            cur_score = values[j]
                            _, other = self.get_pair_score5(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            other += 0.5 * self.e_sim[cur_link_r[e2], e2 - self.split]
                        elif rule_type == 2:
                            _, cur_score = self.get_pair_score4(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score4(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        if other < cur_score:
                            # print(kg1[i], cur_link_r[e2])
                            cur_pair.remove((cur_link_r[e2], e2))
                            cur_pair.add((kg1[i], e2))
                            new_kg1.add(cur_link_r[e2])
                            cur_link_r[e2] = kg1[i]
                            cur_link[kg1[i]] = e2
                            
                            break
            # new_pair.add((kg1[i], e2))
        # print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

    def candidate_ent(self, e1, cur_link):
        candidate = set()
        for cur in self.G_dataset.gid[e1]:
            if cur[0] != int(e1) and cur[0] in cur_link:
                for t in self.G_dataset.gid[cur_link[cur[0]]]:
                    if t[0] ==  cur_link[cur[0]]:
                        candidate.add(int(t[2]))
                    else:
                        candidate.add(int(t[0]))
            if cur[0] != int(e1) and str(cur[0]) in self.train_link:
                for t in self.G_dataset.gid[int(self.train_link[str(cur[0])])]:
                    if t[0] ==  int(self.train_link[str(cur[0])]):
                        candidate.add(int(t[2]))
                    else:
                        candidate.add(int(t[0]))
            else:
                if cur[2] in cur_link:
                    for t in self.G_dataset.gid[cur_link[cur[2]]]:
                        if t[0] ==  cur_link[cur[2]]:
                            candidate.add(int(t[2]))
                        else:
                            candidate.add(int(t[0]))
                if str(cur[2]) in self.train_link:
                    for t in self.G_dataset.gid[int(self.train_link[str(cur[2])])]:
                        if t[0] ==  int(self.train_link[str(cur[2])]):
                            candidate.add(int(t[2]))
                        else:
                            candidate.add(int(t[0]))
        candidate = list(candidate)
        candidate.sort()
        return candidate
    
    def find_low_confidence(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        else:
            return set(), 1

    def adjust(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        
        rank = (-sim).argsort()
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        # print(len(kg1))
        for i in range(rank.shape[0]):
            judge = 0
            for j in range(K):
                e2 = kg2[rank[i][j]]
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    judge = 1
                    break
                else:
                    _, cur_score = self.get_pair_score5(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                    # cur_score =  self.e_sim[kg1[i], e2- self.split]
                    _, other = self.get_pair_score5(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                    # other = self.e_sim[cur_link_r[e2], e2- self.split]
                    # print(other, cur_score)
                    if other < cur_score:
                        # print(other, cur_score)
                        # print(kg1[i], cur_link_r[e2])
                        cur_pair.remove((cur_link_r[e2], e2))
                        cur_pair.add((kg1[i], e2))
                        new_kg1.add(cur_link_r[e2])
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        judge = 1
                        break
            if judge == 0:
                new_kg1.add(kg1[i])

            # new_pair.add((kg1[i], e2))
        # print(len(new_kg1))
        # print(new_kg1)
        kg1_set = set()
        kg2_set = set()
        for p in cur_pair:
            kg1_set.add(p[0])
            kg2_set.add(p[1])
        # print(len(kg1_set), len(kg2_set))
        return new_pair, new_kg1
        
    def find_could_not_do(self):
        cur_link = {}
        cur_pair = set()
        kg2 = []
        for p in self.test_pair:
            cur_link[int(p[0])] = int(p[1])
            cur_pair.add((int(p[0]), int(p[1])))
            kg2.append(int(p[1]))
        kg2.sort()
        count = 0
        score_matrix = []
        r1_func, r1_func_r, r2_func, r2_func_r = self.get_r_func()
        for i in trange(len(self.test_pair)):  
            gid1, gid2 = self.test_pair[i]
            e1 = int(gid1)
            e2 = int(gid2)
            score = []
            for e3 in kg2:
                _, cur_score = self.get_pair_score6(e1, e3, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                score.append(cur_score)
            score_matrix.append(score)
            '''
            values, rank = torch.Tensor(score).sort(descending=True)  
            if len(rank) > 0:
                e3 = int(candidate[int(rank[0])])      
                if e2 != e3 and values[0] != 0.5:
                    count += 1
                    print(self.G_dataset.ent_dict[e1], self.G_dataset.ent_dict[e2], self.G_dataset.ent_dict[e3])
            '''
        m = np.array(score_matrix)
        np.save('score_zh.npy',m)
    def estimate(self, file):
        m = np.load(file)
        m = torch.Tensor(m)
        pre1 = (-m).argsort()
        count = 0
        for i in range(len(pre1)):
            if i == pre1[i][0]:
                count += 1
        print(count / 10500)

    def get_pair_score(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), sigmoid(0)
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1] + 1])
            for cur in tri2:
                r2.append(self.r_embed[cur[1] + 1])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            
            for pr in pair_r:
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                # r_score = max(r_score, cur_score)
                r_score += cur_score
            # score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        return pair_node, sigmoid(score) 

    def get_pair_score1(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            neigh_r1 = set()
            neigh_r2 = set()
            r_score = 0
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                if tri2[pr[1]][1] in self.r_map2:
                    map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                    neigh_r1.add(map_r2)
                if tri1[pr[0]][1] in self.r_map1:
                    map_r1 = int(self.r_map1[tri1[pr[0]][1]])
                    neigh_r2.add(map_r1)


                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            for pr in pair_r:
                for cur in neigh_r1:
                    if ((tri1[pr[0]][1], cur) in self.conflict_r_pair or (cur, tri1[pr[0]][1]) in self.conflict_r_pair):
                        r_score = 0
                for cur in neigh_r2:
                    if ((tri2[pr[1]][1], cur) in self.conflict_r_pair or (cur, tri2[pr[1]][1]) in self.conflict_r_pair):
                        r_score = 0
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            

        return pair_node, score   

    def get_pair_score5(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), sigmoid(0)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            pair_r_d = set() 
            for pr in pair_r:
                direct1 = 0
                direct2 = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    direct1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    direct2 = 1
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add((map_r2, direct2))
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add((map_r1,direct1))
                pair_r_d.add(((tri1[pr[0]][1], direct1), (tri2[pr[1]][1], direct2)))

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                        r_score = 0
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        r_score = 0
            
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            
            if score < 0:
                return self.get_pair_score_high(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
        return pair_node, sigmoid(score)
    def get_pair_score6(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), sigmoid(0)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            pair_r_d = set() 
            for pr in pair_r:
                direct1 = 0
                direct2 = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    direct1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    direct2 = 1
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add((map_r2, direct2))
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add((map_r1,direct1))
                pair_r_d.add(((tri1[pr[0]][1], direct1), (tri2[pr[1]][1], direct2)))

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                        r_score = 0
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        r_score = 0
            
            score += r_score 
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            
            if score < 0:
                return self.get_pair_score_high(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
        return pair_node, sigmoid(score)  
    def pattern_process(self, e, l=2, mode=0):
        p = []
        if l == 1:
            p_embed = torch.zeros(len(self.G_dataset.gid[e]), self.embed.shape[1] + self.r_embed.shape[1]) 
        else:
            p_embed = torch.zeros(len(self.G_dataset.pattern[e]), self.embed.shape[1] + self.r_embed.shape[1]) 
        i = 0
        if l == 2:
            for cur in self.G_dataset.pattern[e]:
                p.append(cur)

                if len(cur) == 3:
                    if cur[0] == e1:
                        p_embed[i] = torch.cat((self.embed[cur[2]], self.r_embed[cur[1] + 1]), dim=0)
                    else:
                        p_embed[i] = torch.cat((self.embed[cur[0]], self.r_embed[cur[1] + 1]), dim=0)
                else:
                    if cur[0][0] == e1:
                        if cur[0][2] == cur[1][0]:
                            p_embed[i] = torch.cat(((self.embed[cur[0][2]] + self.embed[cur[1][2]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                        else:
                            p_embed[i] = torch.cat(((self.embed[cur[0][2]] + self.embed[cur[1][0]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1]] + 1) / 2), dim=0)
                    else:
                        if cur[0][0] == cur[1][0]:
                            p_embed[i] = torch.cat(((self.embed[cur[0][0]] + self.embed[cur[1][2]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                        else:
                            p_embed[i] = torch.cat(((self.embed[cur[0][0]] + self.embed[cur[1][0]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                i += 1
        else:
            if mode == 0:
                for cur in self.G_dataset.gid[e]:
                    p.append(cur)
                    if cur[0] == e1:
                        p_embed[i] = torch.cat((self.embed[cur[2]], self.r_embed[cur[1] + 1]), dim=0)
                    else:
                        p_embed[i] = torch.cat((self.embed[cur[0]], self.r_embed[cur[1] + 1]), dim=0)

                    i += 1
            else:
                for cur in self.G_dataset.gid[e]:
                    p.append(cur)
                    if cur[0] == e1:
                        p_embed[i] = self.embed[cur[2]] + self.r_embed[cur[1] + 1]
                    else:
                        p_embed[i] = self.embed[cur[0]] + self.r_embed[cur[1] + 1]

                    i += 1
        return p, p_embed

    def get_pair_score_high(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        exp_tri1 = []
        exp_tri2 = []
        cur_link = self.model_link
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        score = 0
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh21:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh21:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        
        r_pair = []
        pair = list(pair)
        r_score = 0
        for k in range(len(pair)):
            p = pair[k]
            # print(self.search_2_hop_tri(e1, p[0]))
            two_hop_list = self.search_2_hop_tri1(e1, p[0])
            
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            index = 0
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            
            for i in range(len(pair_r)):
                pr = pair_r[i]
                cur_score = 1
                cur_score1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                if two_hop[pr[0]][0][0] == e1:
                    cur_score1 = min(cur_score1,r1_func_r[str(two_hop[pr[0]][0][1])])
                else:
                    cur_score1 = min(cur_score1,r1_func[str(two_hop[pr[0]][0][1])])
                if two_hop[pr[0]][1][0] == p[0]:
                    cur_score1 = r1_func_r[str(two_hop[pr[0]][1][1])] * cur_score1
                else:
                    cur_score1 = r1_func[str(two_hop[pr[0]][1][1])] * cur_score1

                # r_score = max(r_score, cur_score)
                
            r_score += cur_score * float(self.e_sim[p[0]][p[1] - self.split])

                
        pair = set()
        for cur in neigh11:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh11:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list = self.search_2_hop_tri1(e2, p[1])
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]), dim=0))
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []
            for i in range(len(pair_r)):
                cur_score = 1
                cur_score1 = 1
                pr = pair_r[i]
                
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])

                if two_hop[pr[1]][0][0] == e2:
                    cur_score1 = min(cur_score1,r2_func_r[str(two_hop[pr[1]][0][1])])
                else:
                    cur_score1 = min(cur_score1,r2_func[str(two_hop[pr[1]][0][1])])
                if two_hop[pr[1]][1][0] == p[1]:
                    cur_score1 = r2_func_r[str(two_hop[pr[1]][1][1])] * cur_score1
                else:
                    cur_score1 = r2_func[str(two_hop[pr[1]][1][1])] * cur_score1
                # r_score = max(r_score, cur_score)
            r_score += cur_score * float(self.e_sim[p[0]][p[1] - self.split])
        if r_score == 0:
            pair = set()
            for cur in neigh12:
                if str(cur) in cur_link:
                    if int(cur_link[str(cur)]) in neigh22:
                        pair.add((cur, int(cur_link[str(cur)])))
            for cur in neigh12:
                if str(cur) in self.train_link:
                    if int(self.train_link[str(cur)]) in neigh22:
                        pair.add((cur, int(self.train_link[str(cur)])))
            pair = list(pair)
            for k in range(len(pair)):
                p = pair[k]
                r_score += 0.0001 * float(self.e_sim[p[0]][p[1] - self.split])
        else:
            return  sigmoid(0.001 * r_score)
            
        return sigmoid(0.001 * r_score)

    def change_map(self, tri1, tri2):
        e_dict = {}
        r_dict = {}
        i = 0
        j = 0
        l = 0
        kg1_index = set()
        kg2_index = set()
        tri = tri1 | tri2
        for cur in tri:
            if cur[0] not in e_dict:
                e_dict[cur[0]] = i
                if l >= len(tri1):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[2] not in e_dict:
                e_dict[cur[2]] = i
                if l >= len(tri1):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[1] not in r_dict:
                r_dict[cur[1]] = j
                j += 1
            
            l += 1
        new_tri = []
        '''
        for cur in tri:
            new_tri.add((e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]))
            new_tri.add((e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]))
        '''
        tri1 = []
        tri2 = []
        for cur in tri:
            if e_dict[cur[0]] in kg1_index:
                tri1.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            else:
                tri2.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            new_tri.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
        for cur in tri:
            new_tri.append([e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]])

        e_dict_r = {}
        r_dict_r = {}
        for cur in e_dict:
            # print(cur, e_dict[cur], self.G_dataset.ent_dict[cur])
            e_dict_r[e_dict[cur]] = cur
        for cur in r_dict:
            # print(cur, r_dict[cur], self.G_dataset.r_dict[cur])
            r_dict_r[r_dict[cur]] = cur
        return torch.Tensor(new_tri).long().cuda(), e_dict, r_dict, list(kg1_index), list(kg2_index), tri1, tri2, e_dict_r, r_dict_r

    
    def extract_subgraph(self, e1, e2, l=1):
        if l == 1:
            tri = self.G_dataset.gid[e1] + self.G_dataset.gid[e2]
        # print(tri)
        e_dict = {}
        r_dict = {}
        i = 0
        j = 0
        l = 0
        kg1_index = set()
        kg2_index = set()
        for cur in tri:
            if cur[0] not in e_dict:
                e_dict[cur[0]] = i
                if l >= len(self.G_dataset.gid[e1]):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[2] not in e_dict:
                e_dict[cur[2]] = i
                if l >= len(self.G_dataset.gid[e1]):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[1] not in r_dict:
                r_dict[cur[1]] = j
                j += 1
            
            l += 1
        new_tri = []
        '''
        for cur in tri:
            new_tri.add((e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]))
            new_tri.add((e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]))
        '''
        tri1 = []
        tri2 = []
        for cur in tri:
            if e_dict[cur[0]] in kg1_index:
                tri1.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            else:
                tri2.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            new_tri.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
        for cur in tri:
            new_tri.append([e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]])

        e_dict_r = {}
        r_dict_r = {}
        for cur in e_dict:
            # print(cur, e_dict[cur], self.G_dataset.ent_dict[cur])
            e_dict_r[e_dict[cur]] = cur
        for cur in r_dict:
            # print(cur, r_dict[cur], self.G_dataset.r_dict[cur])
            r_dict_r[r_dict[cur]] = cur
        return torch.Tensor(new_tri).long(), e_dict, r_dict, list(kg1_index), list(kg2_index), tri1, tri2, e_dict_r, r_dict_r

    def Trans_Process(self, e):
        i = 0
        p_embed = torch.zeros(len(self.G_dataset.gid[e]), self.embed.shape[1] + self.r_embed.shape[1]) 
        for cur in self.G_dataset.gid[e]:
            '''
            if cur[0] == e:
                p_embed[i] = self.embed[cur[2]] +  self.r_embed[cur[1]]
            else:
                p_embed[i] = self.embed[cur[0]] -  self.r_embed[cur[1]]
            '''
            # print(self.embed[cur[2]].shape, self.r_embed[cur[1]].shape)
            if cur[0] == e:
                p_embed[i] = torch.cat((self.embed[cur[2]] ,  self.r_embed[cur[1] + 1]), dim=0)
            else:
                p_embed[i] = torch.cat((self.embed[cur[0]] ,  self.r_embed[int(self.r_embed.shape[0] / 2) + cur[1] + 1]), dim=0)
            i += 1
        return p_embed

    def r_func_filter(self, file, file_out):
        local_rule = {}
        train = {}
        test = {}
        ground = {}
        align = {}
        align = {}
        for i in range(len(self.G_dataset.train_pair)):
            train[self.G_dataset.ent_dict[self.G_dataset.train_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.train_pair[i][1]]
            align[self.G_dataset.ent_dict[self.G_dataset.train_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.train_pair[i][1]]
            align[self.G_dataset.ent_dict[self.G_dataset.train_pair[i][1]]] = self.G_dataset.ent_dict[self.G_dataset.train_pair[i][0]]
        for i in range(len(self.G_dataset.model_pair)):
            test[self.G_dataset.ent_dict[self.G_dataset.model_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.model_pair[i][1]]
            align[self.G_dataset.ent_dict[self.G_dataset.model_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.model_pair[i][1]]
            align[self.G_dataset.ent_dict[self.G_dataset.model_pair[i][1]]] = self.G_dataset.ent_dict[self.G_dataset.model_pair[i][0]]
        r1_func, r1_func_r, r2_func, r2_func_r = self.get_r_func()
        for i in range(len(self.G_dataset.test_pair)):
            ground[self.G_dataset.ent_dict[self.G_dataset.test_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.test_pair[i][1]]
        '''
        sim = torch.mm(self.r_embed[1:1701], self.r_embed[1701:3025].t())
        r, _ = read_link('../datasets/zh-en_f/base/rel_dict')
        pre1 = (-sim).argsort()
        for i in range(pre1.shape[0]):
            print(r[str(i)], r[str(int(pre1[i][0]) + 1700)])
        exit(0)
        '''
        with open(file) as f,  open(file_out, 'w') as f1:
            lines = f.readlines()
            align_entities = ''
            rule = []
            ent = {}
            r = {}
            local_rule = {}
            score_r = 0
            fn = 0
            tn = 0
            _, r_dict = read_link('../datasets/zh-en_f/base/rel_dict')
            begin = 1
            for i in trange(len(lines)):
                cur = lines[i].strip().split('\t')
                # print(len(cur))
                if len(cur) == 1 and cur[0][0] == '*':
                    if score_r <= 0.4 and judge == 0:
                        fn += 1
                    elif score_r <= 0.4 and judge == 1:
                        tn += 1
                    
                    rule = []
                    begin = 1
                    score_r = 0
                elif len(cur) == 1 and cur[0][0] == '-':
                    score = 0
                    if e1 == rule[0][0]:
                        score = r1_func_r[r_dict[rule[0][1]]]
                        e3 = rule[0][2]
                    else:
                        score = r1_func[r_dict[rule[0][1]]]
                        e3 = rule[0][0]
                    if e2 == rule[1][0]:
                        score = min(score, r2_func_r[r_dict[rule[1][1]]])
                        e4 = rule[1][2]
                    else:
                        score = min(r2_func[r_dict[rule[1][1]]], score)
                        e4 = rule[1][0]
                    for cur in rule:
                        f1.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')
                    
                    f1.write(str(score) +  '\n')
                    if score >= 1:
                        if e3 in align:
                            if align[e3] != e4 and judge == 1:
                                print(rule[0])
                                print(rule[1])
                                print(e3, align[e3])
                                tn += 1
                            elif align[e3] != e4 and judge == 0:
                                fn += 1
                    score_r += score
                    f1.write('------------------------------------\n')
                    rule = []
                elif len(cur) == 2 and begin == 1:
                    score_r = 0
                    begin = 0
                    ent = {}
                    r = {}
                    count = 0
                    v = 'a'
                    judge = 1
                    if ground[cur[0]] != cur[1]:
                        judge = 0
                    align_entities = [cur[0] , cur[1], judge]
                    f1.write('*******************\n')
                    f1.write(cur[0] + '\t' + cur[1] + '\t' + str(judge) + '\n')
                    e1 = cur[0]
                    e2 = cur[1]
                elif len(cur) == 3:
                    rule.append((cur[0], cur[1], cur[2]))
        print(tn, fn)

    def get_proxy_model_ori_Trans(self, e1, e2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.extract_subgraph(e1, e2, 1)


        p_embed1 = self.Trans_Process(e1)
        p_embed2 = self.Trans_Process(e2)
        model = Proxy(p_embed1, p_embed2)
        
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph


    def get_proxy_model(self, e1, e2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index = self.extract_subgraph(e1, e2, 1)
        model = CompGCNLayer(self.embed.shape[1], self.r_embed.shape[1], len(r_dict))
        ent_embed = torch.zeros(len(e_dict), self.embed.shape[1]).cuda()
        r_embed = torch.zeros(2 * len(r_dict), self.r_embed.shape[1]).cuda()
        for cur in e_dict:
            ent_embed[e_dict[cur]] = self.embed[cur].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]] = self.r_embed[cur + 1].cuda()
            r_embed[r_dict[cur] + len(r_dict)] = self.r_embed[cur + int(self.r_embed.shape[0] / 2) + 1].cuda()
        pre_sim = F.cosine_similarity(ent_embed[e_dict[e1]], ent_embed[e_dict[e2]], dim=0)
        y = torch.mm(ent_embed[kg1_index], ent_embed[kg2_index].t())
        ent_embed[e_dict[e1]] = torch.zeros(self.embed.shape[1]).cuda()
        ent_embed[e_dict[e2]] = torch.zeros(self.embed.shape[1]).cuda()
        # print(ent_embed)
        # print(r_embed)
        # print(ent_embed)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
        # print(explainer.triple_mask.requires_grad)
        model.train()
        # print(kg1_index, kg2_index)
        pre_sim = 0
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1000):
            optimizer.zero_grad()
            # print(ent_embed)
            h, r = model(ent_embed, r_embed, new_graph)
            # print(loss)
            h = h / (torch.linalg.norm(h, dim=-1, keepdim=True) + 1e-5)
            # x = torch.mm(h[kg1_index], h[kg2_index].t())
            sim1 = torch.mm(h[e_dict[e1]].unsqueeze(0), h[kg2_index].t())
            label1 = torch.Tensor([int(e_dict[e2])]).cuda()
            sim2 = torch.mm(h[e_dict[e2]].unsqueeze(0), h[kg1_index].t())
            label2 = torch.Tensor([int(e_dict[e1])]).cuda()
            loss = criterion(sim1, label1.long()) + criterion(sim2, label2.long())
            # kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
            # print(x.softmax(dim=-1).log(), y.softmax(dim=-1))
            # print(x, y, kl)
            # exit(0)
            # loss = F.pairwise_distance(h[e_dict[e1]], h[e_dict[e2]], p=2) 
            # loss = kl
            sim = F.cosine_similarity(h[e_dict[e1]], h[e_dict[e2]], dim=0)
            print('sim:',sim)
            if sim > pre_sim:
                pre_sim = sim
            else:
                break                
            print(loss)
            loss.backward()
            
            optimizer.step()
        return model, ent_embed, r_embed, e_dict, r_dict, new_graph

    def get_proxy_model_ori(self, e1, e2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.extract_subgraph(e1, e2, 1)
        
        e_embed = torch.zeros(len(e_dict), self.e_embed.shape[1]).cuda()
        r_embed = torch.zeros(2 * len(r_dict) + 2, self.r_embed.shape[1]).cuda()
        for cur in e_dict:
            e_embed[e_dict[cur]] = self.e_embed[cur].cuda()
        r_embed[0] = self.r_embed[0].cuda()
        r_embed[len(r_dict)] = self.r_embed[int(self.r_embed.shape[0] / 2)].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]+ 1] = self.r_embed[cur + 1].cuda()
            r_embed[r_dict[cur] + len(r_dict) + 1] = self.r_embed[cur + int(self.r_embed.shape[0] / 2) + 1].cuda()
        model = copy.deepcopy(self.base_model)
        model.ent_embedding = nn.Embedding(len(e_dict), e_embed.shape[1])
        model.ent_embedding.weight = torch.nn.Parameter(e_embed)
        model.rel_embedding = nn.Embedding(len(r_dict), r_embed.shape[1])
        model.rel_embedding.weight = torch.nn.Parameter(r_embed)
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph

    def get_proxy_model_aggre(self, e1, e2, p_embed1, p_embed2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.extract_subgraph(e1, e2, 1)
        model = Proxy(p_embed1, p_embed2)
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph


    def model_score(self, model, e1, e2, tri1, tri2):
        ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, tri1, tri2, True)
        me1, me2 = model.get_embeddings([e1], [e2], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)

        return me1, me2

    def change_pattern_id(self, p, e_dict, r_dict):
        new_p = []
        for cur in p:
            if len(cur) == 3:
                new_p.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
        return new_p

    def extract_feature(self, p_embed1, p_embed2, p1, p2):
        neigh_sim = torch.mm(p_embed1, p_embed2.t())
        _, index = (-neigh_sim).sort()
        select_index = index[:, : 3]
        p = []
        for i in range(len(p1)):
            for j in range(select_index[i].shape[0]):
                p.append([p1[i], p2[select_index[i][j]]])
        return p, torch.zeros(len(p))
    
    
    def std_filter(self, data, thred):
        # filtered_data = data[(data != np.max(data)) & (data != np.min(data))]
        mean = np.mean(data)  # 计算数据均值
        std = np.std(data)    # 计算数据标准差
        
        return mean, std


    def rule_sovler(self, e1, e2, p, p1, p2, e_dict_r, r_dict_r):
        solve_pair = []
        for i in range(len(p), 2):
            rel_score1 = self.G_dataset.rel_score[(r_dict_r[p1[p[i][0]][1]], r_dict_r[p2[p[i][0]][1]])]
            rel_score2 = self.G_dataset.rel_score[(r_dict_r[p1[p[i + 1][0]][1]], r_dict_r[p2[p[i + 1][0]][1]])]
            if e_dict_r[p1[p[i][0]][0]] != e1:
                g_neigh1 = e_dict_r[p1[p[i][0]][0]]
            else:
                g_neigh1 = e_dict_r[p1[p[i][0]][0]]
            if e_dict_r[p2[p[i][0]][0]] != e2:
                g_neigh2 = e_dict_r[p2[p[i][0]][0]]
            else:
                g_neigh2 = e_dict_r[p2[p[i][0]][0]]
            if e_dict_r[p1[p[i + 1][0]][0]] != e1:
                l_neigh1 = e_dict_r[p1[p[i + 1][0]][0]]
            else:
                l_neigh1 = e_dict_r[p1[p[i + 1][0]][0]]
            if e_dict_r[p2[p[i + 1][0]][0]] != e2:
                l_neigh2 = e_dict_r[p2[p[i + 1][0]][0]]
            else:
                l_neigh2 = e_dict_r[p2[p[i + 1][0]][0]]
            if self.G_dataset.test_res[g_neigh1] == g_neigh2 and rel_score1 > 0.6:
                solve_pair.append(p[i])
            elif self.G_dataset.test_res[l_neigh1] == l_neigh2 and rel_score2 > 0.6:
                solve_pair.append(p[i + 1])

    def judge(self, e1, e2, model, tri1, tri2, no_tri1, no_tri2, e_dict, r_dict):
        retain = self.split
        dec = 0
        try:
            if len(tri1) and len(tri2):
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, tri1, tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                _, suf_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                retain = torch.where(suf_rank[0] == e1)[0][0]
                print('suf rank:', retain)
            else:
                retain = self.split

            if len(no_tri1) and len(no_tri2):
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, no_tri1, no_tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                _, nec_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                # print(nec_rank.shape)
                dec = torch.where(nec_rank[0] == e1)[0][0]
                print('nec rank:', dec)
            else:
                dec = self.split
            
            if retain < 2 and dec > 0:
                return True
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
    def analyze_rule(self, rule):
        need = set()
        delete = set()
        for condition in rule:
            tmp = condition.split(' ')
            if tmp[1] == '>' and int(float(tmp[2])) == 0:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '<' and int(float(tmp[2])) == 1:
                delete.add(int(float(tmp[0])))
            elif tmp[1] == '>=' and int(float(tmp[2])) == 1:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '==' and int(float(tmp[2])) == 1:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '==' and int(float(tmp[2])) == 0:
                delete.add(int(float(tmp[0])))
            elif tmp[1] == '<=' and int(float(tmp[2])) == 0:
                delete.add(int(float(tmp[0])))
        return need, delete
    
    def explain_lore_two(self, e1, e2):
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        suff1 = set()
        suff2 = set()
        sample_num = 10
        for cur in neigh11:
            suff1 |= self.search_1_hop_tri(e1, cur)
        for cur in neigh12:
            two_hop = self.search_2_hop_tri1(e1, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff1 |= t1
                suff1 |= t2
        suff1  = set(random.sample(suff1, min(sample_num,len(suff1))))
        for cur in neigh21:
            suff2 |= self.search_1_hop_tri(e2, cur)
        for cur in neigh22:
            two_hop = self.search_2_hop_tri1(e2, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff2 |= t1
                suff2 |= t2
        suff2  = set(random.sample(suff2, min(sample_num,len(suff2))))
        # new_graph, e_dict, r_dict, kg1_index, kg2_index = self.change_map(suff1, suff2)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori_Trans_two(e1, e2, suff1, suff2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        p1 = list(suff1)
        p2 = list(suff2)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        if (len(p1) + len(p2)) < 100:
            explain = LORE(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed, self.lang)
            rule, class_name = explain.compute(30, 0.5)
        else:
            rule = None
        if rule == None:
            need_tri = []
            delete_tri = []
        else:
            need, delete = self.analyze_rule_lore(rule, class_name)
            need_tri = []
            delete_tri = []
            i = 0
            for cur in need:
                if cur < len(p1):
                    need_tri.append(p1[cur])  
                else:
                    need_tri.append(p2[cur - len(p1)])
            for cur in delete:
                if cur < len(p1):
                    delete_tri.append(p1[cur])  
                else:
                    delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule
        
    def explain_anchor_two(self, e1, e2):
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        suff1 = set()
        suff2 = set()
        sample_num = 10
        for cur in neigh11:
            suff1 |= self.search_1_hop_tri(e1, cur)
        for cur in neigh12:
            two_hop = self.search_2_hop_tri1(e1, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff1 |= t1
                suff1 |= t2
        suff1  = set(random.sample(suff1, min(sample_num,len(suff1))))
        for cur in neigh21:
            suff2 |= self.search_1_hop_tri(e2, cur)
        for cur in neigh22:
            two_hop = self.search_2_hop_tri1(e2, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff2 |= t1
                suff2 |= t2
        suff2  = set(random.sample(suff2, min(sample_num,len(suff2))))
        # new_graph, e_dict, r_dict, kg1_index, kg2_index = self.change_map(suff1, suff2)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori_Trans_two(e1, e2, suff1, suff2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        p1 = list(suff1)
        p2 = list(suff2)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        explain = Anchor_two(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed)
        rule = explain.compute(30, 0.6)
        need, delete = self.analyze_rule(rule)
        need_tri = []
        delete_tri = []
        i = 0
        for cur in need:
            if cur < len(p1):
                need_tri.append(p1[cur])  
            else:
                need_tri.append(p2[cur - len(p1)])
        for cur in delete:
            if cur < len(p1):
                delete_tri.append(p1[cur])  
            else:
                delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule

    def explain_anchor(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori_Trans(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        if (len(p1) + len(p2)) < 50:
            explain = Anchor(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed, self.lang)
            rule = explain.compute(30, 0.5)
        else:
            rule = None
        if rule == None:
            need_tri = []
            delete_tri = []
        else:
            need, delete = self.analyze_rule(rule)
            need_tri = []
            delete_tri = []
            i = 0
            for cur in need:
                if cur < len(p1):
                    need_tri.append(p1[cur])  
                else:
                    need_tri.append(p2[cur - len(p1)])
            for cur in delete:
                if cur < len(p1):
                    delete_tri.append(p1[cur])  
                else:
                    delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule

    def analyze_rule_lore(self, rule, class_name):
        need = set()
        delete = set()
        if class_name == 1:
            for condition in rule:
                tmp = rule[condition]
                if tmp[0] == '>' and tmp[1] == '-':
                    continue
                elif tmp[0] == '<' and tmp[1] == '-':
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and tmp[2] == '-':
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    continue
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    delete.add(int(float(condition)))
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and tmp[2] == '-':
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '>' and int(float(tmp[1])) == 0:
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and int(float(tmp[1])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '=' and int(float(tmp[2])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '==' and int(float(tmp[1])) == 0:
                    delete.add(int(float(condition)))
        else:
            for condition in rule:
                tmp = rule[condition]
                if tmp[0] == '>' and tmp[1] == '-':
                    continue
                elif tmp[0] == '<' and tmp[1] == '-':
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and tmp[2] == '-':
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    continue
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    need.add(int(float(condition)))
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and tmp[2] == '-':
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '>' and int(float(tmp[1])) == 0:
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and int(float(tmp[1])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '=' and int(float(tmp[2])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '==' and int(float(tmp[1])) == 0:
                    need.add(int(float(condition)))

        return need, delete

    def explain_lore(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori_Trans(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        if (len(p1) + len(p2)) < 50:
            explain = LORE(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed, self.lang)
            rule, class_name = explain.compute(30, 0.5)
        else:
            rule = None
        if rule == None:
            need_tri = []
            delete_tri = []
        else:
            need, delete = self.analyze_rule_lore(rule, class_name)
            need_tri = []
            delete_tri = []
            i = 0
            for cur in need:
                if cur < len(p1):
                    need_tri.append(p1[cur])  
                else:
                    need_tri.append(p2[cur - len(p1)])
            for cur in delete:
                if cur < len(p1):
                    delete_tri.append(p1[cur])  
                else:
                    delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule
    
    
    def Trans_Process_two(self, e, tri):
        i = 0
        p_embed = torch.zeros(len(tri), self.embed.shape[1] + self.r_embed.shape[1]) 
        for cur in tri:
            '''
            if cur[0] == e:
                p_embed[i] = self.embed[cur[2]] +  self.r_embed[cur[1]]
            else:
                p_embed[i] = self.embed[cur[0]] -  self.r_embed[cur[1]]
            '''
            # print(self.embed[cur[2]].shape, self.r_embed[cur[1]].shape)
            if cur[0] == e:
                p_embed[i] = torch.cat((self.embed[cur[2]] ,  self.r_embed[cur[1] + 1]), dim=0)
            else:
                p_embed[i] = torch.cat((self.embed[cur[0]] ,  self.r_embed[int(self.r_embed.shape[0] / 2) + cur[1] + 1]), dim=0)
            i += 1
        return p_embed

    def get_proxy_model_ori_Trans_two(self, e1, e2, suff1, suff2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.change_map(suff1, suff2)
        
        p_embed1 = self.Trans_Process_two(e1, suff1)
        p_embed2 = self.Trans_Process_two(e2, suff2)
        model = Proxy(p_embed1, p_embed2)
        
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph

    def explain_lime(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        lime = LIME(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset, self.embed)
        res = lime.compute(100)
        res = torch.Tensor(res)
        # res = res.squeeze(1)
        score, indices = res.sort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        for cur in indices:
            if cur < len(p1):
                tri.append(p1[cur])
            else:
                tri.append(p2[cur - len(p1)])
        tri.append((0,0,0))
        return tri

    def explain_llm(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        llm = LLM(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset, self.embed)
        sim_num = int(600 / (len(p1) + len(p2)))
        Z, Y = llm.compute(sim_num)
        return Z, Y, p1, p2

    def explain_shapely(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        if len(p1) > 50 or len(p2) > 50:
            new_p = new_p1 + new_p2
            Shapley = KernelSHAP(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset, self.embed)
            res = Shapley.compute(100)
            res = torch.Tensor(res)
            # res = res.squeeze(1)
            score, indices = res.sort(descending=True)
            tri1 = []
            tri2 = []
            tri = []
            for cur in indices:
                if cur < len(p1):
                    tri.append(p1[cur])
                else:
                    tri.append(p2[cur - len(p1)])
            tri.append((0,0,0))
        else:
            Shapley = Shapley_Value(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset)
            shapley_value = Shapley.MTC(30)
            res = torch.Tensor(shapley_value).argsort(descending=True)
            tri1 = []
            tri2 = []
            tri = []
            for cur in res:
                if cur < len(p1):
                    tri.append(p1[cur])
                else:
                    tri.append(p2[cur - len(p1)])
            # return tri1, tri2
            tri.append((0,0,0))
        return tri
    
    def get_proxy_two(self, tri1, tri2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.change_map(tri1, tri2)
        
        e_embed = torch.zeros(len(e_dict), self.e_embed.shape[1]).cuda()
        r_embed = torch.zeros(2 * len(r_dict) + 2, self.r_embed.shape[1]).cuda()
        for cur in e_dict:
            e_embed[e_dict[cur]] = self.e_embed[cur].cuda()
        r_embed[0] = self.r_embed[0].cuda()
        r_embed[len(r_dict)] = self.r_embed[int(self.r_embed.shape[0] / 2)].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]+ 1] = self.r_embed[cur + 1].cuda()
            r_embed[r_dict[cur] + len(r_dict) + 1] = self.r_embed[cur + int(self.r_embed.shape[0] / 2) + 1].cuda()
        model = copy.deepcopy(self.base_model)
        model.ent_embedding = nn.Embedding(len(e_dict), e_embed.shape[1])
        model.ent_embedding.weight = torch.nn.Parameter(e_embed)
        model.rel_embedding = nn.Embedding(len(r_dict), r_embed.shape[1])
        model.rel_embedding.weight = torch.nn.Parameter(r_embed)
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph

    def explain_lime_two(self, e1, e2):
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        suff1 = set()
        suff2 = set()
        for cur in neigh11:
            suff1 |= self.search_1_hop_tri(e1, cur)
        for cur in neigh12:
            two_hop = self.search_2_hop_tri1(e1, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff1 |= t1
                suff1 |= t2
        for cur in neigh21:
            suff2 |= self.search_1_hop_tri(e2, cur)
        for cur in neigh22:
            two_hop = self.search_2_hop_tri1(e2, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff2 |= t1
                suff2 |= t2
        # new_graph, e_dict, r_dict, kg1_index, kg2_index = self.change_map(suff1, suff2)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_two(suff1, suff2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        p1 = list(suff1)
        p2 = list(suff2)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        lime = LIME(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset, self.embed)
        res = lime.compute(100)
        # res = res.squeeze(1)
        res = torch.Tensor(res)
        score, indices = res.sort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        for cur in indices:
            if cur < len(p1):
                tri.append(p1[cur])
            else:
                tri.append(p2[cur - len(p1)])
        tri.append((0,0,0))
        return tri

    def explain_shapely_two(self, e1, e2):
        # p1, p_embed1 = self.pattern_process(e1, 1)
        # p2, p_embed2 = self.pattern_process(e2, 1)
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        suff1 = set()
        suff2 = set()
        for cur in neigh11:
            suff1 |= self.search_1_hop_tri(e1, cur)
        for cur in neigh12:
            two_hop = self.search_2_hop_tri1(e1, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff1 |= t1
                suff1 |= t2
        for cur in neigh21:
            suff2 |= self.search_1_hop_tri(e2, cur)
        for cur in neigh22:
            two_hop = self.search_2_hop_tri1(e2, cur)
            for cur1 in two_hop:
                t1 = cur1[0]
                t2 = cur1[1]
                suff2 |= t1
                suff2 |= t2
        # new_graph, e_dict, r_dict, kg1_index, kg2_index = self.change_map(suff1, suff2)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_two(suff1, suff2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        p1 = list(suff1)
        p2 = list(suff2)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        # p1 = list(suff1)
        # p2 = list(suff2)
        # new_p = p1 + p2
        Kernelshap = KernelSHAP(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, e_dict, r_dict, self.G_dataset, self.embed)
        # Shapley = Shapley_Value_two(model, e1, e2, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), new_p, self.G_dataset)
        # Shapley = Shapley_Value(model, e1, e2, len(new_p1) + len(new_p2), list(range(len(new_p1) + len(new_p2))), len(new_p1), new_p, e_dict, r_dict, self.G_dataset)
        # shapley_valu e = Shapley.MTC(30)
        res = Kernelshap.compute(100)
        # res = res.squeeze(1)
        res = torch.Tensor(res)
        score, indices = res.sort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        for cur in indices:
            if cur < len(p1):
                tri.append(p1[cur])
            else:
                tri.append(p2[cur - len(p1)])
        tri.append((0,0,0))
        return tri

    def explain_aggre(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_aggre(e1, e2, p_embed1, p_embed2)
        # print(model.ent_embedding.weight.shape)
        # exit(0)
        neigh_sim1 = torch.mm(p_embed1, p_embed2.t())
        neigh_sim2 = torch.mm(p_embed2, p_embed1.t())
       
        neigh_pre1 = (-neigh_sim1).argsort()
        neigh_pre2 = (-neigh_sim2).argsort()
        # res = self.max_weight_match(neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, neigh_sim1, 0.75)
        pair = []
        for i in range(neigh_pre1.shape[0]):
            pair.append((i,int(neigh_pre1[i][0])))
    
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        new_p = new_p1 + new_p2
        Shapley = Shapley_Value(model, e1, e2, len(pair), list(range(len(pair))), len(p1), new_p, e_dict, r_dict, self.G_dataset)
        res = Shapley.compute_shapley_value(100, 'aggre', pair)
    
        tri1 = []
        tri2 = []
        for cur in res[:5]:
            tri1.append(p1[cur[0]])
            tri2.append(p2[cur[1]])
        return tri1, tri2

    def explain_ours1(self, e1, e2):
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        mask = []
        Y = []
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
    
        for cur in p1:
            self.read_triple_name(cur)
        for cur in p2:
            self.read_triple_name(cur)
        
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        match_tri1, match_tri2, match = self.match(p_embed1, p_embed2, p1, p2)
        if len(match) == 1:
            res_tri1 = []
            res_tri2 = []
            for i in range(len(match)):
                if i >= 5:
                    break
                tri1 = match_tri1[i]
                tri2 = match_tri2[i]
                for cur in tri1:
                    res_tri1.append(p1[cur])
                for cur in tri2:
                    res_tri2.append(p2[cur])
            return res_tri1, res_tri2
        
        res_tri1 = []
        res_tri2 = []
        for i in range(len(match)):
            if i >= 5:
                break
            tri1 = match_tri1[i]
            tri2 = match_tri2[i]
            for cur in tri1:
                res_tri1.append(p1[cur])
            for cur in tri2:
                res_tri2.append(p2[cur])
        return res_tri1, res_tri2
        
        Lasso = linear_model.LassoCV(cv=len(match))
        i = 0
        k = 1
        mask = []
        Y = []
        comb_num = 1
        while i < 1000 and k <= min(comb_num, int(len(match) / 2) +1):
            comb = [list(c)  for c in combinations(list(range(len(match))), k)]
            for coal in comb:
                time1 = time.time()
                tri1 = []
                tri2 = []
                no_tri1 = []
                no_tri2 = []
                cur_mask1 = [0] * (len(match))
                cur_mask2 = [1] * (len(match))
                no_tri1 = new_p1.copy()
                no_tri2 = new_p2.copy()
                for cur in coal:
                    for t in match_tri1[cur]:  
                        tri1.append(new_p1[t])
                        no_tri1.remove(new_p1[t])
                    for t in match_tri2[cur]:  
                        tri2.append(new_p2[t])
                        no_tri2.remove(new_p2[t])
                    cur_mask1[cur] = 1
                    cur_mask2[cur] = 0
                # print(tri1, tri2)
                # print(no_tri1, no_tri2)
                try:
                    ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, tri1, tri2, True, len(e_dict), len(r_dict))
                    proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                    v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
                    mask.append(cur_mask1)
                    Y.append(v_suf)
                    if len(no_tri1) and len(no_tri2):
                        ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, no_tri1, no_tri2, True, len(e_dict), len(r_dict))
                        proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                        v_nec = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0) 
                        mask.append(cur_mask2)
                        Y.append(v_nec)
                    i += 1
                        # time2 = time.time()
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print('WARNING: out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        else:
                            raise exception
                    # print(time2 - time1, v_suf, v_nec)
            k += 1
        
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        # print(Z, Y)
        # Lasso.fit(Z, Y)
        # print(Lasso.alpha_)
        # print(Lasso.coef_)
        # exit(0)
        I = torch.eye(Z.shape[1])
        res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        # res = torch.Tensor(Lasso.coef_)
        # print(Z)
        res = res.squeeze(1)
        score, indices = res.sort(descending=True)
        res_tri1 = []
        res_tri2 = []
        for i in range(len(match)):
            if i >= 5 or score[i] == 0:
                break
            tri1 = match_tri1[indices[i]]
            tri2 = match_tri2[indices[i]]
            print(score[i], tri1, tri2)
            for cur in tri1:
                res_tri1.append(p1[cur])
            for cur in tri2:
                res_tri2.append(p2[cur])
        return res_tri1, res_tri2

    def explain_ours2(self, e1, e2):
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        mask = []
        Y = []
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        for cur in p1:
            self.read_triple_name(cur)
        for cur in p2:
            self.read_triple_name(cur)
        
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        global_sim = torch.mm(p_embed1, p_embed2.t())
        neigh_pre1 = (-global_sim).argsort()
        suf_list = []
        tri1 = []
        tri2 = []
        ori_e2 = self.embed[e2]
        for i in range(neigh_pre1.shape[0]):
            tri1 = [new_p1[i]]
            tri2 = [new_p2[neigh_pre1[i][0]]]
            # self.read_triple_name(p1[i])
            # self.read_triple_name(p2[neigh_pre1[i][0]])
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], tri1, tri2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            # v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
            # print(v_suf)
            self.embed[e2] = proxy_e2[0]
            _, suf_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
            retain = torch.where(suf_rank[0] == e1)[0][0]
            suf_list.append(retain)
        nec_list = []
        for i in range(neigh_pre1.shape[0]):
            tri1 = new_p1.copy()
            tri1.remove(new_p1[i])
            tri2 = new_p2.copy()
            tri2.remove(new_p2[neigh_pre1[i][0]])
            self.read_triple_name(p1[i])
            self.read_triple_name(p2[neigh_pre1[i][0]])
            if len(tri1) == 0 or len(tri2) == 0:
                retain = self.split
            else:
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], tri1, tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                # v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
                # print(v_suf)
                self.embed[e2] = proxy_e2[0]
                _, nec_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                retain = torch.where(nec_rank[0] == e1)[0][0]
            nec_list.append(retain)
        score = torch.Tensor(nec_list) - torch.Tensor(suf_list)
        score, indices = score.sort(descending=True)
        no_tri1 = new_p1.copy()
        no_tri2 = new_p2.copy()
        tri1 = []
        tri2 = []
        res_tri1 = []
        res_tri2 = []
        best = -1e10
        for i in range(len(suf_list)):
            tri1.append(new_p1[indices[i]])
            tri2.append(new_p2[neigh_pre1[indices[i]][0]])
            res_tri1.append(p1[indices[i]])
            res_tri2.append(p2[neigh_pre1[indices[i]][0]])
            no_tri1.remove(new_p1[indices[i]])
            if new_p2[neigh_pre1[indices[i]][0]] in no_tri2:
                no_tri2.remove(new_p2[neigh_pre1[indices[i]][0]])
            # self.read_triple_name(p1[indices[i]])
            # self.read_triple_name(p2[neigh_pre1[indices[i]][0]])
            if len(no_tri1) == 0 or len(no_tri2) == 0:
                dec = self.split
                
            else:
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], no_tri1, no_tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                self.embed[e2] = proxy_e2[0]
                _, nec_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                dec = torch.where(nec_rank[0] == e1)[0][0]
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], tri1, tri2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            self.embed[e2] = proxy_e2[0]
            _, suf_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
            retain = torch.where(suf_rank[0] == e1)[0][0]
            if retain <= 0 and dec > 0:
                # print('true res:')
                # print('-------------------------------')
                # for cur in res_tri1:
                    # self.read_triple_name(cur)
                # for cur in res_tri2:
                    # self.read_triple_name(cur)
                # rint('-------------------------------')
                self.embed[e2] = ori_e2
                return res_tri1, res_tri2 

            if dec - retain > best:
                best = dec - retain
            else:
                if random.random() > abs(dec - retain) / abs(best):
                    # print('appro res:')
                    # print('-------------------------------')
                    # for cur in res_tri1:
                        # self.read_triple_name(cur)
                    # for cur in res_tri2:
                        # self.read_triple_name(cur)
                    # print('-------------------------------')
                    self.embed[e2] = ori_e2
                    return res_tri1, res_tri2 
        self.embed[e2] = ori_e2
        return res_tri1, res_tri2   

        '''
        sim_list = []
        for cur in new_p1:
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], [cur], new_p2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
            sim_list.append(v_suf)
        print(sim_list)
        score, indices = torch.Tensor(sim_list).sort(descending=True)
        
        tri1 = []
        tri2 = []
        for i in range(len(sim_list)):
            # tri1.append(new_p1[indices[i]])
            # tri2.append(new_p2[neigh_pre1[indices[i]][0]])
            tri1 = [new_p1[indices[i]]]
            tri2 = [new_p2[neigh_pre1[indices[i]][0]]]
            self.read_triple_name(p1[indices[i]])
            self.read_triple_name(p2[neigh_pre1[indices[i]][0]])
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], tri1, tri2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            # v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
            # print(v_suf)
            _, suf_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
            retain = torch.where(suf_rank[0] == e1)[0][0]
            print(retain)
        '''
        # exit(0)
        
        # print(Z, Y)
        # Lasso.fit(Z, Y)
        # print(Lasso.alpha_)
        # print(Lasso.coef_)
        # exit(0)
        # I = torch.eye(Z.shape[1])
        # res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        # res = torch.Tensor(Lasso.coef_)
        # print(Z)
        # res = res.squeeze(1)
        # score, indices = res.sort(descending=True)
        # res_tri1 = []
        # res_tri2 = []
        

    def solve_weights(self, vectors, target_vector):
        A = np.vstack(vectors).T  # 构建系数矩阵A
        b = target_vector  # 构建目标向量b
        weights, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return weights


    def solve_pair_weights(self, A, B):

        A1 = np.concatenate((A, -B), axis=0)
        print(A1.shape)
        b = np.ones(len(A[0]))
        b /= 1e3
        weights, residuals, _, _ = np.linalg.lstsq(A1.T, b, rcond=None)

        return weights.flatten()

    def explain_ours3(self, e1, e2):
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        mask = []
        Y = []
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        
        for cur in p1:
            self.read_triple_name(cur)
        for cur in p2:
            self.read_triple_name(cur)
        
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        p_embed1 = (p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)).cuda()
        p_embed2 = (p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)).cuda()


        '''
        p_embed1 = []
        p_embed2 = []
        for i in range(len(p1)):
            tri1 = [new_p1[i]]
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], tri1, new_p2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            # print(proxy_e1[0])
            p_embed1.append(proxy_e1[0].detach())
        for i in range(len(p2)):
            tri2 = [new_p2[i]]
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e_dict[e1], e_dict[e2], new_p1, tri2, True, len(e_dict), len(r_dict))
            proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)

            p_embed2.append(proxy_e2[0].detach())
        
        p_embed1 = torch.stack(p_embed1)
        p_embed2 = torch.stack(p_embed2)
        p_embed1 = (p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)).cuda()
        p_embed2 = (p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)).cuda()
        
        # w = self.solve_weights(p_embed1, p_embed2)
        w1 = self.solve_weights(p_embed1.cpu().detach().numpy(), self.embed[e1].cpu().detach().numpy())
        w2 = self.solve_weights(p_embed2.cpu().detach().numpy(), self.embed[e2].cpu().detach().numpy())
        p_embed1 *= torch.Tensor(w1).unsqueeze(1).cuda()
        p_embed2 *= torch.Tensor(w2).unsqueeze(1).cuda()
        # print(p_embed1)
        # print(p_embed2)
        # w = self.solve_pair_weights(p_embed1.cpu().detach().numpy(),p_embed2.cpu().detach().numpy())
        # print(w)
        # print(p_embed1,p_embed2)
        '''
        global_sim = torch.mm(p_embed1, p_embed2.t())
        neigh_pre1 = (-global_sim).argsort()
        for i in range(neigh_pre1.shape[0]):
            print(i, neigh_pre1[i][0])
    def get_1_hop(self, e):
        neigh1 = set()
        for cur in self.G_dataset.gid[e]:
            if cur[0] != int(e):
                neigh1.add(int(cur[0]))
            else:
                neigh1.add(int(cur[2]))
        return neigh1

    def init_2_hop(self, e1):
        neigh2 = set()
        neigh1 = self.get_1_hop(e1)
        for ent in neigh1:
            neigh2 |= self.get_1_hop(ent)
        neigh2.remove(e1)

        return neigh2 - neigh1 , neigh1
    def explain_ours_two(self, e1, e2):
        exp_tri1 = []
        exp_tri2 = []
        cur_link = self.model_link
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        score = 0
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh21:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh21:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            # print(self.search_2_hop_tri(e1, p[0]))
            two_hop_list = self.search_2_hop_tri1(e1, p[0])
            
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            index = 0
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)

            for i in range(len(pair_r)):
                pr = pair_r[i]
                exp_tri1 += [two_hop[pr[0]][0]]
                exp_tri1 += [two_hop[pr[0]][1]]
                exp_tri2 += [tri2[pr[1]]]
                
        pair = set()
        for cur in neigh11:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh11:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list = self.search_2_hop_tri1(e2, p[1])
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]), dim=0))
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                exp_tri1 += [tri1[pr[0]]]
                exp_tri2 += [two_hop[pr[1]][0]]
                exp_tri2 += [two_hop[pr[1]][1]]
                
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list1 = self.search_2_hop_tri1(e1, p[0])

            two_hop_list2 = self.search_2_hop_tri1(e2, p[1])

            r1 = []
            r2 = []
            two_hop1 = []
            for cur in two_hop_list1:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop1 += [(cur1, cur2)]
            
            two_hop2 = []
            for cur in two_hop_list2:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop2 += [(cur1, cur2)]
            # print(two_hop2)
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)

            for i in range(len(pair_r)):
                pr = pair_r[i]
                # print(two_hop1[pr[0]][0])
                exp_tri1 += [two_hop1[pr[0]][0]]
                exp_tri1 += [two_hop1[pr[0]][1]]
                exp_tri2 += [two_hop2[pr[1]][0]]
                exp_tri2 += [two_hop2[pr[1]][1]]
        # print(exp_tri1, exp_tri2)
        return exp_tri1, exp_tri2
    
    def search_2_hop_tri1(self, e, tar):
        neigh1 = self.get_1_hop(e)
        tri2 = []
        cur1 = set()
        cur2 = set()
        for ent in neigh1:
            neigh2 = self.get_1_hop(ent)
            if tar in neigh2:
                t1 = self.search_1_hop_tri(e, ent)
                t2 = self.search_1_hop_tri(ent, tar)
                tri2.append((t1, t2, ent))
                cur1 |= t1
                cur2 |= t2
        # print(cur1, cur2)
        return tri2
    
    def explain_ours4(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        cur_link = self.model_link
        pair = set()
        for cur in neigh1:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh2:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))


        # pair = self.explain_bidrect(e1, e2)[:5]
        tri1_list = []
        tri2_list = []
        # print(pair)
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1] + 1])
            for cur in tri2:
                r2.append(self.r_embed[cur[1] + 1])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                # print(len(pair_r))
            new_tri1 = []
            new_tri2 = []
            for pr in pair_r:
                # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                    # print(self.G_dataset.r_dict[tri2[pr[1]][1]], self.G_dataset.r_dict[tri1[pr[0]][1]])
                new_tri1.append(tri1[pr[0]])
                new_tri2.append(tri2[pr[1]])
            # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                # exit(0)
            tri1_list += new_tri1
            tri2_list += new_tri2
        return tri1_list, tri2_list, pair
    
    def explain_ours_shapley(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        cur_link = self.model_link
        pair = set()
        for cur in neigh1:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh2:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))


        # pair = self.explain_bidrect(e1, e2)[:5]
        tri1_list = []
        tri2_list = []
        # print(pair)
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1] + 1])
            for cur in tri2:
                r2.append(self.r_embed[cur[1] + 1])
            

            '''
            t1, t2 = self.cluster_match(torch.stack(r1), torch.stack(r2))
            new_tri1 = []
            new_tri2 = []
            for cur in t1:
                cur_t = []
                for r in cur:
                    cur_t.append(tri1[r])
                    # self.read_triple_name(tri1[r])
                # print('---------------')
                new_tri1.append(cur_t)
            for cur in t2:
                cur_t = []
                for r in cur:
                    cur_t.append(tri2[r])
                    # self.read_triple_name(tri2[r])
                # print('---------------')
                new_tri2.append(cur_t)
            
            '''
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                # print(len(pair_r))
            new_tri1 = []
            new_tri2 = []
            for pr in pair_r:
                # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                    # print(self.G_dataset.r_dict[tri2[pr[1]][1]], self.G_dataset.r_dict[tri1[pr[0]][1]])
                new_tri1.append(tri1[pr[0]])
                new_tri2.append(tri2[pr[1]])
            # if self.G_dataset.ent_dict[e1] == 'http://zh.dbpedia.org/resource/德文·韦德':
                # exit(0)
            tri1_list += new_tri1
            tri2_list += new_tri2
        return tri1_list, tri2_list, pair

    def explain_ours5(self, e1, e2):
        pair = self.explain_bidrect(e1, e2)
        pair_set = set()
        for p in pair:
            pair_set.add((self.G_dataset.ent_dict[p[0]], self.G_dataset.ent_dict[p[1]]))
        return pair_set

    def explain_ours6(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        pair = set()
        for cur in neigh1:
            if str(cur) in self.model_link:
                if int(self.model_link[str(cur)]) in neigh2:
                    
                    pair.add((cur, int(self.model_link[str(cur)])))
        if len(pair) == 0:
            return []
        dependence = []
        print(pair)
        for p in pair:
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = 0
            r2 = 0
            r3 = 0
            r4 = 0
            for cur in tri1:
                if cur[0] == e1:
                    r1 = max(r1, r1_func[str(cur[1])])
                    r3 = max(r3, r1_func_r[str(cur[1])])
                else:
                    r1 = max(r1, r1_func_r[str(cur[1])])
                    r3 = max(r3, r1_func[str(cur[1])])

            for cur in tri2:
                if cur[0] == e2:
                    r2 = max(r2, r2_func[str(cur[1])])
                    r4 = max(r4, r2_func_r[str(cur[1])])
                else:
                    r2 = max(r2, r2_func_r[str(cur[1])])
                    r4 = max(r4, r2_func[str(cur[1])])
            print(r1, r2, r3, r4)

            # if min(r1, r2) < min(r3, r4):
            dependence.append((p[0], p[1], max(r3, r4)))
            
        return dependence

    def explain_ours7(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if str(cur) in self.model_link:
                if int(self.model_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.model_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1] + 1])
            for cur in tri2:
                r2.append(self.r_embed[cur[1] + 1])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            
            for pr in pair_r:
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                r_score = max(r_score, cur_score)
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            
            
        return pair_node, score

    def explain_ours8(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        pair = set()
        for cur in neigh1:
            if str(cur) in self.model_link:
                if int(self.model_link[str(cur)]) in neigh2:
                    
                    pair.add((cur, int(self.model_link[str(cur)])))
        if len(pair) == 0:
            return []
        dependence = []
        for p in pair:
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1] + 1])
            for cur in tri2:
                r2.append(self.r_embed[cur[1] + 1])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            
            for pr in pair_r:
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                r_score = max(r_score, cur_score)

            # if min(r1, r2) < min(r3, r4):
            dependence.append((p[0], p[1],r_score))
        return dependence
    
    def init_1_hop(self, gid1, gid2):
        neigh1 = set()
        neigh2 = set()
        for cur in self.G_dataset.gid[gid1]:
            if cur[0] != int(gid1):
                neigh1.add(int(cur[0]))
            else:
                neigh1.add(int(cur[2]))
        for cur in self.G_dataset.gid[gid2]:
            if cur[0] != int(gid2):
                neigh2.add(int(cur[0]))
            else:
                neigh2.add(int(cur[2]))
        return neigh1, neigh2


    def search_2_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if t[0] == target or t[2] == target:
                tri.add((t[0], t[1], t[2]))
                continue
            if t[0] != source:
                for t2 in self.G_dataset.gid[t[0]]:
                    # print(t[0],target, t2)
                    if t2[0] == target or t2[2] == target:
                        tri.add(((t[0], t[1], t[2]),(t2[0], t2[1], t2[2])))
            else:
                for t2 in self.G_dataset.gid[t[2]]:
                    if t2[0] == target or t2[2] == target:
                        tri.add(((t[0], t[1], t[2]),(t2[0], t2[1], t2[2])))
        return tri
    
    def search_1_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if ((t[0] == target and t[2] == source ) or (t[2] == target and t[0] == source)):
                tri.add((t[0], t[1], t[2]))
                continue

        return tri
    
    def give_explain_set(self, gid, neigh):
        tri = set()
        for cur in neigh:
            tri |= self.search_2_hop_tri(gid, cur)
        return tri

    def max_weight_match(self, neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, sim, thred):
        G = nx.Graph()
        edges = []
        # print(sim.shape)
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                # print(sim[i][neigh_pre1[i][j]])
                if sim[i][neigh_pre1[i][j]] >= thred:
                    edges.append(( neigh_list2[int(neigh_pre1[i][j])],neigh_list1[i], sim[i][neigh_pre1[i][j]]))
                
                    # edges.append((int(neigh_pre1[i][j]), i, sim[i][neigh_pre1[i][j]]))
                else:
                    break
        G.add_weighted_edges_from(edges)
        return sorted(nx.max_weight_matching(G))


    def bidirect_match(self, neigh_pre1, neigh_pre2, neigh_list1=None, neigh_list2=None, sim=None):
        res = []
        for i in range(neigh_pre1.shape[0]):
            select = neigh_pre1[i][0]
            if i == neigh_pre2[select][0]:
                # res.append([[i, select], sim[i][select]])
                res.append((i, select))
        # res.sort(key=lambda x:x[1], reverse=True)
        # match = []
        # for cur in res:
            # match.append(cur[0])
        return res

    def save_pair(self, pair, file):
        with open(file, 'w') as f:
            for p in pair:
                f.write(p[0] + '\t' + p[1] + '\n')

    def change_to_list(self, exp):
        exp_list = []
        for cur in exp:
            exp_list.append(list(cur))
        return exp_list


    def compute_coal_value(self, gid1, gid2, c):
        c = torch.Tensor(c)
        neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        neigh1 = F.normalize(neigh1, dim = 0)
        neigh2 = self.embed[c.t()[1].long()].mean(dim = 0)
        neigh2 = F.normalize(neigh2, dim = 0)
        return F.cosine_similarity(neigh1, neigh2, dim=0)


    def Shapely_value_debug(self, exp, gid1, gid2):
        exp = self.change_to_list(exp)
        num_exp = len(exp)

        all_coal = [list(coal) for i in range(num_exp) for coal in combinations(exp, i + 1)]
        shapely_value = []
        for e in exp:
            e_coal = []
            no_e_coal = []
            for c in copy.deepcopy(all_coal):
                if e in c:
                    value = self.compute_coal_value(gid1, gid2, c)
                    e_coal.append((copy.deepcopy(c), value))
                    c.remove(e)
                    if len(c) == 0:
                        no_e_coal.append((c, 0))
                    else:
                        value = self.compute_coal_value(gid1, gid2, c)
                        no_e_coal.append((c, value))
            shapelyvalue = 0
            for i in range(len(e_coal)):
                s = len(e_coal[i][0])
                e_payoff = e_coal[i][1] - no_e_coal[i][1]
                e_weight = math.factorial(s-1)*math.factorial(num_exp-s)/math.factorial(num_exp)
                shapelyvalue += e_payoff * e_weight
            shapely_value.append((e,shapelyvalue))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        for cur in shapely_value:
            print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.ent_dict[cur[0][1]])


    def Shapely_value(self, exp, gid1, gid2, suf=True):
        exp = self.change_to_list(exp)
        num_exp = len(exp)

        if num_exp > 10:
            exp = exp[:10]
            num_exp = 10
        all_coal = [list(coal) for i in range(num_exp) for coal in combinations(exp, i + 1)]
        # print(num_exp)
        shapely_value = []
        for e in exp:
            # print(e)
            e_coal = []
            no_e_coal = []
            
            for c in copy.deepcopy(all_coal):
                if e in c:
                    if suf:
                        value = self.compute_coal_value(gid1, gid2, c)
                    else:
                        tmp_exp = copy.deepcopy(exp)
                        for cur in c:
                            tmp_exp.remove(cur)
                        if len(tmp_exp) == 0:
                            # no_e_coal.append((0, 0))
                            value = 0
                        else:
                            value = -self.compute_coal_value(gid1, gid2, tmp_exp)
                    l = len(c)
                    e_coal.append((l, value))
                    c.remove(e)
                    if len(c) == 0:
                        no_e_coal.append((0, 0))
                    else:
                        if suf:
                            value = self.compute_coal_value(gid1, gid2, c)
                        else:
                            tmp_exp = copy.deepcopy(exp)
                            for cur in c:
                                tmp_exp.remove(cur)
                            if len(tmp_exp) == 0:
                                # no_e_coal.append((0, 0))
                                value = 0
                            else:
                                value = -self.compute_coal_value(gid1, gid2, tmp_exp)
                        no_e_coal.append((l - 1, value))
            shapelyvalue = 0
            for i in range(len(e_coal)):
                s = e_coal[i][0]
                e_payoff = e_coal[i][1] - no_e_coal[i][1]
                e_weight = math.factorial(s-1)*math.factorial(num_exp-s)/math.factorial(num_exp)
                shapelyvalue += e_payoff * e_weight
            shapely_value.append((e,shapelyvalue))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        new_exp = []
        for cur in shapely_value:
            new_exp.append((cur[0][0], cur[0][1]))
            # if len(new_exp) > 4:
                # break
            print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.ent_dict[cur[0][1]])
        return new_exp        

    def cosine_matrix(self, A, B):
        A_sim = torch.mm(A, B.t())
        a = torch.norm(A, p=2, dim=-1)
        b = torch.norm(B, p=2, dim=-1)
        cos_sim = A_sim / a.unsqueeze(-1)
        cos_sim /= b.unsqueeze(-2)
        return cos_sim

    def read_triple_name(self, triple):
        print(self.G_dataset.ent_dict[triple[0]], self.G_dataset.r_dict[triple[1]],self.G_dataset.ent_dict[triple[2]])
    
    def get_r_func(self):
        tri1 = read_tri('../datasets/' + self.lang + '-en_f/base/triples_1')
        tri2 = read_tri('../datasets/' + self.lang + '-en_f/base/triples_2')
        r, _ = read_link('../datasets/' + self.lang + '-en_f/base/rel_dict')
        r1 = defaultdict(set)
        r2 = defaultdict(set)
        r1_func = defaultdict(int)
        r2_func = defaultdict(int)
        r1_func_r = defaultdict(int)
        r2_func_r = defaultdict(int)
        for cur in tri1:
            r1[cur[1]].add((cur[0], cur[2]))
        for cur in tri2:
            r2[cur[1]].add((cur[0], cur[2]))
        
        for cur in r1:
            x = defaultdict(int)
            for t in r1[cur]:
                x[t[0]] = 1
            r1_func[cur] = len(x) / len(r1[cur])
            x_r = defaultdict(int)
            for t in r1[cur]:
                x_r[t[1]] = 1
            r1_func_r[cur] = len(x_r) / len(r1[cur])
        
        for cur in r2:
            x = defaultdict(int)
            for t in r2[cur]:
                x[t[0]] = 1
            r2_func[cur] = len(x) / len(r2[cur])
            x_r = defaultdict(int)
            for t in r2[cur]:
                x_r[t[1]] = 1
            r2_func_r[cur] = len(x_r) / len(r2[cur])
        
        return r1_func, r1_func_r, r2_func, r2_func_r

    def get_test_file(self, file):
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        with open('../datasets/zh-en_f/base/test_triples_1_nec', 'w') as f1, open('../datasets/zh-en_f/base/test_triples_2_nec', 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
        with open('../datasets/zh-en_f/base/test_triples_1_suf', 'w') as f1, open('../datasets/zh-en_f/base/test_triples_2_suf', 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

    def save_two_hop_tri(self, file):
        suff = set()
        for gid1, gid2 in self.test_indices:
            e1 = int(gid1)
            e2 = int(gid2)
            neigh12, neigh11 = self.init_2_hop(e1)
            neigh22, neigh21 = self.init_2_hop(e2)
            for cur in neigh11:
                suff |= self.search_1_hop_tri(e1, cur)
            for cur in neigh12:
                two_hop = self.search_2_hop_tri1(e1, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
            for cur in neigh21:
                suff |= self.search_1_hop_tri(e2, cur)
            for cur in neigh22:
                two_hop = self.search_2_hop_tri1(e2, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
        with open(file + self.lang, 'w') as f:
            for cur in suff:
                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

    def get_test_file_mask_two(self, file, thred, method=''):
        if 'lime' in file or 'shapley' in file:
            print('Save ranked triples in {}'.format(file))
            return
        elif 'lore' in file or 'anchor' in file:
            print('Save rules in {}'.format(file))
            return
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                # print(cur)
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        suff = set()
        for gid1, gid2 in self.test_indices:
            e1 = int(gid1)
            e2 = int(gid2)
            neigh12, neigh11 = self.init_2_hop(e1)
            neigh22, neigh21 = self.init_2_hop(e2)
            for cur in neigh11:
                suff |= self.search_1_hop_tri(e1, cur)
            for cur in neigh12:
                two_hop = self.search_2_hop_tri1(e1, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
            for cur in neigh21:
                suff |= self.search_1_hop_tri(e2, cur)
            for cur in neigh22:
                two_hop = self.search_2_hop_tri1(e2, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
        self.G_dataset.suff_kgs = suff
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        with open('../datasets/' + self.lang + '-en_f/base/test_triples_1_nec' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_nec' + method +thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
        with open('../datasets/'+ self.lang +'-en_f/base/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_suf' + method + thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

    def get_test_file_mask(self, file, thred, method=''):
        if 'lime' in file or 'shapley' in file:
            print('Save ranked triples in {}'.format(file))
            return
        elif 'lore' in file or 'anchor' in file:
            print('Save rules in {}'.format(file))
            return
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                # print(cur)
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        with open('../datasets/' + self.lang + '-en_f/base/test_triples_1_nec' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_nec' + method +thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
        with open('../datasets/'+ self.lang +'-en_f/base/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_suf' + method + thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

    def get_test_file_case(self, file, thred, e1, e2):
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(self.G_dataset.id_ent[cur[0]]), int(self.G_dataset.id_r[cur[1]]), int(self.G_dataset.id_ent[cur[2]])))
        new_kg = self.G_dataset.kgs - nec_tri
        with open('../datasets/zh-en_f/base/test_triples_1_nec' + thred, 'w') as f1, open('../datasets/zh-en_f/base/test_triples_2_nec' + thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        print(int(self.G_dataset.id_ent[e1]))
        print(self.G_dataset.tri[int(self.G_dataset.id_ent[e1])] , self.G_dataset.tri[int(self.G_dataset.id_ent[e2])])
        print(len(self.G_dataset.kgs))
        new_kg = (self.G_dataset.kgs - self.G_dataset.tri[int(self.G_dataset.id_ent[e1])] - self.G_dataset.tri[int(self.G_dataset.id_ent[e2])]) | nec_tri
        with open('../datasets/zh-en_f/base/test_triples_1_suf' + thred, 'w') as f1, open('../datasets/zh-en_f/base/test_triples_2_suf' + thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')


    def filter_tri_file(self, file, num, thred, method):
        nec_tri = set()
        test = []
        tri = []
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                # print(cur)
                cur = cur.strip().split('\t')
                if cur[0] == '0' and cur[1] == '0' and cur[2] == '0':
                    test += tri[: num]
                    tri = []
                else:
                    tri.append((int(cur[0]), int(cur[1]), int(cur[2])))
        for cur in test:
            nec_tri.add(cur)
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        with open('../datasets/' + self.lang + '-en_f/base/test_triples_1_nec' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_nec' + method +thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
        with open('../datasets/'+ self.lang +'-en_f/base/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/' + self.lang + '-en_f/base/test_triples_2_suf' + method + thred, 'w') as f2:
            for cur in new_kg:
                if cur[0] < self.split:
                    f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                else:
                    f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

if __name__ == '__main__':

    seed_torch()
    parser = argparse.ArgumentParser(f'arguments for Explanation Generation or Entity Alignment Repair')

    parser.add_argument('lang', type=str, help='which dataset', default='zh')
    parser.add_argument('method', type=str, help='Explanation Generation or Entity Alignment Repair', default='repair')
    parser.add_argument('--version', type=int, help='the hop num of candidate neighbor', default=1)
    parser.add_argument('--num', type=str, help='the len of explanation', default=15)
    
    args = parser.parse_args()
    lang = args.lang
    method = args.method
    if args.version:
        version = args.version
    if args.num:
        num = args.num
    pair = '/pair'
    device = 'cuda'
    Lvec = torch.load('../saved_model/Lvec.pt')
    Rvec = torch.load('../saved_model/Rvec.pt')
    model_name = 'mean_pooling'
    args = None
    in_d = None
    out_d = None
    m_adj=None
    e1=None
    e2=None
    device = 'cuda'
    if lang == 'zh':
    # saved_model = '../saved_model/base_model.pt'
        saved_model = '../saved_model/dual_amn_no_csls.pt'
        G_dataset = DBpDataset('../datasets/zh-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/zh-en_f/base/' + pair)
        split = len(read_list('../datasets/zh-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/zh-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=38960,
                        rel_size=6050,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)
    elif lang == 'zh2':
    # saved_model = '../saved_model/base_model.pt'
        saved_model = '../saved_model/zh2_model.pt'
        G_dataset = DBpDataset('../datasets/zh2-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/zh2-en_f/base/' + pair)
        split = len(read_list('../datasets/zh2-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/zh2-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=38960,
                        rel_size=6050,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)
    elif lang == 'ja':
        saved_model = '../saved_model/base_model_ja.pt'
        G_dataset = DBpDataset('../datasets/ja-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/ja-en_f/base/' + pair)
        split = len(read_list('../datasets/ja-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/ja-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=39594,
                        rel_size=4906,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)

    elif lang=='fr':
        saved_model = '../saved_model/base_model_fr.pt'
        G_dataset = DBpDataset('../datasets/fr-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/fr-en_f/base/' + pair)
        split = len(read_list('../datasets/fr-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/fr-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=39654,
                        rel_size=4224,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)

    elif lang=='w':
        saved_model = '../saved_model/base_model_w.pt'
        G_dataset = DBpDataset('../datasets/w-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/w-en_f/base/' + pair)
        split = len(read_list('../datasets/w-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/w-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=30000,
                        rel_size=836,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)
    elif lang=='w2':
        saved_model = '../saved_model/w2_model.pt'
        G_dataset = DBpDataset('../datasets/w2-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/w2-en_f/base/' + pair)
        split = len(read_list('../datasets/w2-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/w2-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=30000,
                        rel_size=836,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)
    elif lang=='y':
        saved_model = '../saved_model/base_model_y.pt'
        G_dataset = DBpDataset('../datasets/y-en_f/base', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/y-en_f/base/' + pair)
        split = len(read_list('../datasets/y-en_f/base/ent_dict1'))
        rsplit = len(read_list('../datasets/y-en_f/base/rel_dict1'))
        model = Encoder_Model(node_hidden=100,
                        rel_hidden=100,
                        node_size=30000,
                        rel_size=388,
                        device=device,
                        new_ent_nei = np.array([]),
                        dropout_rate=0,
                        ind_dropout_rate=0,
                        gamma=2,
                        lr=0.0001,
                        depth=2,
                        alpha=0.1,
                        beta=0.1, 
                        in_d=in_d, 
                        out_d=out_d,
                        m_adj = m_adj, e1=e1, e2=e2).to(device)

    evaluator = Evaluate(
                        test_dict=None,
                        valid_dict=None,
                        test_pairs=None,
                        new_test_pairs=None,
                        new_ent=None,
                        valid_pairs=None,
                        device=device,
                        eval_batch_size=None,
                        k=None,
                        dataset=None,
                        batch=512,
                        M=None)
    




    model.load_state_dict(torch.load(saved_model, map_location=device))
    model_name = 'load'
    model_name = 'mean_pooling'

    explain = EAExplainer(model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, lang, split, rsplit)
    explain.explain_EA(method,0.7, num, version, lang)
