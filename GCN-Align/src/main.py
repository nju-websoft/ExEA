from re import S
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import difflib
import torch
import math
import tqdm
import pandas as pd
import sys
# from model import Encoder_Model
import warnings
import argparse
import time
from collections import defaultdict
import argparse
from preprocessing import DBpDataset
from count import read_tri, read_link
import numpy as np
from anchor import anchor_tabular
import logging
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
import time
from tqdm import trange
import copy
from count import read_list
from itertools import combinations
import random
import numpy as np

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, add_remaining_self_loops
from torch_scatter import scatter_add
# --- torch_geometric Packages end ---
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter_sum
from LORE import util
from LORE.prepare_dataset import prepare_EA_dataset
from LORE import lore
from LORE.neighbor_generator import genetic_neighborhood


def comb(n, m):
    return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
        
    def forward(self, x, edge_index):
        fill_value = 1
        edge_weight=torch.ones((edge_index.size(1), ), dtype=None,
                                     device=edge_index.device)
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, x.size(0))

        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, name, hiddens, heads, activation, feat_drop, attn_drop, negative_slope, bias):
        super(Encoder, self).__init__()
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = activation
        self.feat_drop = feat_drop
        self.highways = nn.ModuleList()
        self.gat = GAT(hiddens[-1])
        for l in range(0, self.num_layers):
            if self.name == "gcn-align":
                self.gnn_layers.append(
                    GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l+1], improved=False, cached=False, bias=bias)
                )
            
            # elif self.name == "SLEF-DESIGN":
            #     self.gnn_layers.append(
            #         SLEF-DESIGN_Conv()
            #     )
            else:
                raise NotImplementedError("bad encoder name: " + self.name)
        if self.name == "naea":
            self.weight = Parameter(torch.Tensor(self.hiddens[0], self.hiddens[-1]))
            nn.init.xavier_normal_(self.weight)
        # if self.name == "SLEF-DESIGN":
        #     '''SLEF-DESIGN: extra parameters'''

    def forward(self, edges, x, r=None):
        edges = edges.t()
        
        for l in range(self.num_layers):
            x = F.dropout(x, p=self.feat_drop)
            x_ = self.gnn_layers[l](x, edges)
            x = x_
            if l != self.num_layers - 1:
                x = self.activation(x)
        return x            

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, "\n".join([layer.__repr__() for layer in self.gnn_layers]))

class GCNAlign_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GCNAlign_GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.mul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class Proxy:
    def __init__(self, embed1, embed2):
        self.embed1 = embed1
        self.embed2 = embed2


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
    def __init__(self, model, num_players, players, split, n1, n2):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.n1 = n1
        self.n2 = n2

    def MTC(self, num_simulations):
        shapley_values = np.zeros(self.num_players)
        for _ in range(num_simulations):
        # 生成随机排列的玩家列表
            players = np.random.permutation(self.num_players)
            # print(players, split)
            # 初始化联盟价值和玩家计数器
            coalition_value = 0
            player_count = 0
            tri1 = []
            tri2 = []
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                if player < self.split:
                    tri1.append([player + 1, 0])
                else:
                    tri2.append([player - self.split + 1, 0])
                if len(tri1) == 0 or len(tri2) == 0:
                    coalition_value_with_player = 0
                else:
                    coalition_value_with_player = F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0)
                # print(tri1, tri2)
                # print(coalition_value_with_player)
                marginal_contribution = coalition_value_with_player - coalition_value

                # 计算当前玩家的 Shapley 值
                shapley_values[player] += marginal_contribution / num_simulations

                # 更新联盟价值和玩家计数器
                coalition_value = coalition_value_with_player
                # player_count += 1
        return shapley_values

class KernelSHAP:
    def __init__(self, model, num_players, players, split, n1, n2, embed, e1, e2):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.n1 = n1
        self.n2 = n2
        self.embed = embed
        self.e1 = e1
        self.e2 = e2

    def sim_kernel(self, tri1, tri2):
        z = len(tri1) + len(tri2)
        if z == self.num_players:
            return 1
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        # print(z, self.num_players)
        sim = (self.num_players - 1) / (comb(self.num_players, z) * z * (self.num_players - z))

        return sim


    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append([player + 1, 0])
                else:
                    tri2.append([player - self.split + 1, 0])
            
            sim = self.sim_kernel(tri1, tri2)
            if sim == 0:
                continue
            mask.append(cur_mask)
            Y.append(F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0))
            
            pi.append(float(sim))
                    
        
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


class Anchor:
    def __init__(self, model, num_players, players, split, n1, n2, embed, e1, e2):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.n1 = n1
        self.n2 = n2
        self.embed = embed
        self.e1 = e1
        self.e2 = e2

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
                            tri1.append([i + 1, 0])
                        else:
                            tri2.append([i - self.split + 1, 0])
                tri.append((tri1, tri2))
            return tri
        def predict(tri):
            res = []
            # print(tri)
            for tri1, tri2 in tri:
                # print(tri1, tri2)
                if len(tri1) and len(tri2):
                    sim = F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0)
                else:
                    sim = 0
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
            mask.append(cur_mask)
                    
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

class BLACK:
    def __init__(self, model, split, alpha, n1, n2):
        self.model = model
        self.split = split
        self.alpha = alpha
        self.n1 = n1
        self.n2 = n2
    
    def predict(self, X):
        tri = []
        # print(X)
        for x in X:
            tri1 = []
            tri2 = []
            for i in range(len(x)):
                if x[i] == 1:
                    if i < self.split:
                        tri1.append([i + 1, 0])
                    else:
                        tri2.append([i - self.split + 1, 0])
            tri.append((tri1, tri2))
        res = []
        # print(tri)
        for tri1, tri2 in tri:
            # print(tri1, tri2)
            if len(tri1) and len(tri2):
                sim = F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0)
            else:
                sim = 0
            if sim >= self.alpha:
                c = 1
            else:
                c = 0
            res.append(c)
        return np.array(res)
        
class LORE:
    def __init__(self, model, num_players, players, split, n1, n2, embed, e1, e2, lang):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.n1 = n1
        self.n2 = n2
        self.embed = embed
        self.e1 = e1
        self.e2 = e2
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
                    tri1.append([player + 1, 0])
                else:
                    tri2.append([player - self.split + 1, 0])
            # print(tri1, tri2)
            if len(tri1) and len(tri2):
                sim = F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0)
            else:
                sim = 0
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
        blackbox = BLACK(self.model, self.split, alpha, self.n1, self.n2)
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


class LIME:
    def __init__(self, model, num_players, players, split, n1, n2, embed, e1, e2):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.n1 = n1
        self.n2 = n2
        self.embed = embed
        self.e1 = e1
        self.e2 = e2

    def sim_kernel(self, tri1, tri2):
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        tmp_e1 = self.model(torch.Tensor(tri1).long(), self.n1)[0]
        tmp_e2 = self.model(torch.Tensor(tri2).long(), self.n2)[0]
        
        sim1 = F.cosine_similarity(tmp_e1, self.embed[self.e1], dim=0)
        sim2 = F.cosine_similarity(tmp_e2, self.embed[self.e2], dim=0)
        # print(tri1, tri2, sim1, sim2)
        return (sim1 + sim2) / 2


    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append([player + 1, 0])
                else:
                    tri2.append([player - self.split + 1, 0])
            
            sim = self.sim_kernel(tri1, tri2)
            if sim == 0:
                continue
            mask.append(cur_mask)
            Y.append(F.cosine_similarity(self.model(torch.Tensor(tri1).long(), self.n1)[0], self.model(torch.Tensor(tri2).long(), self.n2)[0], dim=0))
            
            pi.append(float(sim))
                    
        
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
    def __init__(self, model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr=0, lang='zh'):
        super(EAExplainer, self).__init__()
        self.model_name = model_name
        if model_name == 'load':
            self.model = model
        self.dist = nn.PairwiseDistance(p=2)
        self.split = split
        self.splitr = splitr
        self.lang = lang
        self.conflict_r_pair = G_dataset.conflict_r_pair
        self.conflict_id = G_dataset.conflict_id
        if lang == 'zh':
            e_embed = np.load('../saved_model/ent_zh.npy')
        elif lang == 'ja':
            e_embed = np.load('../saved_model/ent_ja.npy')
        elif lang == 'fr':
            e_embed = np.load('../saved_model/ent_fr.npy')
        elif lang == 'de':
            e_embed = np.load('../saved_model/ent_de.npy')
        elif lang == 'y':
            e_embed = np.load('../saved_model/ent_y.npy')
        elif lang == 'w':
            e_embed = np.load('../saved_model/ent_w.npy')

        # print(mapping.shape)
        # exit(0)
        # test_embeds1_mapped = np.matmul(embeds1, mapping)
        self.G_dataset = G_dataset
        # self.embed = torch.Tensor(mapping).cuda()
        self.embed = torch.Tensor(e_embed)
        self.e_embed = self.embed
        self.e_sim =self.cosine_matrix(self.embed[:self.split], self.embed[self.split:])
        self.G_dataset = G_dataset
        self.r_embed = self.proxy_r()
        self.get_r_map(lang)
        self.conflict_r_pair = G_dataset.conflict_r_pair
        
        self.Lvec = Lvec
        self.Rvec = Rvec
        if self.Lvec is not None:
            self.Lvec.requires_grad = False
            self.Rvec.requires_grad = False
        self.test_indices = test_indices
        self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.test_indices = test_indices
        self.test_pair = G_dataset.test_pair
        self.train_pair = G_dataset.train_pair
        self.model_pair = G_dataset.model_pair
        self.model_link = G_dataset.model_link
        self.train_link = G_dataset.train_link
        self.test_link = G_dataset.test_link
        self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.evaluator = evaluator
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


    def proxy_r(self):
        r_list = defaultdict(list)

        for (h, r, t) in self.G_dataset.kg1:
            r_list[int(r)].append([int(h), int(t)])
        for (h, r, t) in self.G_dataset.kg2:
            r_list[int(r)].append([int(h), int(t)])
        
        r_embed = torch.Tensor(len(self.G_dataset.rel), self.embed.shape[1])
        for i in range(r_embed.shape[0]):
            cur_ent = torch.Tensor(r_list[i]).reshape(2,-1)
            h = self.embed[cur_ent[0].long()]
            t = self.embed[cur_ent[1].long()]
            r_embed[i] = (h - t).mean(dim=0)
        return r_embed


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

            with open('../datasets/D_Y/rel_links') as f:
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
            with open('../datasets/D_W/rel_links') as f:
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


    
    def explain_EA(self, method, thred, num,  version = ''):
        # num=100
        self.version = version
        if method == 'EG':
            if self.lang == 'zh':
                if self.version == 1:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_z_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                            self.get_test_file_mask_two('../datasets/dbp_z_e/exp_ours', str(version))
            elif self.lang == 'ja':
                if self.version == 1:
                    with open('../datasets/dbp_j_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_j_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_j_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                            self.get_test_file_mask_two('../datasets/dbp_j_e/exp_ours', str(version))
            elif self.lang == 'fr':
                if self.version == 1:
                    with open('../datasets/dbp_f_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_f_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_f_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                            self.get_test_file_mask_two('../datasets/dbp_f_e/exp_ours', str(version))
            elif self.lang == 'y':
                if self.version == 1:
                    with open('../datasets/D_Y/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_Y/exp_ours', str(version))
                else:
                    with open('../datasets/D_Y/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                            self.get_test_file_mask_two('../datasets/D_Y/exp_ours', str(version))
            elif self.lang == 'w':
                if self.version == 1:
                    with open('../datasets/D_W/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_W/exp_ours', str(version))
                else:
                    with open('../datasets/D_W/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                            self.get_test_file_mask_two('../datasets/D_W/exp_ours', str(version))
        
        elif method == 'shapley':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        
                self.get_test_file_mask('../datasets/dbp_z_e/exp_shapley', str(version), method)
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        
                self.get_test_file_mask('../datasets/dbp_j_e/exp_shapley', str(version), method)
            elif self.lang == 'y':
                if self.version == 1:
                    with open('../datasets/D_Y/exp_shapley', 'w') as f:
                        for i in trange(len(self.test_indices)):
                            gid1, gid2 = self.test_indices[i]
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            # exp = self.explain(gid1, gid2)
                            tri = self.explain_shapely(gid1, gid2)
                            
                            for cur in tri:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_Y/exp_shapley', str(version))
            elif self.lang == 'w':
                if self.version == 1:
                    with open('../datasets/D_W/exp_shapley', 'w') as f:
                        for i in trange(len(self.test_indices)):
                            gid1, gid2 = self.test_indices[i]
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            # exp = self.explain(gid1, gid2)
                            tri = self.explain_shapely(gid1, gid2)
                            for cur in tri: 
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_W/exp_shapley', str(version))
            else:
                with open('../datasets/dbp_f_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        
                self.get_test_file_mask('../datasets/dbp_f_e/exp_shapley', str(version), method)
        
        elif method == 'lime':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        f.write('0' + '\t' + '0' + '\t' + '0' + '\n')
                        
                self.get_test_file_mask('../datasets/dbp_z_e/exp_lime', str(version), method)
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        f.write('0' + '\t' + '0' + '\t' + '0' + '\n')
                self.get_test_file_mask('../datasets/dbp_j_e/exp_lime', str(version), method)
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        f.write('0' + '\t' + '0' + '\t' + '0' + '\n')
                self.get_test_file_mask('../datasets/dbp_f_e/exp_lime', str(version), method)
            elif self.lang == 'y':
                if self.version == 1:
                    with open('../datasets/D_Y/exp_lime', 'w') as f:
                        for i in trange(len(self.test_indices)):
                            gid1, gid2 = self.test_indices[i]
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            # exp = self.explain(gid1, gid2)
                            tri = self.explain_lime(gid1, gid2)
                            for cur in tri:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_Y/exp_lime', str(version))
            elif self.lang == 'w':
                if self.version == 1:
                    with open('../datasets/D_W/exp_lime', 'w') as f:
                        for i in trange(len(self.test_indices)):
                            gid1, gid2 = self.test_indices[i]
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            # exp = self.explain(gid1, gid2)
                            tri = self.explain_lime(gid1, gid2)
                            for cur in tri:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_W/exp_lime', str(version))
        elif method == 'anchor':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')             
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')    
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'w':
                with open('../datasets/D_W/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
            print('Save rules to exp_anchor in the corresponding dataset directory')
        elif method == 'lore':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')             
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')    
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'w':
                with open('../datasets/D_W/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
            print('Save rules to exp_lore in the corresponding dataset directory')
        elif method == 'repair':
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
                pair, score= self.get_pair_score(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                if len(pair) > 0:
                    node[str(gid1) + '\t' + str(gid2)] = pair
                    node_set[str(gid1) + '\t' + str(gid2)] = score
            
            c_set, new_model_pair, count1 = self.conflict_count(self.model_pair)
            
            kg1, _, cur_pair = self.conflict_solve(c_set, node_set, new_model_pair, kg2, count1, ground)
            # kg1 |= new_kg1
            # print(len(kg1), len(cur_pair))
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
                _, kg1 = self.adjust(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100)
                if len(kg1) >= last_len:
                    break
                # print(len(cur_pair))
                # print(len(delete_pair & ans_pair), len(delete_pair) )
                print('current acc: {}'.format(len(cur_pair & ans_pair) / len(ans_pair)))
            # find low confidence conflict
            last_len1 = None
            print('start low confidence conflict solving')
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
                # print(len(cur_pair & ans_pair) / len(ans_pair))
                # while(len(kg1) > 0):
                if last_len1 != None and len(kg1) >= last_len1:
                    break
                else:
                    last_len1 = len(kg1)
                cur_link = {}
                cur_link_r = {}
                # print(len(cur_pair))
                # print(len(kg1))
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    cur_link_r[int(p[1])] = int(p[0])
                last_len = len(kg1)
                # _, kg1 = self.adjust_conflict(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100, conflict_link)
                _, kg1 = self.adjust_no_explain(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100, 1)
                # if len(kg1) >= last_len:
                    # break
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
                score = -1e5
                max_e = None

                for e in c_set[ent]:
                    cur_score = node_set[str(e) + '\t' + str(ent)] # +  0.5 * self.e_sim[e, ent - self.split]
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
            if kg1[i] not in cur_link:
                new_kg1.add(kg1[i])
        # print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

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
        # print(len(new_pair & set(ans_pair)) / 10500)
        return new_pair, set(kg2)

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
            for j in range(K):
                e2 = kg2[rank[i][j]]
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    _, cur_score = self.get_pair_score5(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    break
                else:
                    
                    _, cur_score = self.get_pair_score5(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                    # cur_score =  self.e_sim[kg1[i], e2- self.split]
                    _, other = self.get_pair_score5(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                    # other = self.e_sim[cur_link_r[e2], e2- self.split]
                    if other < cur_score:
                        # print(kg1[i], cur_link_r[e2])
                        cur_pair.remove((cur_link_r[e2], e2))
                        cur_pair.add((kg1[i], e2))
                        new_kg1.add(cur_link_r[e2])
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        break
            if kg1[i] not in cur_link:
                new_kg1.add(kg1[i])
            # new_pair.add((kg1[i], e2))
        # print(len(new_kg1))
        # print(new_kg1)
        count = 0
        return new_pair, new_kg1


    def adjust1(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_set=None):
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
            for j in range(K):
                e2 = kg2[rank[i][j]]
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    break
                else:
                    _, cur_score = self.get_pair_score3(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set)
                    # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                    # cur_score =  self.e_sim[kg1[i], e2- self.split]
                    _, other = self.get_pair_score3(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set)
                    # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                    # other = self.e_sim[cur_link_r[e2], e2- self.split]
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
        return new_pair, new_kg1

    def adjust_conflict(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, conflict_link):
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
        # print(len(kg1))
        # print(rank.shape[0])
        for i in range(rank.shape[0]):
            # print(kg1[i])
            if kg1[i] in conflict_link:
                _, cur_score = self.get_pair_score5(kg1[i], conflict_link[kg1[i]], r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                for j in range(K):
                    e2 = kg2[rank[i][j]]
                    # if e2 != conflict_link[kg1[i]]:
                        # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                    if e2 != conflict_link[kg1[i]]:
                        if e2 not in cur_link_r:   
                            # _, cur_score = self.get_pair_score(kg1[i], conflict_link[kg1[i]], r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score5(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            if other > cur_score:
                                cur_pair.add((kg1[i], e2))
                                if (kg1[i], conflict_link[kg1[i]]) in cur_pair:
                                    cur_pair.remove((kg1[i], conflict_link[kg1[i]]))
                                    del cur_link_r[conflict_link[kg1[i]]]
                                    # print('del', conflict_link[kg1[i]])
                                cur_link_r[e2] = kg1[i]
                                cur_link[kg1[i]] = e2
                                
                                break
                        else:
                            
                            _, other_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            if other_score >= cur_score:
                                # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                                # cur_score =  self.e_sim[kg1[i], e2- self.split]
                                _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                                # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                                # other = self.e_sim[cur_link_r[e2], e2- self.split]
                                if other < other_score:
                                    # print(kg1[i], cur_link_r[e2])
                                    cur_pair.remove((cur_link_r[e2], e2))
                                    if (kg1[i], conflict_link[kg1[i]]) in cur_pair:
                                        cur_pair.remove((kg1[i], conflict_link[kg1[i]]))
                                        del cur_link_r[conflict_link[kg1[i]]]
                                        # print('del', conflict_link[kg1[i]])
                                    cur_pair.add((kg1[i], e2))
                                    new_kg1.add(cur_link_r[e2])
                                    cur_link_r[e2] = kg1[i]
                                    cur_link[kg1[i]] = e2
                                    
                                    break
            else:
                for j in range(K):
                    e2 = kg2[rank[i][j]]
                    # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                    if e2 not in cur_link_r:   
                        cur_pair.add((kg1[i], e2))
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
                    else:
                        
                        _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                        # cur_score =  self.e_sim[kg1[i], e2- self.split]
                        _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                        # other = self.e_sim[cur_link_r[e2], e2- self.split]
                        if other < cur_score:
                            # print(kg1[i], cur_link_r[e2])
                            cur_pair.remove((cur_link_r[e2], e2))
                            cur_pair.add((kg1[i], e2))
                            new_kg1.add(cur_link_r[e2])
                            cur_link_r[e2] = kg1[i]
                            cur_link[kg1[i]] = e2
                            
                            break
                # new_pair.add((kg1[i], e2))
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        for cur in new_kg1:
            if self.model_link[str(cur)] != self.test_link[str(cur)]:
                count += 1
        print(count)
        return new_pair, new_kg1
   
                    
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
                r_score += cur_score
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        return pair_node, score
    
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
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
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
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
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
    
    def get_pair_score2(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set1, rule_set2, thred):
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
        r_pair = []
        pair = list(pair)
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
            judge = 0
            cur_r_pair = []
            for i in range(len(pair_r)):
                pr = pair_r[i]
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    cur_score_r = min(cur_score_r,r1_func[str(tri1[pr[0]][1])])
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    cur_score_r = min(cur_score_r,r1_func_r[str(tri1[pr[0]][1])])
                    direct = 1
                    r_1 = (tri1[pr[0]][1], 1)
                
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                    cur_score_r = min(cur_score_r,r2_func[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 0)
                    if direct == 1:
                        judge = 1
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    cur_score_r = min(cur_score_r,r2_func_r[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 1)
                    if direct == 0:
                        judge = 1
                cur_r_pair.append((r_1, r_2))
                if cur_score_r >= thred and judge == 0:
                    # rule_set1[(tri1[pr[0]][1], tri2[pr[1]][1])].add((e1, e2))
                    rule_set1[(r_1, r_2)].add((e1, e2))
                r_score += cur_score
            r_pair.append(cur_r_pair)
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        
        for i in range(len(r_pair) - 1):
            for j in range(i + 1, len(r_pair)):
                for cur1 in r_pair[i]:
                    for cur2 in r_pair[j]:
                        # if str(pair[i][0]) in self.train_link:
                        rule_set2[(cur1[0], cur2[1])].add((e1, e2))
                        rule_set2[(cur2[0], cur1[1])].add((e1, e2))
                        # else:
                            # rule_set2[(cur1[0], cur2[1])].add((pair[i][0], pair[j][1]))
                            # rule_set2[(cur2[0], cur1[1])].add((e1, e2))
                            # rule_set2[(cur2[0], cur1[1])].add((pair[i][0], pair[j][1]))
        return pair_node, score

    def get_pair_score3(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set):
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
        r_pair = []
        pair = list(pair)
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
            judge = 0
            cur_r_pair = []
            for i in range(len(pair_r)):
                pr = pair_r[i]
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    
                    direct = 1
                    r_1 = (tri1[pr[0]][1], 1)
                
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                    
                    r_2 = (tri2[pr[1]][1], 0)
                    if direct == 1:
                        judge = 1
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                   
                    r_2 = (tri2[pr[1]][1], 1)
                    if direct == 0:
                        judge = 1
                cur_r_pair.append((r_1, r_2))
                # if (r_1, r_2) in rule_set:
                cur_score = 1
                r_score += cur_score
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        return pair_node, score

    def get_pair_score4(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set=None):
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
        score_list = []
        score = 0
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
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
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])

                    r_1 = (tri1[pr[0]][1], 1)
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])       
                    r_2 = (tri2[pr[1]][1], 0)
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 1)
                
                # map_r2 = None
                # map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                    # map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                    # print(self.r_map1[tri1[pr[0]][1]])
                    # map_r1 = int(self.r_map1[tri1[pr[0]][1]])
                map_r1 = self.r_map1[tri1[pr[0]][1]]

                neigh_r2.add(map_r1)
                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                r_score += cur_score
                cur_r_pair.append((r_1, r_2))
            
            for pr in pair_r:
                for cur in neigh_r1:
                    if ((tri1[pr[0]][1], cur) in self.conflict_r_pair or (cur, tri1[pr[0]][1]) in self.conflict_r_pair):
                        r_score = 0
                for cur in neigh_r2:
                    if ((tri2[pr[1]][1], cur) in self.conflict_r_pair or (cur, tri2[pr[1]][1]) in self.conflict_r_pair):
                        r_score = 0
                if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    cur_score = 0
            
            r_pair.append(cur_r_pair)
            score_list.append(r_score)
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        for i in range(len(r_pair) - 1):
            for j in range(i + 1, len(r_pair)):
                for cur1 in r_pair[i]:
                    for cur2 in r_pair[j]:
                        cur_sim = self.e_sim[pair[i][0]][pair[j][1] - self.split]
                        # print(cur1[0][1], cur2[1][1], self.r_map1[cur1[0][0]], cur1[0][0], cur2[1][0], self.r_map2[cur2[1][0]])
                        if cur1[0][1] == cur2[1][1] and (self.r_map1[cur1[0][0]] == cur2[1][0] or cur1[0][0] == self.r_map2[cur2[1][0]]):
                            # print('exist same relation')
                            # print(self.G_dataset.r_dict[cur1[0][0]], self.G_dataset.r_dict[cur1[1][0]])
                            if (cur1[0][1] == 0 and min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]) >= 1):
                                print('exist same relation and r_func high')
                                # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2])
                                # score_list[i] *= min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])])
                                # score_list[j] *= min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]) 
                                # score_list[i] *= (min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # score_list[j] *= (min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                score_list[i] = 0
                                score_list[j] = 0
                                # self.e_sim[e1][e2 - self.split] *= (0.5 + 1 - min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]))
                                score_list.append(min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # print(self.e_sim[pair[i][0]][pair[j][1] - self.split])
                                pair.append((pair[i][0], pair[j][1]))
                                # self.e_sim[pair[i][0]][pair[j][1] - self.split] = max(min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]), cur_sim)
                            elif (cur1[0][1] == 1 and min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]) >= 1):
                                print('exist same relation and r_func high')
                                # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2])
                                # self.e_sim[pair[i][0]][pair[j][1] - self.split] = max(min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]), cur_sim)
                                # score_list[i] *= min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])])
                                self.e_sim[e1][e2 - self.split] *= (0.5 + 1 - min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # score_list[j] *= min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])])
                                score_list[i] = 0
                                score_list[j] = 0
                                # print(self.e_sim[pair[i][0]][pair[j][1] - self.split])
                                score_list.append(min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]))
                                pair.append((pair[i][0], pair[j][1]))
        
        for i in range(len(score_list)):
            score += score_list[i] * float(self.e_sim[pair[i][0]][pair[i][1] - self.split])
            # score += score_list[i] 
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
            return set(), 0
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
            

        return pair_node, score


    def get_pair_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
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
            return 0
        
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
            fact_conflict = 0
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] != e1:
                    direct = 1
                if tri2[pr[1]][0] == e2:
                    if direct == 1:
                        fact_conflict += 1
                else:
                    if direct == 0:
                        fact_conflict += 1
        return fact_conflict

    def get_r_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
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
            return 0
        
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
            fact_conflict = 0
            # print(len(self.conflict_r_pair))
            for pr in pair_r:
                map_r2 = None
                map_r1 = None
                if tri2[pr[1]][1] in self.r_map2:
                    map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                if tri1[pr[0]][1] in self.r_map1:
                    map_r1 = int(self.r_map1[tri1[pr[0]][1]])

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # print(self.G_dataset.r_dict[tri1[pr[0]][1]], self.G_dataset.r_dict[tri2[pr[1]][1]])
                    # print(map_r1, map_r2)
                    # if map_r1:
                        # print(self.G_dataset.r_dict[map_r1])
                    # if map_r2:
                        # print(self.G_dataset.r_dict[map_r2])
                    # print('--------------------------')
                    return 1
        return 0

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
                # print(self.G_dataset.ent_dict[ent])
                # for cur in self.G_dataset.gid[ent]:
                    # self.read_triple_name(cur)
                # print('************************')
                count += 1
                for e in conflict_set[ent]:
                    new_model_pair[e] = None
                    if self.test_link[str(e)] == str(ent):
                        count1 += 1
        return conflict_set, new_model_pair, count1

    
    def get_r_func(self):
        if self.lang == 'y':
            tri1 = read_tri('../datasets/D_Y/triples_1')
            tri2 = read_tri('../datasets/D_Y/triples_2')
            r, _ = read_link('../datasets/D_Y/rel_dict')
        elif self.lang == 'w' or self.lang == 'w2':
            tri1 = read_tri('../datasets/D_W/triples_1')
            tri2 = read_tri('../datasets/D_W/triples_2')
            r, _ = read_link('../datasets/D_W/rel_dict')
        elif self.lang == 'zh2':
            tri1 = read_tri('../datasets/z_e_2/triples_1')
            tri2 = read_tri('../datasets/z_e_2/triples_2')
            r, _ = read_link('../datasets/z_e_2/rel_dict')
        elif self.lang == 'zh_2':
            tri1 = read_tri('../datasets/dbp_z_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_z_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_z_e_wrong/rel_dict')
        elif self.lang == 'ja2':
            tri1 = read_tri('../datasets/dbp_j_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_j_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_j_e_wrong/rel_dict')
        elif self.lang == 'fr2':
            tri1 = read_tri('../datasets/dbp_f_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_f_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_f_e_wrong/rel_dict')
        elif self.lang == 'zh':
            tri1 = read_tri('../datasets/dbp_z_e/triples_1')
            tri2 = read_tri('../datasets/dbp_z_e/triples_2')
            r, _ = read_link('../datasets/dbp_z_e/rel_dict')
        elif self.lang == 'ja':
            tri1 = read_tri('../datasets/dbp_j_e/triples_1')
            tri2 = read_tri('../datasets/dbp_j_e/triples_2')
            r, _ = read_link('../datasets/dbp_j_e/rel_dict')
        elif self.lang == 'fr':
            tri1 = read_tri('../datasets/dbp_f_e/triples_1')
            tri2 = read_tri('../datasets/dbp_f_e/triples_2')
            r, _ = read_link('../datasets/dbp_f_e/rel_dict')
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
        '''
        for cur in r1_func:
            if r1_func[cur] == 1:
                print(cur)
        for cur in r1_func_r:
            if r1_func_r[cur] == 1:
                print(cur)
        for cur in r2_func:
            if r2_func[cur] == 1:
                print(cur)
        for cur in r2_func_r:
            if r2_func_r[cur] == 1:
                print(cur)
        exit(0)
        '''
        return r1_func, r1_func_r, r2_func, r2_func_r
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
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
        
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            new_tri1 = []
            new_tri2 = []
            
            for pr in pair_r:
                new_tri1.append(tri1[pr[0]])
                new_tri2.append(tri2[pr[1]])

            tri1_list += new_tri1
            tri2_list += new_tri2
        return tri1_list, tri2_list, pair

    def explain_ours5(self, e1, e2):
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

    def search_2_hop_tri(self, e, tar):
        neigh1 = self.get_1_hop(e)
        tri2 = []
        cur1 = set()
        cur2 = set()
        for ent in neigh1:
            neigh2 = self.get_1_hop(ent)
            if tar in neigh2:
                t1 = self.search_1_hop_tri(e, ent)
                t2 = self.search_1_hop_tri(ent, tar)
                tri2.append((t1, t2))
                cur1 |= t1
                cur2 |= t2
        # print(cur1, cur2)
        return cur1, cur2

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

    def pattern_process(self, e, l=2):
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
            for cur in self.G_dataset.gid[e]:
                p.append(cur)
                if cur[0] == e1:
                    p_embed[i] = torch.cat((self.embed[cur[2]], self.r_embed[cur[1]]), dim=0)
                else:
                    p_embed[i] = torch.cat((self.embed[cur[0]], self.r_embed[cur[1]]), dim=0)

                i += 1
        return p, p_embed

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
        return torch.Tensor(new_tri).long().cuda(), e_dict, r_dict, list(kg1_index), list(kg2_index), tri1, tri2, e_dict_r, r_dict_r

    def Trans_Process(self, e):
        i = 0
        p_embed = torch.zeros(len(self.G_dataset.gid[e]), self.embed.shape[1]) 
        for cur in self.G_dataset.gid[e]:
            if cur[0] == e:
                p_embed[i] = self.embed[cur[2]] +  self.r_embed[cur[1]]
            else:
                p_embed[i] = self.embed[cur[0]] +  self.r_embed[cur[1]]
            i += 1
        return p_embed

    def bidirect_match(self, neigh_pre1, neigh_pre2, neigh_list1=None, neigh_list2=None, sim=None):
        res = []
        for i in range(neigh_pre1.shape[0]):
            select = neigh_pre1[i][0]
            if i == neigh_pre2[select][0]:
                # res.append([[i, select], sim[i][select]])
                res.append((i, int(select)))
        # res.sort(key=lambda x:x[1], reverse=True)
        # match = []
        # for cur in res:
            # match.append(cur[0])
        return res

    def greedy_match(self, neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, sim):
        res = []
        for i in range(neigh_pre1.shape[0]):
            select = neigh_pre1[i][0]
            res.append((i, int(select)))
        return res

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
        r_embed = torch.zeros(len(r_dict), self.r_embed.shape[1]).cuda()
        for cur in e_dict:
            e_embed[e_dict[cur]] = self.e_embed[cur].cuda()
        # r_embed[0] = self.r_embed[0].cuda()
        # r_embed[len(r_dict)] = self.r_embed[int(self.r_embed.shape[0] / 2)].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]] = self.r_embed[cur].cuda()
           
        p_embed1 = self.Trans_Process(e1)
        p_embed2 = self.Trans_Process(e2)
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
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        if self.lang == 'zh':
            with open('../datasets/dbp_z_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'ja':
            with open('../datasets/dbp_j_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_j_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'y':
            with open('../datasets/D_Y/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/D_Y/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/D_Y/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/D_Y/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'w':
            with open('../datasets/D_W/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/D_W/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/D_W/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/D_W/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        else:
            with open('../datasets/dbp_f_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_f_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
    
    def get_test_file_mask_two(self, file, thred, method=''):
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
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
        if self.lang == 'zh':
            with open('../datasets/dbp_z_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'ja':
            with open('../datasets/dbp_j_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_j_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        else:
            with open('../datasets/dbp_f_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_f_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')  

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

    
    def search_1_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if t[0] == target or t[2] == target:
                tri.add((t[0], t[1], t[2]))
                continue

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

    def change_to_list(self, exp):
        exp_list = []
        for cur in exp:
            exp_list.append(list(cur))
        return exp_list


    def compute_coal_value(self, gid1, gid2, c):
        c = torch.Tensor(c)
        if self.mapping is not None:
            neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        else:
            neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        neigh1 = F.normalize(neigh1, dim = 0)
        neigh2 = self.embed[c.t()[1].long()].mean(dim = 0)
        neigh2 = F.normalize(neigh2, dim = 0)
        return F.cosine_similarity(neigh1, neigh2, dim=0)

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

    def explain_anchor(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        n1 = self.embed[neigh_list1]
        n2 = self.embed[neigh_list2]
        n1 = torch.cat((self.embed[e1].unsqueeze(0), n1), dim=0)
        n2 = torch.cat((self.embed[e2].unsqueeze(0), n2), dim=0)
        explain = Anchor(self.model, len(n1) + len(n2) - 2, list(range(len(n1) + len(n2) - 2)), len(n1) - 1, n1, n2, self.embed, e1, e2)
        rule = explain.compute(100, 0.6)
        need, delete = self.analyze_rule(rule)
        need_tri = []
        delete_tri = []
        i = 0
        for cur in need:
            if cur < len(neigh_list1):
                need_tri += self.G_dataset.gid[neigh_list1[cur]]
            else:
                need_tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
        for cur in delete:
            if cur < len(neigh_list1):
                delete_tri += self.G_dataset.gid[neigh_list1[cur]]
            else:
                delete_tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
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
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        n1 = self.embed[neigh_list1]
        n2 = self.embed[neigh_list2]
        n1 = torch.cat((self.embed[e1].unsqueeze(0), n1), dim=0)
        n2 = torch.cat((self.embed[e2].unsqueeze(0), n2), dim=0)
        explain = LORE(self.model, len(n1) + len(n2) - 2, list(range(len(n1) + len(n2) - 2)), len(n1) - 1, n1, n2, self.embed, e1, e2, self.lang)
        rule, class_name = explain.compute(50, 0.6)
        if rule == None:
            need_tri = []
            delete_tri = []
        else:
            need, delete = self.analyze_rule_lore(rule, class_name)
            need_tri = []
            delete_tri = []
            i = 0
            for cur in need:
                if cur < len(neigh_list1):
                    need_tri += self.G_dataset.gid[neigh_list1[cur]]
                else:
                    need_tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
            for cur in delete:
                if cur < len(neigh_list1):
                    delete_tri += self.G_dataset.gid[neigh_list1[cur]]
                else:
                    delete_tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule

    def explain_lime(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        n1 = self.embed[neigh_list1]
        n2 = self.embed[neigh_list2]
        n1 = torch.cat((self.embed[e1].unsqueeze(0), n1), dim=0)
        n2 = torch.cat((self.embed[e2].unsqueeze(0), n2), dim=0)
        lime = LIME(self.model, len(n1) + len(n2) - 2, list(range(len(n1) + len(n2) - 2)), len(n1) - 1, n1, n2, self.embed, e1, e2)
        res = lime.compute(100)
        # res = res.squeeze(1)
        res = torch.Tensor(res)
        score, indices = res.sort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        for cur in indices:
            if cur < len(neigh_list1):
                tri += self.G_dataset.gid[neigh_list1[cur]]
            else:
                tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
        tri.append((0,0,0))
        return tri

    def explain_shapely(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        n1 = self.embed[neigh_list1]
        n2 = self.embed[neigh_list2]
        n1 = torch.cat((self.embed[e1].unsqueeze(0), n1), dim=0)
        n2 = torch.cat((self.embed[e2].unsqueeze(0), n2), dim=0)
        sim_num = 100
        if len(n1) + len(n2) - 2 > 100:
            Shapley = KernelSHAP(self.model, len(n1) + len(n2) - 2, list(range(len(n1) + len(n2) - 2)), len(n1) - 1, n1, n2, self.embed, e1, e2)
            res = Shapley.compute(100)
            # res = res.squeeze(1)
            res = torch.Tensor(res)
            score, res = res.sort(descending=True)
        else:
            Shapley = Shapley_Value(self.model, len(n1) + len(n2) - 2, list(range(len(n1) + len(n2) - 2)), len(n1) - 1, n1, n2)
            shapley_value = Shapley.MTC(sim_num)
            res = torch.Tensor(shapley_value).argsort(descending=True)
        tri = []
        for cur in res:
            if cur < len(neigh_list1):
                tri += self.G_dataset.gid[neigh_list1[cur]]
            else:
                tri += self.G_dataset.gid[neigh_list2[cur - len(neigh_list1)]]
        tri.append((0,0,0))
        return tri
        # print(shapley_value)
        # exit(0)


    def Shapely_value_debug(self, exp, gid1, gid2):
        exp = self.change_to_list(exp)
        num_exp = len(exp)

        all_coal = [list(coal) for i in range(num_exp) for coal in combinations(exp, i + 1)]
        # print(num_exp)
        shapely_value = []
        for e in exp:
            # print(e)
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
            print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.ent_dict[cur[0][1]])
        return new_exp        

    def cosine_matrix(self, A, B):
        A_sim = torch.mm(A, B.t())
        a = torch.norm(A, p=2, dim=-1)
        b = torch.norm(B, p=2, dim=-1)
        cos_sim = A_sim / a.unsqueeze(-1)
        cos_sim /= b.unsqueeze(-2)
        return cos_sim


if __name__ == '__main__':
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
    if lang == 'zh':
        G_dataset = DBpDataset('../datasets/dbp_z_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_z_e/' + pair)
    elif lang == 'ja':
        G_dataset = DBpDataset('../datasets/dbp_j_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_j_e/' + pair)
    elif lang == 'fr':
        G_dataset = DBpDataset('../datasets/dbp_f_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_f_e/' + pair)
    elif lang == 'y':
        G_dataset = DBpDataset('../datasets/D_Y/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/D_Y/' + pair)
    elif lang == 'w':
        G_dataset = DBpDataset('../datasets/D_W/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/D_W/' + pair)


    Lvec = None
    Rvec = None
    model_name = 'mean_pooling'
    saved_model = None
    args = None
    in_d = None
    out_d = None
    m_adj=None
    e1=None
    e2=None
    device = 'cuda'
    model = None
    model_name = 'mean_pooling'

    if lang == 'zh':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/zh_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/dbp_z_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_z_e/rel_dict1'))
    elif lang == 'ja':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/ja_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/dbp_j_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_j_e/rel_dict1'))
    elif lang == 'fr':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/fr_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/dbp_f_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_f_e/rel_dict1'))
    elif lang == 'w':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/w_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/D_W/ent_dict1'))
        splitr = len(read_list('../datasets/D_W/rel_dict1'))
    elif lang == 'y':
        model = Encoder('gcn-align', [100,100,100], [1,1,1], activation=F.elu, feat_drop=0, attn_drop=0, negative_slope=0.2, bias=False)
        model.load_state_dict(torch.load('../saved_model/y_model.pt'))
        model_name = 'load'
        split = len(read_list('../datasets/D_Y/ent_dict1'))
        splitr = len(read_list('../datasets/D_Y/rel_dict1'))


    evaluator = None
    explain = EAExplainer(model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr, lang)
    explain.explain_EA(method,0.4, num, version)

