import numpy as np
import os
import torch
from count import read_tri, read_link, read_list,read_tri_list
from collections import defaultdict
import scipy.sparse as sp
import networkx as nx
class DBpDataset:
    def __init__(self, file_path, device, pair, lang='zh'):
        self.kg1 = read_tri_list(file_path + '/triples_1')
        self.kg2 = read_tri_list(file_path + '/triples_2')
        '''
        self.two_hop1, self.one_hop1 = self.get_2_all_hop(self.kg1)
        self.two_hop2, self.one_hop2 = self.get_2_all_hop(self.kg2)
        self.all_2_hop1 = defaultdict(set)
        self.all_2_hop2 = defaultdict(set)
        for cur in self.one_hop1:
            self.all_2_hop1[cur] = self.one_hop1[cur] | self.two_hop1[cur]
        for cur in self.one_hop2:
            self.all_2_hop2[cur] = self.one_hop2[cur] | self.two_hop2[cur]
        '''
        self.kgs = set()
        self.ent_dict, self.id_ent = self.read_dict(file_path +  '/ent_dict')
        self.r_dict, self.id_r = self.read_dict(file_path +  '/rel_dict')
        # self.target_link1, self.target_link2 = read_link(file_path + '/sample_pair_v1')
        self.target_link1, self.target_link2 = read_link(file_path + pair)
        self.gid1 = defaultdict(list)
        self.gid2 = defaultdict(list)
        # self.model_link, _ = read_link(file_path + '/ori_pair.txt')
        self.model_link, _ = read_link(file_path + '/pair.txt')
        # self.model_link, _ = read_link('hard_pair')
        self.gid = defaultdict(list)
        self.triple_size = len(self.kg1) + len(self.kg2)
        # self.rel_fact = self.read_rel_fact(file_path)
        # self.rel_fact_pair = self.read_rel_fact_pair(file_path)
        self.test_pair = self.load_alignment_pair(file_path + '/test')
        self.test_pair = np.array(self.test_pair)
        # self.model_pair = self.load_alignment_pair(file_path + '/ori_pair.txt')
        self.model_pair = self.load_alignment_pair(file_path + '/pair.txt')
        # self.model_pair = self.load_alignment_pair('hard_pair')
        # print(len(set(self.load_alignment_pair('cur_pair')) & set(self.load_alignment_pair(file_path + '/test'))) / 10500)
        self.model_pair = np.array(self.model_pair)
        self.train_pair = self.load_alignment_pair(file_path + '/train_links')
        self.conflict_r_pair = set(self.load_alignment_pair(file_path + '/triangle_id'))
        if os.path.exists(file_path + '/triangle_id_2'):
            self.conflict_id = self.read_line_rel(file_path + '/triangle_id_2')
        else:
            self.conflict_id = None
        self.train_link, _ = read_link(file_path + '/train_links')
        self.test_link, _ = read_link(file_path + '/test')
        self.train_pair = np.array(self.train_pair)
        self.suff_kgs = set()
        self.entity1= set()
        self.tri = defaultdict(set)
        self.entity2 = set()
        # print(self.rel_fact)
        self.entity = set()
        self.rel = set()
        self.device = device
        self.rfunc = self.read_r_func(file_path + '/rfunc')
        self.r_ent_set = defaultdict(set)
        self.r_o_set = defaultdict(set)
        for (h, r, t) in self.kg1:
            self.entity.add(int(h))
            self.entity.add(int(t))
            self.entity1.add(int(h))
            self.entity1.add(int(t))
            self.rel.add(int(r))
            # self.r_ent_set[int(r)].add((int(h),int(t)))
            # self.r_o_set[int(r)].add(int(t))
            self.gid[int(h)].append([int(h), int(r), int(t)])
            self.gid[int(t)].append([int(h), int(r), int(t)])
            self.tri[int(h)].add((int(h), int(r), int(t)))
            self.tri[int(t)].add((int(h), int(r), int(t)))
            if h in self.target_link1:
                
                self.gid1[int(h)].append([int(h), int(r), int(t)])
                self.suff_kgs.add((int(h), int(r), int(t)))
            if t in self.target_link1:
                
                self.suff_kgs.add((int(h), int(r), int(t)))
                self.gid1[int(t)].append([int(h), int(r), int(t)])
            self.kgs.add((int(h), int(r), int(t)))
        for (h, r, t) in self.kg2:
            self.entity.add(int(h))
            self.entity.add(int(t))
            self.entity2.add(int(h))
            self.entity2.add(int(t))
            self.rel.add(int(r))
            # self.r_ent_set[int(r)].add((int(h),int(t)))
            # self.r_o_set[int(r)].add(int(t))
            self.tri[int(h)].add((int(h), int(r), int(t)))
            self.tri[int(t)].add((int(h), int(r), int(t)))
            if h in self.target_link2:
                
                self.gid2[int(h)].append([int(h), int(r), int(t)])
                self.suff_kgs.add((int(h), int(r), int(t)))
                # self.gid[int(h)].append([int(h), int(r), int(t)])
            if t in self.target_link2:
                
                self.gid2[int(t)].append([int(h), int(r), int(t)])
                self.suff_kgs.add((int(h), int(r), int(t)))
                # self.gid[int(t)].append([int(h), int(r), int(t)])
            self.gid[int(h)].append([int(h), int(r), int(t)])
            self.gid[int(t)].append([int(h), int(r), int(t)])
            self.kgs.add((int(h), int(r), int(t)))
        self.pattern = defaultdict(set) 
        for e in self.target_link1:
            self.pattern[int(e)] |= self.tri[int(e)]
            for (h, r, t) in self.tri[int(e)]:
                if h != int(e):
                    for t2 in self.tri[int(h)]:
                        if t2[1] != int(r) and t2[2] != int(e):
                            self.pattern[int(e)].add(((h, r, t), t2))
                elif t != int(e):
                    for t2 in self.tri[int(t)]:
                        if t2[0] != int(e) and t2[1] != int(r):
                            self.pattern[int(e)].add(((h, r, t), t2))
        for e in self.target_link2:
            self.pattern[int(e)] |= self.tri[int(e)]
            for (h, r, t) in self.tri[int(e)]:
                if h != int(e):
                    for t2 in self.tri[int(h)]:
                        if t2[1] != int(r) and t2[2] != int(e):
                            self.pattern[int(e)].add(((h, r, t), t2))
                elif t != int(e):
                    for t2 in self.tri[int(t)]:
                        if t2[0] != int(e) and t2[1] != int(r):
                            self.pattern[int(e)].add(((h, r, t), t2))
        

        self.G = nx.Graph()
        # for r in self.r_ent_set:
            # self.rfunc[r] = len(self.r_o_set[r]) / len(self.r_ent_set[r])
        # self.save_dict(self.rfunc, file_path + '/rfunc')
        tri = self.kg1 + self.kg2
        edge_list = []
        for cur in tri:
            edge_list.append((int(cur[0]), int(cur[2]), 1))
        self.G.add_weighted_edges_from(edge_list)

    def read_r_func(self, file):
        d1 = {}
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                d1[self.id_r[cur[0]]] = float(cur[1])
        return d1

    def read_dict(self,file):
        d1 = {}
        d2 = {}
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                d1[int(cur[0])] = cur[1]
                d2[cur[1]] = int(cur[0])
        return d1, d2

    def save_dict(self, d, file):
        with open(file, 'w') as f:
            for cur in d:
                f.write(str(self.r_dict[cur]) + '\t' + str(d[cur]) + '\n')

    def read_triple_name(self, triple):
        print(self.ent_dict[triple[0]], self.r_dict[triple[1]],self.ent_dict[triple[2]])

    def read_line_rel(self, file):
        line_id = set()
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                line_id.add(((int(cur[0]), int(cur[1])), (int(cur[2]), int(cur[3]))))
        return line_id
        
    def get_2_all_hop(self,tri):
        one_hop = defaultdict(set)
        two_hop = defaultdict(set)

        for cur in tri:
            e1 = int(cur[0])
            e2 = int(cur[2])
            one_hop[e1].add(e2)
            one_hop[e2].add(e1)
        for cur in one_hop:
            for neigh in one_hop[cur]:
                if neigh not in one_hop:
                    continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
        
        return two_hop, one_hop
    def read_rel_fact(self, path):
        rel_fact = defaultdict(float)
        with open(path + '/tri_fact_id_v1') as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                rel_fact[(int(cur[0]), int(cur[1]), int(cur[2]))] = float(cur[3])
        return rel_fact
    
    def read_rel_fact_pair(self, path):
        rel_fact = defaultdict(float)
        with open(path + '/rel_fact_pair_id') as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                rel_fact[((int(cur[0]), int(cur[1]), int(cur[2])),(int(cur[3]), int(cur[4]), int(cur[5])) )] = float(cur[6])
        return rel_fact

    def load_alignment_pair(self, file_name):
        alignment_pair = []
        for line in open(file_name, 'r'):
            e1, e2 = line.split()
            alignment_pair.append((int(e1), int(e2)))
        return alignment_pair

    def reconstruct(self, gid1, gid2, new = False):
        entity = set()
        if new == False:
            old_triples = self.gid1[gid1] + self.gid2[gid2]

        rel = {0}  # self-loop edge
        triples = []
        for head, r, tail in old_triples:
            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)  # here all relation add 1
            triples.append([head, r + 1, tail])
        adj_matrix, r_index, r_val, adj_features, rel_features= self.get_matrix(triples,
                                                                                entity,
                                                                                rel)
        m_adj = None
        # print(rel_features)
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        triple_size=ent_adj.shape[0]
        ent_adj = torch.from_numpy(np.transpose(ent_adj))
        ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
        ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
        r_index = torch.from_numpy(np.transpose(r_index))
        r_val = torch.from_numpy(r_val)
        node_size=adj_features.shape[0]
        rel_size=rel_features.shape[1]
        # print(r_index)
        fact_confi = torch.zeros(r_index.shape[1])
        tri_list = []
        tri1 = []
        tri2 = []
        # print(ent_adj)
        # print(r_index)
        d_reverse = [0] * r_index.shape[1]
        tri_dict = {}
        for i in range(r_index.shape[1]):
            # print((int(ent_adj[0][i]),int(r_index[1][i]), int(ent_adj[1][i])))
            index = r_index[0][i]
            if int(ent_adj[0][index]) in self.entity1:
                tri1.append(i)
            else:
                tri2.append(i)
            
            if r_index[1][i] > max(self.rel):
                fact_confi[i] = self.rel_fact[(int(ent_adj[1][index]),int(r_index[1][i]) - max(self.rel) - 3, int(ent_adj[0][index]))]
                t = (int(ent_adj[1][index]),int(r_index[1][i]) - max(self.rel) - 3, int(ent_adj[0][index]))
                tri_list.append(t)
                if t in tri_dict:
                    d_reverse[i] = tri_dict[t]
                    d_reverse[tri_dict[t]] = i
                else:
                    tri_dict[t] = i
                # print((int(ent_adj[1][index]),int(r_index[1][i]) - max(self.rel) - 3, int(ent_adj[0][index])))

            else:
                fact_confi[i] = self.rel_fact[(int(ent_adj[0][index]),int(r_index[1][i]) - 1, int(ent_adj[1][index]))]
                t = (int(ent_adj[0][index]),int(r_index[1][i]) - 1, int(ent_adj[1][index]))
                tri_list.append(t)
                if t in tri_dict:
                    d_reverse[i] = tri_dict[t]
                    d_reverse[tri_dict[t]] = i
                else:
                    tri_dict[t] = i
                # print((int(ent_adj[0][index]),int(r_index[1][i]) - 1, int(ent_adj[1][index])))
        # print(tri_list)
        # for cur in tri_list:
            # print(self.ent_dict[str(cur[0])], self.r_dict[str(cur[1])],self.ent_dict[str(cur[2])])
        adj_list=ent_adj.to(self.device)
        r_index=r_index.to(self.device)
        r_val=r_val.to(self.device)
        rel_matrix=ent_rel_adj.to(self.device)
        ent_matrix=ent_adj_with_loop.to(self.device)
        rel_adj = rel_matrix.to(self.device)
        # print(rel_adj.shape)
        ent_adj = ent_matrix.to(self.device)
        # print(triple_size)
        # print(r_val)
        return [int(gid1)], [int(gid2)], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size, fact_confi, tri_list, tri1, tri2, d_reverse


    def reconstruct_search(self, gid1, gid2, tri1, tri2, new=False, entity_num=0, rel_num=0):
        entity = set()
        old_triples = tri1 + tri2
       
        rel = {0}  # self-loop edge
        triples = []
        for head, r, tail in old_triples:
            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)  # here all relation add 1
            triples.append([head, r + 1, tail])
        # print(triples)
        # entity_num = len(entity)
        # rel_num = len(rel)
        adj_matrix, r_index, r_val, adj_features, rel_features= self.get_matrix(triples,
                                                                                entity,
                                                                                rel, new, entity_num, rel_num)
        m_adj = None
        # print(rel_features)
        # print(adj_features, rel_features)
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        triple_size=ent_adj.shape[0]
        ent_adj = torch.from_numpy(np.transpose(ent_adj))
        ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
        ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
        r_index = torch.from_numpy(np.transpose(r_index))
        r_val = torch.from_numpy(r_val)
        node_size=adj_features.shape[0]
        rel_size=rel_features.shape[1]
        # fact_confi = torch.zeros(r_index.shape[1])


        adj_list=ent_adj.to(self.device)
        r_index=r_index.to(self.device)
        r_val=r_val.to(self.device)
        rel_matrix=ent_rel_adj.to(self.device)
        ent_matrix=ent_adj_with_loop.to(self.device)
        rel_adj = rel_matrix.to(self.device)
        # print(rel_adj.shape)
        # print(rel_adj)
        ent_adj = ent_matrix.to(self.device)
        # print(triple_size)
        return  ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size

    
    def reconstruct_test(self, kgs):
        entity = set()
        rel = {0}  # self-loop edge
        triples = []
        for head, r, tail in kgs:
            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)  # here all relation add 1
            triples.append([head, r + 1, tail])
        
        adj_matrix, r_index, r_val, adj_features, rel_features= self.get_matrix(triples,
                                                                                entity,
                                                                                rel)
        m_adj = None
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        triple_size=ent_adj.shape[0]
        ent_adj = torch.from_numpy(np.transpose(ent_adj))
        ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
        ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
        r_index = torch.from_numpy(np.transpose(r_index))
        r_val = torch.from_numpy(r_val)
        node_size=adj_features.shape[0]
        rel_size=rel_features.shape[1]
        adj_list=ent_adj.to(self.device)
        r_index=r_index.to(self.device)
        r_val=r_val.to(self.device)
        rel_matrix=ent_rel_adj.to(self.device)
        ent_matrix=ent_adj_with_loop.to(self.device)
        rel_adj = rel_matrix.to(self.device)
        # print(rel_adj.shape)
        ent_adj = ent_matrix.to(self.device)
        # print(triple_size)
        return ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size


    def get_matrix(self, triples, entity, rel, new=False, entity_num=0, rel_num=0):
        if new == False:
            ent_size = max(self.entity) + 1
            rel_size = (max(self.rel) + 2)
        else:
            ent_size = entity_num
            rel_size = rel_num + 1
        # print(rel_size)
        # row-based list of lists sparse matrix
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))
        
        # add self-loop
        for i in range(ent_size):
            adj_features[i, i] = 1
        # print(triples)
        for h, r, t in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1
            adj_features[t, h] = 1
            radj.append([h, t, r])
            radj.append([t, h, r + rel_size])
        for h, r, t in triples:

            rel_out[h][r] += 1
            rel_in[t][r] += 1
        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
            if ' '.join([str(h), str(t)]) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h), str(t)]))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        # print(rel_features.shape)
        # print(rel_features)
        # print(rel_in, rel_out, rel_features)
        rel_features = sp.lil_matrix(rel_features)
        # print(in_d)
        return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features
    
    def construct_masked(self, triples1, triples2):
        entity = set()
        old_triples = torch.cat((triples1, triples2), dim=0)
        rel = {0}  # self-loop edge
        triples = []
        for head, r, tail in old_triples:
            entity.add(int(head))
            entity.add(int(tail))
            rel.add(int(r))  # here all relation add 1
            triples.append([int(head), int(r), int(tail)])
        
        adj_matrix, r_index, r_val, adj_features, rel_features= self.get_matrix(triples,
                                                                                entity,
                                                                                rel)

        m_adj = None
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)

        triple_size=ent_adj.shape[0]
        ent_adj = torch.from_numpy(np.transpose(ent_adj))
        
        ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
        ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
        r_index = torch.from_numpy(np.transpose(r_index))
        r_val = torch.from_numpy(r_val)
        node_size=adj_features.shape[0]
        rel_size=rel_features.shape[1]
        
        adj_list=ent_adj.to(self.device)
        r_index=r_index.to(self.device)
        r_val=r_val.to(self.device)
        rel_matrix=ent_rel_adj.to(self.device)
        ent_matrix=ent_adj_with_loop.to(self.device)
        rel_adj = rel_matrix.to(self.device)
        # print(rel_adj.shape)
        ent_adj = ent_matrix.to(self.device)
        # print(triple_size)
        return  ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size
        



        