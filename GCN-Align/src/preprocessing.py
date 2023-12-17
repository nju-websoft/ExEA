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

        self.kgs = set()
        if lang == 'zh':
            self.ent_dict, self.id_ent = self.read_dict('../datasets/dbp_z_e/ent_dict')
            self.r_dict, self.id_r = self.read_dict('../datasets/dbp_z_e/rel_dict')
        elif lang == 'ja':
            self.ent_dict, self.id_ent = self.read_dict('../datasets/dbp_j_e/ent_dict')
            self.r_dict, self.id_r = self.read_dict('../datasets/dbp_j_e/rel_dict')
        elif lang == 'fr':
            self.ent_dict, self.id_ent = self.read_dict('../datasets/dbp_f_e/ent_dict')
            self.r_dict, self.id_r = self.read_dict('../datasets/dbp_f_e/rel_dict')
        elif lang == 'y':
            self.ent_dict, self.id_ent = self.read_dict('../datasets/D_Y/ent_dict')
            self.r_dict, self.id_r = self.read_dict('../datasets/D_Y/rel_dict')
        elif lang == 'w':
            self.ent_dict, self.id_ent = self.read_dict('../datasets/D_W/ent_dict')
            self.r_dict, self.id_r = self.read_dict('../datasets/D_W/rel_dict')
        # self.target_link1, self.target_link2 = read_link(file_path + '/sample_pair_v1')
        
        self.target_link1, self.target_link2 = read_link(file_path + pair)
        self.gid1 = defaultdict(list)
        self.gid2 = defaultdict(list)
        self.gid = defaultdict(list)
        self.triple_size = len(self.kg1) + len(self.kg2)
        # self.rel_fact = self.read_rel_fact(file_path)
        # self.rel_fact_pair = self.read_rel_fact_pair(file_path)
        self.test_pair = self.load_alignment_pair(file_path + '/test')
        self.test_pair = np.array(self.test_pair)
        self.train_pair = self.load_alignment_pair(file_path + '/train_links')
        self.train_pair = np.array(self.train_pair)
        
        # self.model_link, _ = read_link('hard_pair')
        self.train_link, _ = read_link(file_path + '/train_links')
        self.gid = defaultdict(list)
        self.triple_size = len(self.kg1) + len(self.kg2)
        # self.rel_fact = self.read_rel_fact(file_path)
        # self.rel_fact_pair = self.read_rel_fact_pair(file_path)
        self.test_pair = self.load_alignment_pair(file_path + '/test')
        self.test_pair = np.array(self.test_pair)
        # self.model_pair = self.load_alignment_pair(file_path + '/ori_pair.txt')
        self.model_link, _ = read_link(file_path + '/pair.txt')
        self.model_pair = self.load_alignment_pair(file_path + '/pair.txt')
        # self.model_link, _ = read_link('one2one_pair_zh')
        # self.model_pair = self.load_alignment_pair('one2one_pair_zh')
        self.conflict_r_pair = set(self.load_alignment_pair(file_path + '/triangle_id'))
        if os.path.exists(file_path + '/triangle_id_2'):
            self.conflict_id = self.read_line_rel(file_path + '/triangle_id_2')
        else:
            self.conflict_id = None
        # self.model_pair = self.load_alignment_pair('hard_pair')
        self.model_pair = np.array(self.model_pair)
        self.test_link, _ = read_link(file_path + '/test')
        self.train_pair = np.array(self.train_pair)
        self.suff_kgs = set()
        self.entity1= set()
        self.entity2 = set()
        # print(self.rel_fact)
        self.entity = set()
        self.rel = set()
        self.device = device
        # self.rfunc = self.read_r_func(file_path + '/rfunc')
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
        '''
        with open(file_path + '/triples_name', 'w') as f:
            for cur in self.gid:
                f.write(self.ent_dict[cur] + '\n')
                f.write('--------------------\n')
                for tri in self.gid[cur]:
                    f.write(self.ent_dict[tri[0]] + '\t' + self.r_dict[tri[1]] + '\t' + self.ent_dict[tri[2]] + '\n')
        exit(0)
        '''
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