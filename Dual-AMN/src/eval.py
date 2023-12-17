import tarfile
import torch
import logging
import numpy as np
import sys


class Evaluate:
    def __init__(self, test_dict, valid_dict,test_pairs, new_test_pairs, new_ent, valid_pairs, device, eval_batch_size, k, dataset, batch, M):
        self.test_pairs = test_pairs
        self.test_dict = test_dict
        self.valid_dict = valid_dict
        self.new_test_pairs = new_test_pairs
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.topk = k
        self.batch = batch
        self.dataset = dataset
        self.M = M
        self.valid_pairs = valid_pairs
        self.new_ent = new_ent

    def sim_results(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        return A_sim
    
    def sim_res(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        a = torch.norm(Matrix_A, p=2, dim=-1)
        b = torch.norm(Matrix_B, p=2, dim=-1)
        cos_sim = A_sim / a.unsqueeze(-1)
        cos_sim /= b.unsqueeze(-2)
        return A_sim, cos_sim

    def avg_results(self, Matrix_A):
        k = 10
        avg_results = torch.sum(torch.topk(Matrix_A, k=k)[0], dim=-1) / k
        return avg_results

    def csls_sim(self, sim_mat, k):
        """
        Compute pairwise csls similarity based on the input similarity matrix.

        Parameters
        ----------
        sim_mat : matrix-like
            A pairwise similarity matrix.
        k : int
            The number of nearest neighbors.

        Returns
        -------
        csls_sim_mat : A csls similarity matrix of n1*n2.
        """
        nearest_values1 = self.calculate_nearest_k(sim_mat, k)
        nearest_values2 = self.calculate_nearest_k(sim_mat.T, k)
        csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
        return csls_sim_mat

    def calculate_nearest_k(self, sim_mat, k):
        sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
        nearest_k = sorted_mat[:, 0:k]
        return np.mean(nearest_k, axis=1, keepdims=True)

    def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
        assert 1 in top_k
        mr = 0
        mrr = 0
        hits = [0] * len(top_k)
        hits1_rest = set()
        for i in range(len(idx)):
            gold = idx[i]
            if accurate:
                rank = (-sim_mat[i, :]).argsort()
                
            else:
                rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
            hits1_rest.add((gold, rank[0]))
            assert gold in rank
            rank_index = np.where(rank == gold)[0][0]
            # print(rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
        mr /= total_num
        mrr /= total_num
        return mr, mrr, hits, hits1_rest
    def CSLS_results(self, inputs):
        SxT_sim, TxS_avg, SxT_avg = inputs

        TxS_avg, SxT_avg = [torch.unsqueeze(m, dim=1) for m in [TxS_avg, SxT_avg]]
        sim = 2 * SxT_sim - SxT_avg - TxS_avg.t()
        rank = torch.argsort(sim, dim=-1, descending=True)

        targets = rank[:, :self.topk]
        values = torch.gather(sim, 1, targets)

        return targets.cpu().numpy(), values.cpu().numpy()

    def rank(self, sourceVec, targetVec, idx, entity):
        sim_mat = self.sim_results(sourceVec, targetVec)
        mr = 0
        mrr = 0
        top_k = [1,3,10]
        hits = [0] * len(top_k)
        hits1_rest = set()
        for i in range(len(idx)):
            gold = idx[i]
            rank = (-sim_mat[i, :]).argsort()
            # print(rank)
            # print(entity)
            rank = entity[rank]
            hits1_rest.add((gold, rank[0]))
            assert gold in rank
            rank_index = np.where(rank == gold)[0][0]
            # print(rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
        mrr /= sourceVec.shape[0]
        print('hits1 is ', hits[0] / sourceVec.shape[0])
        print('hits3 is ', hits[1] / sourceVec.shape[0])
        print('hits10 is ', hits[2] / sourceVec.shape[0])
        print('mrr is ', mrr)
        return hits[1]

    def test_rank_tar(self, sourceVec, targetVec, gold):
        sim_mat = self.sim_results(sourceVec, targetVec)
        sim_ori, sim_cos = self.sim_res(sourceVec, targetVec)
        # torch.save(sim_cos, '/data/xbtian/amie-master/zh-en/sim_cos.pt')
        # print(sim_cos[0][1])
        # sim_mat = sim_ori
        # sim_ori = sim_mat
        mr = 0
        mrr = 0
        top_k = [1,3,10]
        hits = [0] * len(top_k)
        hits1_rest = set()
        pair = set()
        rank = (-sim_mat[0, :]).argsort()
        print(rank)
        rank_index = np.where(rank == gold)[0][0]
        print('rank is ', rank_index + 1)
        return rank_index 

    def test_rank(self, sourceVec, targetVec):
        sim_mat = self.sim_results(sourceVec, targetVec)
        sim_ori, sim_cos = self.sim_res(sourceVec, targetVec)
        # torch.save(sim_cos, '/data/xbtian/amie-master/zh-en/sim_cos.pt')
        # print(sim_cos[0][1])
        # sim_mat = self.csls_sim(sim_ori, 5)
        # sim_mat = sim_cos
        # sim_ori = sim_mat
        mr = 0
        mrr = 0
        top_k = [1,3,10]
        hits = [0] * len(top_k)
        hits1_rest = set()
        pair = set()
        sim_list = []
        for i in range(sourceVec.shape[0]):
            # print(i)
            gold = i
            rank = (-sim_mat[i, :]).argsort()
            # print(rank)
            # print(entity)
            # rank = entity[rank]
            pair.add((i, rank[0]))
            hits1_rest.add((gold, rank[0]))
            sim_list.append((i, gold, rank[0],sim_mat[i, gold], sim_mat[i, rank[0]] ))
            # print(i, gold, rank[0],sim_mat[i, gold], sim_mat[i, rank[0]])
            assert gold in rank
            rank_index = np.where(rank == gold)[0][0]
            # print(rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
        mrr /= targetVec.shape[0]
        print('hits1 is ', hits[0] / sourceVec.shape[0])
        print('hits3 is ', hits[1] / sourceVec.shape[0])
        print('hits10 is ', hits[2] / sourceVec.shape[0])
        print('mrr is ', mrr)
        return hits[0] / sourceVec.shape[0], pair, sim_list
    def CSLS_cal(self, sourceVec, targetVec):

        batch_size = self.eval_batch_size
        SxT_sim = []
        for epoch in range(len(sourceVec) // batch_size + 1):
            SxT_sim.append(self.sim_results(sourceVec[epoch * batch_size:(epoch + 1) * batch_size], targetVec))

        SxT_avg = []
        for epoch in range(len(sourceVec) // batch_size + 1):
            SxT_avg.append(self.avg_results(SxT_sim[epoch]))

        TxS_sim = self.sim_results(targetVec, sourceVec)

        TxS_avg = self.avg_results(TxS_sim)

        targets = np.empty((0, self.topk), int)
        values = np.empty((0, self.topk), float)
        for epoch in range(len(sourceVec) // batch_size + 1):
            temp_targets, temp_values = self.CSLS_results([SxT_sim[epoch].to(device=self.device), TxS_avg.to(device=self.device), SxT_avg[epoch].to(device=self.device)])
            targets = np.concatenate((targets, temp_targets), axis=0)
            values = np.concatenate((values, temp_values), axis=0)
        return targets, values

    def test_with_threshold(self, sourceVec, targetVec, entity1, entity2):
        # from source predict target
        topk_targets_s2t, topk_values_s2t = self.CSLS_cal(sourceVec, targetVec)
        # from target predict source
        topk_targets_t2s, topk_values_t2s = self.CSLS_cal(targetVec, sourceVec)

        credible_pairs_s2t, credible_pairs_t2s = set(), set()
        pair_value_dic_s2t, pair_value_dic_t2s = dict(), dict()

        entity1 = np.array(entity1)
        entity2 = np.array(entity2)

        for i, p in enumerate(topk_targets_s2t):
            e1 = entity1[i]
            for i2, j in enumerate(p):
                e2 = entity2[j]
                credible_pairs_s2t.add((e1, e2))
                pair_value_dic_s2t[(e1, e2)] = topk_values_s2t[i][i2]
        for i, p in enumerate(topk_targets_t2s):
            e2 = entity2[i]
            for i2, j in enumerate(p):
                e1 = entity1[j]
                credible_pairs_t2s.add((e1, e2))
                pair_value_dic_t2s[(e1, e2)] = topk_values_t2s[i][i2]

        # intersection
        final_credible_pairs = credible_pairs_s2t.intersection(credible_pairs_t2s)

        if self.batch == 'base':
            logging.info(f"total predicted pairs:{len(final_credible_pairs)}")
            with open("../datasets/"+self.dataset+"/"+self.batch+"/predicted_pairs", 'w', encoding='utf-8') as f:
                for p in final_credible_pairs:
                    e1, e2 = p[0], p[1]
                    value = pair_value_dic_s2t[p]
                    f.write(str(e1)+'\t'+str(e2)+'\t'+str(value)+'\n')
        else:
            final_credible_pairs, final_credible_pairs_with_value = self.merge_predicted_pairs(final_credible_pairs, pair_value_dic_s2t)
            logging.info(f"total predicted pairs:{len(final_credible_pairs)}")

        logging.info(f"topk={self.topk}")

        logging.info(f"credible pairs size={len(final_credible_pairs)}, golden test pairs size={len(self.test_pairs)}")
        self.P_R_F1(final_credible_pairs, self.test_pairs)

        new_ent_final_credile_pairs = set()
        for p in final_credible_pairs:
            e1, e2 = p[0], p[1]
            if e1 in self.new_ent or e2 in self.new_ent:
                new_ent_final_credile_pairs.add(p)

        logging.info(f"credible pairs with new entity size={len(new_ent_final_credile_pairs)}, golden new test pairs size={len(self.new_test_pairs)}")

        self.P_R_F1(new_ent_final_credile_pairs, self.new_test_pairs)

        if self.batch == 'base':
            topM_value_s2t = np.sort(topk_values_s2t.reshape((-1)))[-self.M]
            topM_value_t2s = np.sort(topk_values_t2s.reshape((-1)))[-self.M]
            new_train_pairs_s2t = set()
            new_train_pairs_t2s = set()

            loc = np.where(topk_values_s2t >= topM_value_s2t)
            e1 = entity1[loc[0]]
            e2_index = topk_targets_s2t[loc]
            e2 = entity2[e2_index]
            for i in range(len(e1)):
                new_train_pairs_s2t.add((e1[i], e2[i]))

            loc = np.where(topk_values_t2s >= topM_value_t2s)
            e2 = entity2[loc[0]]
            e1_index = topk_targets_t2s[loc]
            e1 = entity1[e1_index]
            for i in range(len(e1)):
                new_train_pairs_t2s.add((e1[i], e2[i]))

            final_new_train_pairs = new_train_pairs_s2t.intersection(new_train_pairs_t2s)

            logging.info(f"new train pairs num={len(final_new_train_pairs)}")
            with open("../datasets/"+self.dataset+"/"+self.batch+"/credible_pairs", 'w', encoding='utf-8') as f:
                for p in final_new_train_pairs:
                    f.write(str(p[0])+'\t'+str(p[1])+'\n')

        else:
            def take_third(elem):
                return elem[2]
            sorted_list = sorted(list(final_credible_pairs_with_value), key=take_third, reverse=True)
            ent_set = set()
            count = 0
            finals = set()
            for p in sorted_list:
                e1, e2, v = p
                if count >= self.M:
                    break
                # if e1 not in ent_set and e2 not in ent_set:
                ent_set.add(e1)
                ent_set.add(e2)
                count += 1
                finals.add((e1, e2))
            logging.info(f"new train pairs num={len(finals)}")
            with open("../datasets/" + self.dataset + "/" + self.batch + "/credible_pairs", 'w', encoding='utf-8') as f:
                for p in finals:
                    f.write(str(p[0]) + '\t' + str(p[1]) + '\n')

    def valid(self, sourceVec, targetVec, entity1, entity2):
        # from source predict target

        return self.rank(sourceVec, targetVec, self.valid_dict, entity2)
         
        topk_targets_s2t, topk_values_s2t = self.CSLS_cal(sourceVec, targetVec)

        credible_pairs = set()
        for id1, i in enumerate(topk_targets_s2t):
            for j in i:
                e1 = entity1[id1]
                e2 = entity2[j]
                credible_pairs.add((e1, e2))

        P, R, F1 = self.P_R_F1(credible_pairs, self.valid_pairs)
        return P, R, F1

    def P_R_F1(self, credible_pairs, golden_pairs):
        hit = 0
        for p in credible_pairs:
            if p in golden_pairs:
                hit += 1

        if hit == 0:
            logging.info(f"hit = 0")
            return

        P = hit/len(credible_pairs)
        R = hit/len(golden_pairs)
        F1 = 2*P*R/(P+R)

        logging.info(f"Precision: {(P):.3f}, Recall: {(R):.3f}, F1: {(F1):.3f}")

        return P, R, F1

    def merge_predicted_pairs(self, final_credible_pairs, pair_value_dict):
        last_predicted_pairs = set()
        last_pair_value_dict = dict()
        last_ent_dic_1to2 = dict()
        last_ent_dic_2to1 = dict()
        last_ent_set = set()
        last_batch = int(self.batch[-1])-1
        if last_batch == 0:
            file = "../datasets/"+self.dataset+"/"+"base"+"/predicted_pairs"
        else:
            file = "../datasets/" + self.dataset + "/" + "batch"+str(last_batch) + "/predicted_pairs"
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                l = line.strip().split('\t')
                e1, e2, v = int(l[0]), int(l[1]), float(l[2])
                last_predicted_pairs.add((e1, e2))
                last_pair_value_dict[(e1, e2)] = v
                last_ent_dic_1to2[e1] = e2
                last_ent_dic_2to1[e2] = e1
                last_ent_set.add(e1)
                last_ent_set.add(e2)

        new_ent = set()
        ent_dic_1to2 = dict()
        ent_dic_2to1 = dict()
        for p in final_credible_pairs:
            e1, e2 = p
            new_ent.add(e1)
            new_ent.add(e2)
            ent_dic_1to2[e1] = e2
            ent_dic_2to1[e2] = e1

        count=0
        new_final_credible_pairs = set()
        for p in final_credible_pairs:
            e1, e2 = p[0], p[1]
            cur_v = pair_value_dict[p]
            if e1 in self.new_ent or e2 in self.new_ent:
                count += 1
                new_final_credible_pairs.add((e1, e2, cur_v))
                continue
            if p in last_predicted_pairs:
                new_final_credible_pairs.add((e1, e2, last_pair_value_dict[p]))
                continue
            if e1 in last_ent_set:
                last_v = last_pair_value_dict[(e1, last_ent_dic_1to2[e1])]
                if last_v >= cur_v:
                    new_final_credible_pairs.add((e1, last_ent_dic_1to2[e1], last_v))
                else:
                    new_final_credible_pairs.add((e1, e2, cur_v))
            if e2 in last_ent_set:
                last_v = last_pair_value_dict[(last_ent_dic_2to1[e2], e2)]
                if last_v >= cur_v:
                    new_final_credible_pairs.add((last_ent_dic_2to1[e2], e2, last_v))
                else:
                    new_final_credible_pairs.add((e1, e2, cur_v))

        for p in last_predicted_pairs:
            e1, e2 = p[0], p[1]
            last_v = last_pair_value_dict[p]
            if e1 not in new_ent and e2 not in new_ent:
                new_final_credible_pairs.add((e1, e2, last_v))

        new_final_credible_pairs_no_value = set()
        with open("../datasets/"+self.dataset+"/"+self.batch+"/predicted_pairs", 'w', encoding='utf-8') as f:
            for item in new_final_credible_pairs:
                e1, e2, v = item[0], item[1], item[2]
                f.write(str(e1)+'\t'+str(e2)+'\t'+str(v)+'\n')
                new_final_credible_pairs_no_value.add((e1, e2))

        return new_final_credible_pairs_no_value, new_final_credible_pairs
