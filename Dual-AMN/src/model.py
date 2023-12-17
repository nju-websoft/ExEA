import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention
from tabulate import tabulate
import logging
from torch_scatter import scatter_mean
import numpy as np
from torch_scatter import scatter_add
import torch_geometric.utils as utils

def getPositionEncoding(seq_len,dim,n=10000):
  PE = np.zeros(shape=(seq_len,dim))
  for pos in range(seq_len):
    for i in range(int(dim/2)):
      denominator = np.power(n, 2*i/dim)
      PE[pos,2*i] = np.sin(pos/denominator)
      PE[pos,2*i+1] = np.cos(pos/denominator)

  return PE

class ALL_entroy(nn.Module):
    def __init__(self, device):
        super(ALL_entroy, self).__init__()
        self.device = device
    def forward_one(self,train_set, x, e2):
        x1_train, x2_train = x[train_set[:, 0]], x[train_set[:, 1]]
        label = torch.arange(0, x1_train.shape[0]).to(self.device)
        d = {}
        for i in range(e2.shape[0]):
            d[int(e2[i])] = i 
        x2 = x[e2]
        # print(x1_train.shape[0])
        pred = torch.matmul(x1_train, x2.transpose(0 , 1))
        self.bias_0 = torch.nn.Parameter(torch.zeros(x2.shape[0])).to(self.device)
        pred += self.bias_0.expand_as(pred)
        for i in range(x1_train.shape[0]):
            label[i] = d[int(train_set[i, 1])]
        # label = train_set[:, 1].unsqueeze(1)
        label = label.unsqueeze(1)
        # print(label.shape)
        # print(label)
        # exit(0)
        # print( torch.zeros(x1_train.shape[0], x2.shape[0]).shape)
        soft_targets = torch.zeros(x1_train.shape[0], x2.shape[0]). \
                    to(self.device).scatter_(1, label, 1)
        # print(soft_targets[2][train_set[2, 1]])
        soft = 0.8
        soft_targets = soft_targets * soft \
                            + (1.0 - soft_targets) \
                            * ((1.0 - soft) / (x2.shape[0] - 1))
        # print(soft_targets[2][train_set[2, 1]])
        logsoftmax = nn.LogSoftmax(dim=1)
        # exit(0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    
    def forward(self, train_set, x, e2):
        loss_l = self.forward_one(train_set, x, e2)
        # loss_r = self.forward_one(train_set[[1, 0]], x)
        return loss_l 
'''
class ALL_entroy(nn.Module):
    def __init__(self, device):
        super(ALL_entroy, self).__init__()
        self.device = device
    def forward_one(self,emb, train_set, x):
        x1_train, x2_train = emb[train_set[:, 0]], emb[train_set[:, 1]]
        print(x1_train.shape[0])
        pred = torch.matmul(x1_train, x.transpose(0 , 1))
        self.bias_0 = torch.nn.Parameter(torch.zeros(x.shape[0])).to(self.device)
        pred += self.bias_0.expand_as(pred)
        label = train_set[:, 0].unsqueeze(1)
        soft_targets = torch.zeros(x1_train.shape[0], x.shape[0]). \
                    to(self.device).scatter_(1, label, 1)
        soft = 0.8
        soft_targets = soft_targets * soft \
                            + (1.0 - soft_targets) \
                            * ((1.0 - soft) / (x.shape[0] - 1))
        # print(soft_targets[2][train_set[2, 1]])
        logsoftmax = nn.LogSoftmax(dim=1)
        # exit(0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    
    def forward(self, x1_train, train_set, x, e2):
        loss_l = self.forward_one(x1_train, train_set, x, e2)
        loss_r = self.forward_one(train_set[[1, 0]], x)
        return loss_l + loss_r
'''
class LapEncoding:
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, edge_index, num_nodes):
        edge_index, edge_attr = utils.get_laplacian(
            edge_index.long(),  normalization=self.normalization,
            num_nodes=num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()
class RWEncoding:
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, edge_index, num_nodes):
        W0 = normalize_adj(edge_index, num_nodes=num_nodes).tocsc()
        W = W0
        vector = torch.zeros((num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden, device,node_size,rel_size,
                 alpha, beta, new_ent_nei,
                 dropout_rate=0.0, ind_dropout_rate=0.0, gamma=3, lr=0.005, depth=2, in_d=None, out_d=None, m_adj=None, e1=None, e2=None):
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.ind_dropout = nn.Dropout(ind_dropout_rate)
        self.gamma = gamma
        self.loss = ALL_entroy(self.device)

        self.lr = lr
        self.ind_loss = nn.MSELoss(reduction='sum')
        self.alpha = alpha
        
        self.beta = beta
        self.new_ent_nei = torch.from_numpy(new_ent_nei).long().to(device)
        self.m_adj = None
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        # print(self.ent_embedding.weight.shape)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        # lap = LapEncoding(node_hidden)
        # res = lap.compute_pe(self.adj_list, node_size)
        # print(res.shape)
        # print(res)
        # exit(0)
        # self.ent_embedding.weight.data += in_pos[in_d.long()] + out_pos[out_d.long()]
        self.e_encoder = NR_GraphAttention(
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True,

                                           )
        self.r_encoder = NR_GraphAttention(
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True,
                                           )
    def get_embeddings(self, index_a, index_b, ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, mask):
        # forward
        out_feature = self.gcn_forward(ent_adj,  rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, mask)
        # out_feature = self.ent_embedding.weight.data
        out_feature = out_feature.cpu()
        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

        return Lvec, Rvec

    def print_all_model_parameters(self):
        logging.info('\n------------Model Parameters--------------')
        info = []
        head = ["Name", "Element Nums", "Element Bytes", "Total Size (MiB)", "requires_grad"]
        total_size = 0
        total_element_nums = 0
        for name, param in self.named_parameters():
            info.append((name,
                         param.nelement(),
                         param.element_size(),
                         round((param.element_size()*param.nelement())/2**20, 3),
                         param.requires_grad)
                        )
            total_size += (param.element_size()*param.nelement())/2**20
            total_element_nums += param.nelement()
        logging.info(tabulate(info, headers=head, tablefmt="grid"))
        logging.info(f'Total # parameters = {total_element_nums}')
        logging.info(f'Total # size = {round(total_size, 3)} (MiB)')
        logging.info('--------------------------------------------')
        logging.info('')
    def avg(self, adj, emb, size: int, node_size, mask, r_index, adj_list):
        # print(adj.shape)
        if mask != None:
            tmp = torch.zeros(adj_list.shape[1]).cuda()
            # cur = (mask * r_val).float()
            # print(cur)
            # print(mask)
            tmp = tmp.scatter_add_(0, r_index[0].long(), mask)
            # print(torch.ones_like(adj[0, :], dtype=torch.float) * tmp)
            # exit(0)
            # tmp[tmp == 0] = -1e10
            # print(adj_list.T[tmp == 0])
            indices = (adj_list.T[tmp == 0]).T
            tmp = torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(indices[0, :], dtype=torch.float) * -1e10,
                                      size=[node_size, size])
            adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])
            # adj += tmp
            adj = torch.sparse.softmax(adj, dim=1)
            # exit(0)
        else:
            adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

            adj = torch.sparse.softmax(adj, dim=1)
        # print('adj is ',adj)
        # print(adj.shape)
        # print(emb.shape)
        return torch.sparse.mm(adj, emb)
    
    def avg_r(self, adj, emb, size: int, node_size, mask, r_index, adj_list):
        # print(adj.shape)
        if mask != None:
            # tmp = torch.zeros(adj_list.shape[1]).cuda()
            # cur = (mask * r_val).float()
            # print(cur)
            # print(mask)
            # tmp = tmp.scatter_add_(0, r_index[0].long(), mask)
            # print(torch.ones_like(adj[0, :], dtype=torch.float) * tmp)
            # exit(0)
            # tmp[tmp == 0] = -1e10
            # print(adj_list.T[tmp == 0])
            # indices = (adj_list.T[tmp == 0]).T
            # tmp = torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(indices[0, :], dtype=torch.float) * -1e10,
                                      # size=[node_size, size])
            adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])
            # adj *= tmp
            # exit(0)
        else:
            adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        # print('adj is ',adj)
        # print(adj.shape)
        # print(emb.shape)
        return torch.sparse.mm(adj, emb)

    def gcn_forward(self, ent_adj, rel_adj,node_size, rel_size, adj_list, r_index, r_val, triple_size, mask):
        # [Ne x Ne] · [Ne x dim] = [Ne x dim]
        ent_feature = self.avg(ent_adj, self.ent_embedding.weight, node_size, node_size, mask, r_index, adj_list)
        # [Ne x Nr] · [Nr x dim] = [Ne x dim]
        rel_feature = self.avg_r(rel_adj, self.rel_embedding.weight, rel_size, node_size, mask, r_index, adj_list)
        opt = [self.rel_embedding.weight, adj_list, r_index, r_val, triple_size, rel_size ,node_size, mask]
        out_feature = torch.cat([self.e_encoder([ent_feature] + opt), self.r_encoder([rel_feature] + opt)], dim=-1)
        out_feature = self.dropout(out_feature)

        return out_feature

    

    def forward(self, train_paris:torch.Tensor, ent_adj, rel_adj,node_size, rel_size, adj_list, r_index, r_val, triple_size, mask):
        out_feature = self.gcn_forward(ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, mask)
        loss1 = self.align_loss(train_paris, out_feature, node_size)
        return loss1
    
    def align_loss(self, pairs, emb, node_size):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=node_size) - F.one_hot(r, num_classes=node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=node_size) - F.one_hot(r, num_classes=node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)


