import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter_sum
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
class GCN1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x

class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x



class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


    

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
        
    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x

class NR_GraphAttention(nn.Module):
    def __init__(self,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False,
                 in_d = None,
                 out_d = None):
        super(NR_GraphAttention, self).__init__()

        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        # create parameters
        feature = self.node_dim*(self.depth+1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        # proxy node
        self.proxy = torch.nn.Parameter(data=torch.empty(64, feature, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.proxy)

        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        triple_size = inputs[5]
        rel_size = inputs[6]
        node_size = inputs[7]
        mask = inputs[8]
        # print('init------------')
        # print(features[6017])
        # print(features[5371])
        features = self.activation(features)
        # print(features.shape)
        outputs.append(features)
        
        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            # matrix shape: [N_tri x N_rel]
            

            # shape: [N_tri x dim]

            neighs = features[adj[1, :].long()]
            # print(features[6017])
            # print(features[5371])
            # print(adj[1,:])
            # print(neighs.shape)
            # print(features.shape)
            # print(adj.shape)
            # print(self.triple_size)
            # print(triple_size)
            # print(r_val.shape)
            if mask == None:
                tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                            size=[triple_size, rel_size], dtype=torch.float32)
            else:
                tmp = torch.zeros(adj.shape[1]).cuda()
                cur = (mask * r_val).float()
                # print(cur)
                tmp = tmp.scatter_add_(0, r_index[0].long(), cur)
                tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                            size=[triple_size, rel_size], dtype=torch.float32)
        # shape: [N_tri x dim]
            # print(triple_size)
            # print(tri_rel.shape)
            # print(r_index.shape)
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            
            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            # print(neighs.shape)
            # print(neighs)
            # print(neighs[2])
            # print(neighs[5])
            if mask != None:
                # print(tmp.shape)
                # print(torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel)
               # neighs = neighs - 2*tmp.unsqueeze(1) * (torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel)
               neighs = neighs - 2* (torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel)
            else:
                neighs = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel
            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            # print('neighs-------------')
            # print(neighs.shape)
            # print(torch.where(neighs != 0))
            # print(neighs[2])
            # print(neighs[5])
            # att = att * mask
            
            # print(att)
            # print(mask)
            # print(att)
            if mask != None:
                att[tmp == 0] = -1e10
             
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[node_size, node_size])
            att = torch.sparse.softmax(att, dim=1)
            if mask != None:
                
                mask_adj = torch.sparse_coo_tensor(indices=adj, values= tmp, size=[node_size, node_size])
                # print('--------mask-----------')
                # print(mask_adj)
                # print(tmp)
                att = att * mask_adj 
                # print('-------attn----------')
                # print(att)
            # print(att.shape)
            # print(adj[0,:].shape)
            # print(adj[0,:].long().max())
            # print(att.coalesce().values())
            # print(torch.where(neighs != 0))
            # print(neighs.shape)
            # print(att.shape)
            # print(torch.unsqueeze(att.coalesce().values(), dim=-1))
            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0,:].long())
            # print(torch.where(new_features != 0))
            # print(new_features[6017])
            # print(new_features[5371])
            pad = torch.zeros([att.shape[0] - new_features.shape[0], neighs.shape[1]]).to('cuda')
            # print(pad.shape)
            new_features = torch.cat((new_features, pad), 0)
            # print(new_features.shape)
            # new_features = scatter_sum(src=neighs, dim=0,
                                       # index=adj[0,:].long())
            features = self.activation(new_features)
            # print(features.shape)
            outputs.append(features)
        # print(outputs)
        outputs = torch.cat(outputs, dim=-1)
        
        proxy_att = torch.mm(F.normalize(outputs, p=2, dim=-1), torch.transpose(F.normalize(self.proxy, p=2, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)
        proxy_feature = outputs - torch.mm(proxy_att, self.proxy)

        gate_rate = torch.sigmoid(self.gate(proxy_feature))

        final_outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        
        return final_outputs
        
        # return outputs

