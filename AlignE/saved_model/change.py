import numpy as np
import torch
def read_dict(file):
    d1 = {}
    d2 = {}
    with open(file) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            d1[cur[0]] = cur[1]
            d2[cur[1]] = cur[0]
    return d1, d2

def read_pair_list(file):
    l1 = []
    l2 = []
    with open(file) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            l1.append(int(cur[0]))
            l2.append(int(cur[1]))
    return l1, l2

def read_triples(file):
    tri = set()
    with open(file) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            tri.add((cur[0], cur[1], cur[2]))
    return tri

def save_triples(tri, file):
    with open(file, 'w') as f:
        for t in tri:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')

def save_links(pair, file):
    with open(file, 'w') as f:
        for p in pair:
            f.write(p[0] + '\t' + p[1] +'\n')

def cosine_matrix(A, B):
    A_sim = torch.mm(A, B.t())
    a = torch.norm(A, p=2, dim=-1)
    b = torch.norm(B, p=2, dim=-1)
    cos_sim = A_sim / a.unsqueeze(-1)
    cos_sim /= b.unsqueeze(-2)
    return cos_sim

def change_retain(file, file2):
    id_ent1, ent_dict1 = read_dict(file + '/kg1_ent_ids')
    id_ent2, ent_dict2 = read_dict(file + '/kg2_ent_ids')

    e_embed = np.load(file + '/ent_embeds.npy')


    ent_dict, id_ent = read_dict(file2 + '/ent_dict')
    e_embed_new = np.zeros((len(ent_dict), e_embed.shape[1]))

    test, _ = read_dict(file2 + '/pair')
    for i in range(len(ent_dict)):
        if ent_dict[str(i)] in id_ent1:
            e_embed_new[i] = e_embed[int(id_ent1[ent_dict[str(i)]])]
        elif ent_dict[str(i)] in id_ent2:
            e_embed_new[i] = e_embed[int(id_ent2[ent_dict[str(i)]])]
    
    
    kg1, kg2 = read_pair_list(file2 + '/test')
    ans_pair = set()
    for cur in test:
        ans_pair.add((int(cur), int(test[cur])))
    e1 = e_embed_new[kg1]
    e2 = e_embed_new[kg2]
    e1 = torch.Tensor(e1)
    e2 = torch.Tensor(e2)
    sim = cosine_matrix(e1, e2)
    rank = (-sim).argsort()
    count = 0
    res_pair = set()
    for i in range(rank.shape[0]):
        res_pair.add((kg1[i], kg2[rank[i][0]]))
    print(len(res_pair & ans_pair) / len(ans_pair))

def change(file, file2, lang):
    id_ent1, ent_dict1 = read_dict(file + '/kg1_ent_ids')
    id_ent2, ent_dict2 = read_dict(file + '/kg2_ent_ids')
    id_r1, r_dict1 = read_dict(file + '/kg1_rel_ids')
    id_r2, r_dict2 = read_dict(file + '/kg2_rel_ids')
    e_embed = np.load(file + '/ent_embeds.npy')
    r_embed = np.load(file + '/rel_embeds.npy')
    ent_dict, id_ent = read_dict(file2 + '/ent_dict')
    r_dict, id_r = read_dict(file2 + '/rel_dict')
    e_embed_new = e_embed.copy()
    r_embed_new = r_embed.copy()
    test, _ = read_dict(file2 + '/test')
    for i in range(len(ent_dict)):
        if ent_dict[str(i)] in id_ent1:
            e_embed_new[i] = e_embed[int(id_ent1[ent_dict[str(i)]])]
        else:
            e_embed_new[i] = e_embed[int(id_ent2[ent_dict[str(i)]])]

    for i in range(len(r_dict)):
        if r_dict[str(i)] in id_r1:
            r_embed_new[i] = e_embed[int(id_r1[r_dict[str(i)]])]
        else:
            r_embed_new[i] = e_embed[int(id_r2[r_dict[str(i)]])]
    
    np.save(file + '/ent_' + lang + '.npy', e_embed_new)
    np.save(file + '/rel_' + lang + '.npy', r_embed_new)
    kg1, kg2 = read_pair_list(file2 + '/test')
    ans_pair = set()
    for cur in test:
        ans_pair.add((int(cur), int(test[cur])))
    e1 = e_embed_new[kg1]
    e2 = e_embed_new[kg2]
    e1 = torch.Tensor(e1)
    e2 = torch.Tensor(e2)
    sim = cosine_matrix(e1, e2)
    rank = (-sim).argsort()
    count = 0
    res_pair = set()
    with open(file2 + '/pair.txt', 'w') as f:
        for i in range(rank.shape[0]):
            f.write(str(kg1[i]) + '\t'  + str(kg2[rank[i][0]]) + '\n')
            res_pair.add((kg1[i], kg2[rank[i][0]]))
    print(len(res_pair & ans_pair) / len(res_pair))

def change_to_name(file_in, file_out, method):
    ent_dict, id_ent = read_dict(file_in + '/ent_dict')
    r_dict, id_r = read_dict(file_in + '/rel_dict')
    tri1 = read_triples(file_in + '/test_triples_1_suf')
    tri2 = read_triples(file_in + '/test_triples_2_suf')
    new_tri1 = set()
    new_tri2 = set()
    for t in tri1:
        new_tri1.add((ent_dict[t[0]], r_dict[t[1]], ent_dict[t[2]]))
    for t in tri2:
        new_tri2.add((ent_dict[t[0]], r_dict[t[1]], ent_dict[t[2]]))
    save_triples(new_tri1, file_out + '/' + method + '_rel_triples_1')
    save_triples(new_tri2, file_out + '/' + method + '_rel_triples_2')




# change('D_W', '../datasets/D_W', 'w')
# change_to_name('../datasets/D_Y', '../changeData/D_Y', 'anchor')
# change_to_name('../datasets/D_W', '../changeData/D_W', 'anchor')
# change_to_name('../datasets/dbp_z_e', '../changeData/dbp_z_e', 'anchor')
# change_to_name('../datasets/dbp_j_e', '../changeData/dbp_j_e', 'anchor')
# change_to_name('../datasets/dbp_f_e', '../changeData/dbp_f_e', 'anchor')

# change_to_name('../datasets/D_Y', '../changeData/D_Y', 'lore')
# change_to_name('../datasets/D_W', '../changeData/D_W', 'lore')
# change_to_name('../datasets/dbp_z_e', '../changeData/dbp_z_e', 'lore')
# change_to_name('../datasets/dbp_j_e', '../changeData/dbp_j_e', 'lore')
# change_to_name('../datasets/dbp_f_e', '../changeData/dbp_f_e', 'lore')

# change_retain('anchor_w', '../datasets/D_W')
# change_retain('anchor_y', '../datasets/D_Y')
# change_retain('anchor_z', '../datasets/dbp_z_e')
# change_retain('anchor_j', '../datasets/dbp_j_e')
# change_retain('anchor_f', '../datasets/dbp_f_e')

# change_retain('lore_w', '../datasets/D_W')
# change_retain('lore_y', '../datasets/D_Y')
# change_retain('lore_z', '../datasets/dbp_z_e')
change_retain('lore_j', '../datasets/dbp_j_e')
# change_retain('lore_f', '../datasets/dbp_f_e')
