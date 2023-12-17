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



def change(file, file_out):
    ent_dict, _ = read_dict(file + '/ent_dict')
    rel_dict, _ = read_dict(file + '/rel_dict')
    tri1 = read_triples(file + '/triples_1')
    tri2 = read_triples(file + '/triples_2')
    train_dict, _ = read_dict(file + '/train_links')
    valid_dict, _ = read_dict(file + '/valid_links')
    test_dict, _ = read_dict(file + '/test')
    train = set()
    valid = set()
    test = set()
    new_tri1 = set()
    new_tri2 = set()
    for t in tri1:
        new_tri1.add((ent_dict[t[0]], rel_dict[t[1]], ent_dict[t[2]]))
    for t in tri2:
        new_tri2.add((ent_dict[t[0]], rel_dict[t[1]], ent_dict[t[2]]))
    for cur in train_dict:
        train.add((ent_dict[cur], ent_dict[train_dict[cur]]))
    for cur in valid_dict:
        valid.add((ent_dict[cur], ent_dict[valid_dict[cur]]))
    for cur in test_dict:
        test.add((ent_dict[cur], ent_dict[test_dict[cur]]))
    save_triples(new_tri1, file_out + '/rel_triples_1')
    save_triples(new_tri2, file_out + '/rel_triples_2')
    save_links(train, file_out + '/train_links')
    save_links(valid, file_out + '/valid_links')
    save_links(test, file_out + '/test_links')

change('../datasets/D_Y', 'D_Y')
change('../datasets/D_W', 'D_W')

