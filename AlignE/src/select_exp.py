import argparse
def read_tri(file):
    tri = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri.add((cur[0], cur[1], cur[2]))
    return tri
def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            l.append((cur[0], cur[1]))
    return l
def read_link(file):
    d1 = {}
    d2 = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            d1[cur[0]] = cur[1]
            d2[cur[1]] = cur[0]
    return d1, d2
def get_2_no_nec_kg(file, exp, split):
    tri1 = read_tri(file + '/triples_1')
    tri2 = read_tri(file + '/triples_2')
    with open(file + exp) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if int(cur[0]) < split:
                tri1 -= {(cur[0], cur[1], cur[2])}
            else:
                tri2 -= {(cur[0], cur[1], cur[2])}
    with open(file + '/test_triples_1_nec', 'w') as f:
        for t in tri1:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')
    with open(file + '/test_triples_2_nec', 'w') as f:
        for t in tri2:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')

def get_2_nec_kg(file, exp, split):
    tri1 = read_tri(file + '/triples_1')
    tri2 = read_tri(file + '/triples_2')
    d1, d2 = read_link(file + '/pair')
    tar_tri1 = set()
    tar_tri2 = set()
    for cur in tri1:
        if cur[0] in d1 or cur[2] in d1:
            tar_tri1.add(cur)
    for cur in tri2:
        if cur[0] in d2 or cur[2] in d2:
            tar_tri2.add(cur)
    tri1 -= tar_tri1
    tri2 -= tar_tri2
    print(len(tar_tri1 | tar_tri2))
    count1 = 0
    count2 = 0
    nec_tri1 = set()
    nec_tri2 = set()
    with open(file + exp) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if int(cur[0]) < split:
                tri1.add((cur[0], cur[1], cur[2]))
                nec_tri1.add((cur[0], cur[1], cur[2]))
            else:
                tri2.add((cur[0], cur[1], cur[2]))
                nec_tri2.add((cur[0], cur[1], cur[2]))
                count2 += 1
    print(len(nec_tri1 | nec_tri2))
    print('sparsity :', 1 - (len(nec_tri1 | nec_tri2)) / (len(tar_tri1 | tar_tri2)))
    with open(file + '/test_triples_1_suf', 'w') as f:
        for t in tri1:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')
    with open(file + '/test_triples_2_suf', 'w') as f:
        for t in tri2:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')

def read_from_exp(file, num, file_out):
    exp = []
    with open(file) as f:
        cur_exp = []
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if cur[0] == '0' and cur[1] == '0' and cur[2] == '0':
                exp.append(cur_exp)
                # print(len(cur_exp))
                cur_exp = []
            else:
                cur_exp.append((cur[0], cur[1], cur[2]))
    with open(file_out, 'w') as f:
        for cur_exp in exp:
            # print(cur_exp)
            for cur in cur_exp[:num]:
                f.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')


def read_from_exp_anchor(file, num, file_out, rules):
    exp = []
    with open(file) as f:
        cur_exp = []
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if cur[0] == '0' and cur[1] == '0' and cur[2] == '0':
                exp.append(cur_exp)
                # print(len(cur_exp))
                cur_exp = []
            else:
                cur_exp.append((cur[0], cur[1], cur[2]))
    with open(file_out, 'w') as f:
        for i in range(len(exp)):
            cur_exp = exp[i]
            rule = rules[i]
            need = rule[0]
            delete = rule[1]
            j = 0
            for cur in cur_exp[:num]:
                if cur not in delete:
                    if cur in need:
                        need.remove(cur)
                    f.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')
                    j += 1
                    if len(need) == num - j:
                        break 
            for cur in need:
                f.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')

def read_from_exp_lore(file, num, file_out, rules):
    exp = []
    with open(file) as f:
        cur_exp = []
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if cur[0] == '0' and cur[1] == '0' and cur[2] == '0':
                exp.append(cur_exp)
                # print(len(cur_exp))
                cur_exp = []
            else:
                cur_exp.append((cur[0], cur[1], cur[2]))
    with open(file_out, 'w') as f:
        for i in range(len(exp)):
            cur_exp = exp[i]
            rule = rules[i]
            need = rule[0]
            delete = rule[1]
            j = 0
            for cur in cur_exp[:num]:
                if cur not in delete:
                    if cur in need:
                        need.remove(cur)
                    f.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')
                    j += 1
                    if len(need) == num - j:
                        break 
            for cur in need:
                f.write(cur[0] + '\t' + cur[1] + '\t' + cur[2] + '\n')

def read_from_exp_name(file, num, file_out, d_file1, d_file2):
    exp = []
    with open(file) as f:
        cur_exp = []
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if cur[0] == '0' and cur[1] == '0' and cur[2] == '0':
                exp.append(cur_exp)
                # print(len(cur_exp))
                cur_exp = []
            else:
                cur_exp.append((cur[0], cur[1], cur[2]))
    d, _ = read_link(d_file1)
    d1,_ = read_link(d_file2)
    with open(file_out, 'w') as f:
        for cur_exp in exp:
            # print(cur_exp)
            for cur in cur_exp[:num]:
                f.write(d[cur[0]] + '\t' + d1[cur[1]] + '\t' + d[cur[2]] + '\n')
            f.write('------------------------------')

def read_from_exp_name1(file,  file_out, d_file1, d_file2):
    exp = []
    with open(file) as f:
        cur_exp = []
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            cur_exp.append((cur[0], cur[1], cur[2]))
    d, _ = read_link(d_file1)
    d1,_ = read_link(d_file2)
    with open(file_out, 'w') as f:
        for cur in cur_exp:
            f.write(d[cur[0]] + '\t' + d1[cur[1]] + '\t' + d[cur[2]] + '\n')
        f.write('------------------------------')

def read_rule(file):
    rules = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            need_tri = set()
            delete_tri = set()
            cur = line.strip().split(',d')
            print(cur[0])
            need = cur[0].split('\t')
            print(need)
            if len(need) > 1:
                for i in range(len(need) - 1):
                    tri = eval(need[i + 1])
                    need_tri.add((str(tri[0]), str(tri[1]), str(tri[2])))    
            delete = cur[1].split('\t')
            if len(delete) > 1:
                for i in range(len(delete) - 1):
                    tri = eval(delete[i + 1])
                    delete_tri.add((str(tri[0]), str(tri[1]), str(tri[2])))
            rules.append((need_tri, delete_tri))
    return rules
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(f'arguments for selecting important features for baselines in Explanation Generation')
    parser.add_argument('lang', type=str, help='which dataset', default='z')
    parser.add_argument('method', type=str, help='baseline name', default='lime')
    # parser.add_argument('--version', type=str, help='the hop num of candidate neighbor', default=1)
    parser.add_argument('--num', type=int, help='the len of explanation', default=15)
    args = parser.parse_args()
    lang = args.lang
    if lang == 'zh':
        lang = 'z'
    elif lang == 'fr':
        lang = 'f'
    elif lang == 'ja':
        lang = 'j'
    method = args.method
    num = args.num
    method1 = 'lime'
    align_method = 'MTransE'
    if method == 'lore':
        if lang == 'w':
            rules = read_rule('../datasets/D_W/exp_' + method)
            read_from_exp_lore('../datasets/D_W/exp_' + method1, 9, '../datasets/D_W/exp_'+ method +'1', rules)
        elif lang == 'y':
            rules = read_rule('../datasets/D_Y/exp_' + method)
            read_from_exp_lore('../datasets/D_Y/exp_' + method1, 7, '../datasets/D_Y/exp_'+ method +'1', rules)
        else:
            rules = read_rule('../datasets/dbp_' + lang + '_e/exp_' + method)
            read_from_exp_lore('../datasets/dbp_' + lang + '_e/exp_' + method1, 14, '../datasets/dbp_' + lang + '_e/exp_'+ method +'1', rules)
    elif method == 'anchor':
        if lang == 'w':
            rules = read_rule('../datasets/D_W/exp_' + method)
            read_from_exp_anchor('../datasets/D_W/exp_' + method1, 7, '../datasets/D_W/exp_'+ method +'1', rules)
        elif lang == 'y':
            rules = read_rule('../datasets/D_Y/exp_' + method)
            read_from_exp_anchor('../datasets/D_Y/exp_' + method1, 6, '../datasets/D_Y/exp_'+ method +'1', rules)
        else:
            rules = read_rule('../datasets/dbp_' + lang + '_e/exp_' + method)
            read_from_exp_anchor('../datasets/dbp_' + lang + '_e/exp_' + method1, 10, '../datasets/dbp_' + lang + '_e/exp_'+ method +'1', rules)
    else:
        if lang == 'w':
            read_from_exp('../datasets/D_W/exp_' + method, 6, '../datasets/D_W/exp_'+ method +'1')
        elif lang == 'y':
            read_from_exp('../datasets/D_Y/exp_' + method, 6, '../datasets/D_Y/exp_'+ method +'1')
        else:
            read_from_exp('../datasets/dbp_' + lang + '_e/exp_' + method, 11, '../datasets/dbp_' + lang + '_e/exp_'+ method +'1')
        # read_from_exp_name('/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/exp_' + method, 12, '/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/exp_' + method +'name','/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/ent_dict','/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/rel_dict')
    if lang == 'w':
        split = len(read_list('../datasets/D_W/ent_dict1'))
    elif lang == 'y':
        split = len(read_list('../datasets/D_Y/ent_dict1'))
    if lang == 'w':
        get_2_no_nec_kg('../datasets/D_W',  '/exp_' + method + '1', split)
        get_2_nec_kg('../datasets/D_W',  '/exp_' + method + '1', split)
    elif lang == 'y':
        get_2_no_nec_kg('../datasets/D_Y',  '/exp_' + method + '1', split)
        get_2_nec_kg('../datasets/D_Y',  '/exp_' + method + '1', split)
    else:
        split = len(read_list('../datasets/dbp_' + lang + '_e/ent_dict1'))
        get_2_no_nec_kg('../datasets/dbp_' + lang + '_e/',  '/exp_' + method +'1', split)
        get_2_nec_kg('../datasets/dbp_' + lang + '_e/',  '/exp_' + method +'1', split)
    # read_from_exp_name1('/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/exp_' + method,  '/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/exp_' + method +'name','/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/ent_dict','/data/xbtian/Explain/' + align_method + '/datasets/dbp_' + lang + '_e/rel_dict')