from collections import defaultdict
import numpy as np 
import torch
import math
from itertools import combinations

def count(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    print(len(s1 & s2) / len(s2))

def no_align(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    return s2 - (s1 & s2), s1 & s2

def wrong_align(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    return s1 - (s1 & s2), s1 & s2


def diff_align(file1, file2):
    s1 = set()
    s2 = set()
    d1 = {}
    d2 = {}
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
            d1[cur[0]] = cur[1]
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
            d2[cur[0]] = cur[1]
    return s1 - (s1 & s2), s1 & s2, d1, d2

def right_align(file1, file2):
    s1 = set()
    s2 = set()
    d1 = {}
    d2 = {}
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
            d1[cur[0]] = cur[1]
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
            d2[cur[0]] = cur[1]
    return s1 & s2, d1, d2


def read_tri_set(file):
    tri = defaultdict(set)
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri[cur[0]].add((cur[1], cur[2]))
    return tri
def read_tri(file):
    tri = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri.add((cur[0], cur[1], cur[2]))
    return tri

def read_tri_list(file):
    tri =[]
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri.append((cur[0], cur[1], cur[2]))
    return tri

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
def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            l.append((cur[0], cur[1]))
    return l


def get_2_hop_direct(tri):
    one_hop = defaultdict(set)
    one_hop_inverse = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop_inverse[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for neigh2 in one_hop[neigh]:
                two_hop[cur].add(neigh2)
                for hop1 in neigh_r[(cur, neigh)]:
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
            if neigh not in one_hop_inverse:
                continue
            for neigh2 in one_hop_inverse[neigh]:
                two_hop[cur].add(neigh2)
                for hop1 in neigh_r[(cur, neigh)]:
                    for hop2 in neigh_r[(neigh2, neigh)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_2_hop_no_mid(tri, r_func):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
           
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_2_hop(tri, r_func):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, neigh, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
           
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_1_hop_direct(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    one_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        one_hop_r[cur[1]].add(cur[0])
    return one_hop, neigh_r, one_hop_r


def get_1_hop(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    one_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
        one_hop_r[cur[1]].add(cur[0])
        one_hop_r[cur[1]].add(cur[2])
    return one_hop, neigh_r, one_hop_r
            
def get_r_func(file):
    d = {}
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            d[cur[0]] = float(cur[1])
    return d

def read_pair(file):
    p = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            p.add((cur[0], cur[1]))
    return p

def get_2_hop_all(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])

    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
        two_hop[cur] |= one_hop[cur]
           
    # print('len two hop:' ,len(two_hop))
    return two_hop
