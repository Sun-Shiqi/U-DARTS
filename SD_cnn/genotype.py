import torch
from cnn.genotypes import PRIMITIVES
import torch.nn.functional as F
import os

############## 产生一个net 的所有cell结构


def genotype(weight , step, multiplier):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(step):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene

    gene_normal = _parse(weight)


    concat = [_ for _ in range(2 + step - multiplier, step + 2)]
    genotype = (
        gene_normal, concat
    )
    return genotype

def strict_num1( cell_weight ):########### 限制所有的操作
    for j in cell_weight:
        for k in range(len(PRIMITIVES)):
            if 'pool' in PRIMITIVES[k]:
                j[k] = j[k] * 0.7

            if 'skip' in PRIMITIVES[k]:
                j[k] = j[k] * 0.5
            if 'dil' in PRIMITIVES[k]:
                j[k] = j[k] * 0.6
    return cell_weight


def strict_num_max_pool( cell_weight ,genotype ):##################限制pool 操作
    pool_num = 0
    for name in genotype[0]:

        if 'max_pool' in name[0]:
            pool_num = pool_num +1
    if pool_num > 2:
        for i in cell_weight:
            for j in range(len(PRIMITIVES)):
                if 'max_pool' in PRIMITIVES[j]:
                    i[j] = i[j] *  0.3
    return cell_weight


def strict_num_skip( cell_weight ,genotype ):##################限制pool 操作
    pool_num = 0
    for name in genotype[0]:

        if 'skip_connect' in name[0]:
            pool_num = pool_num +1
    if pool_num > 2:
        for i in cell_weight:
            for j in range(len(PRIMITIVES)):
                if 'skip_connect' in PRIMITIVES[j]:
                    i[j] = i[j] *  0.4
    return cell_weight

def strict_avg_pool( cell_weight ,genotype ):##################限制pool 操作
    pool_num = 0
    for name in genotype[0]:

        if 'skip_connect' in name[0]:
            pool_num = pool_num +1
    if pool_num > 2:
        for i in cell_weight:
            for j in range(len(PRIMITIVES)):
                if 'skip_connect' in PRIMITIVES[j]:
                    i[j] = i[j] *  0.4
    return cell_weight

def strict_dil( cell_weight ,genotype ):##################限制pool 操作
    pool_num = 0
    for name in genotype[0]:

        if 'dil' in name[0]:
            pool_num = pool_num +1
    if pool_num > 4:
        for i in cell_weight:
            for j in range(len(PRIMITIVES)):
                if 'dil' in PRIMITIVES[j]:
                    i[j] = i[j] *  0.4
    return cell_weight


def get_glob_arc(glob_arc):
    layer = 20 -3
    encode = [ sum(n for n in range(i+1)) for i in range(layer)]
    encode.append(len(glob_arc)-1)
    anser =[]
    for i in range(layer):
        anser.append(glob_arc.index(max(glob_arc[encode[i]:encode[i+1]]))-encode[i])
    return anser


def genecell ():

    file_path =  './exp_data99_epoch_archi.pt'
    arch = []

    cell_arch = torch.load(file_path)
    a = get_glob_arc(cell_arch[-1].detach().cpu().numpy().tolist())

    for i in cell_arch[:20]:
        i = i.detach().cpu().numpy()
        arch.append(genotype(i , 4 ,4 ))


    return arch,a





