
import torch.nn as nn
import torch.nn.functional as F
from cnn.operations import  OPS
import torch
import numpy as np


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


# 定义深度
DEPTH = {

'none':0,
'avg_pool_3x3':0,
'max_pool_3x3':0,
'skip_connect':0,
'sep_conv_3x3':2,
'sep_conv_5x5' :2,
'dil_conv_3x3' :1,
'dil_conv_5x5' :1,

}

def count(model):
    return sum(v.numel() for name, v in model.named_parameters() )


def get_para_num ():
    alpha = 1.8
    beta = 0.1

    softmax= torch.nn.Softmax(dim = 0)
    softplus = torch.nn.Softplus()
    opration_list = nn.ModuleList()
    for op in PRIMITIVES:
        opration_list+=[OPS[op](36 , 1 , True)]


    opration_num = []

    for op in opration_list:
        opration_num.append(count(op))


    ten = torch.tensor(opration_num)

    con_para = ten.float().div_(ten.float().sum())
    con_para = 0.1*con_para
    #print("参数比例：")
    #print(con_para)



    dep_list = []
    for op in PRIMITIVES:
        dep_list.append(DEPTH[op])

    depth = torch.tensor(dep_list).float()####深度信息
    a= torch.ones(len(depth)).float()
    b= torch.div(a,(torch.add(depth ,a)))
    #print("深度信息：")
    #print(b)
    c= torch.add(b, con_para)







    t4 = c.numpy().tolist()


    anser  = dict(zip(PRIMITIVES , t4))




    return anser



#print(get_para_num ())
