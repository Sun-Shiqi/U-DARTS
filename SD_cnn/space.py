import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn.operations import *
from torch.autograd import Variable
from cnn.genotypes import PRIMITIVES
from cnn.genotypes import Genotype
import random


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))





class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class SD_MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(SD_MixedOp, self).__init__()
        self._C = C
        self._stride  = stride
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

        self.CR_op = OPS[PRIMITIVES[1]](C, stride, False)


    def add_CR(self, indice):
        self.CR_op = OPS[PRIMITIVES[indice]](self._C, self._stride, False).cuda()
        #if PRIMITIVES[indice] not in  ['skip_connect' ,'avg_pool_3x3', 'max_pool_3x3' ]:
        for i , j in zip(self.CR_op.parameters() , self._ops[indice].parameters()):
            i.data = j



    def forward(self, x, weights , indice):
        self.add_CR(indice)
        sum = 0


        return weights* self.CR_op(x)


class SD_Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(SD_Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = SD_MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights , rand_net):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j] ,rand_net[offset + j] ) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)



class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x





class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, auxiliary, steps=4, multiplier=4, stem_multiplier=3 ):
        super(Network, self).__init__()
        self.auxiliary = auxiliary
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps

        self.operation_num = self.get_operation_num()

        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.CR_stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = SD_Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        self.CR_global_pooling = nn.AdaptiveAvgPool2d(1)
        self.CR_classifier = nn.Linear(C_prev, num_classes)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

        self._initialize_alphas()
        self.initialize_weight_matrix()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,self.auxiliary).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.CR_stem(input)
        logits_aux = None
        self.rand_net = self.get_rand_opration_matrix()
        weights = self.get_rand_opration_weight(self.rand_net)

        for i in weights:
            for j in i :
                j.data = torch.sigmoid(j)

        for i, cell in enumerate(self.cells):

            #if cell.reduction:
                #weights = F.softmax(self.alphas_reduce, dim=-1)
            #else:
                #weights = F.softmax(self.alphas_normal, dim=-1)


            s0, s1 = s1, cell(s0, s1, weights[i] , self.rand_net[i])

            if i == 2 * self._layers // 3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.CR_global_pooling(s1)
        logits = self.CR_classifier(out.view(out.size(0), -1))
        return logits , logits_aux

    def _loss(self, input, target):
        logits, _  = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
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

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def get_no_auxi_paramiters(self ,  recurse = True):
        for name, param in self.get_no_name_auxi_paramiters(recurse=recurse):
            yield param



    def get_no_name_auxi_paramiters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)

        for elem in gen:
            if 'auxiliary_head' not in elem[0]:
                yield elem


    def initialize_weight_matrix(self):

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.arc_weight_list = []

        for i in range (self._layers):
            self.arc_weight_list.append(Variable(1e-3 * torch.randn(k, num_ops).cuda() , requires_grad=True))

    def get_operation_num(self):
        operation_num = 0  # operation 的总数

        for i in range(self._steps):
            for j in range(2 + i):
                operation_num = operation_num + 1

        return operation_num


    def get_rand_opration_matrix(self):
        net_rand = []
        for i in range(self._layers):
            net_rand.append([ random.randint(0 , len(PRIMITIVES) -1 )  for _ in range (self.operation_num) ])

        return net_rand

    def get_rand_opration_weight(self , net_rand):
        weight = []
        for i , j in zip(net_rand , self.arc_weight_list):
            weight.append([jj[ii] for ii ,jj in zip(i ,j)])

        return weight

    def update_arc_weight(self , anser_weight , net_rand , result_weight):
        for i in range(len(net_rand)):
            for j in range(len(net_rand[i])):
                anser_weight[i][j][net_rand[i][j]] = result_weight[i][j]











