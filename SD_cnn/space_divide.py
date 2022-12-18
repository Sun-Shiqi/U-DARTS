import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn.operations import *
from torch.autograd import Variable
from cnn.genotypes import PRIMITIVES
from cnn.genotypes import Genotype
import random
import  sys

from SD_cnn.constrain_arch import get_para_num


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

    def __init__(self, C, stride, op_id):
        super(SD_MixedOp, self).__init__()
        self._C = C
        self.id = op_id
        self._stride = stride

        self.CR_op = OPS[PRIMITIVES[op_id]](C, stride, False)

    def forward(self, x, weights):
         return weights *self.CR_op(x)


class SD_Cell(nn.Module):

    def __init__(self, net_rand, steps,
                 multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
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
        sum1 = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = SD_MixedOp(C, stride, net_rand[sum1])
                sum1 = sum1 + 1
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


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
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
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Network(nn.Module):

    def __init__(self,args, C, num_classes, layers, criterion, auxiliary, auxiliary_weight, steps=4, multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self.args = args
        self.auxiliary = auxiliary
        self.auxiliary_weight = auxiliary_weight
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
            if i>2:
                cell = Cell(steps, multiplier, int(C_prev_prev*1.25), C_prev, C_curr, reduction, reduction_prev)
            else:
                cell = Cell(steps, multiplier,C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
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
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.auxiliary,
                            self.auxiliary_weight).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.CR_stem(input)
        logits_aux = None
        self.rand_net = self.get_rand_opration_matrix()
        weights = self.get_rand_opration_weight(self.rand_net)

        for i in weights:
            for j in i:
                j.data = torch.sigmoid(j)

        for i, cell in enumerate(self.cells):

            # if cell.reduction:
            # weights = F.softmax(self.alphas_reduce, dim=-1)
            # else:
            # weights = F.softmax(self.alphas_normal, dim=-1)

            s0, s1 = s1, cell(s0, s1, weights[i], self.rand_net[i])

            if i == 2 * self._layers // 3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.CR_global_pooling(s1)
        logits = self.CR_classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    def _loss(self, input, target):
        logits, _ = self(input)
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

    def get_no_auxi_paramiters(self, recurse=True):
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

        for i in range(self._layers):
            self.arc_weight_list.append(Variable(1e-1 * torch.randn(k, num_ops).cuda().sigmoid_(), requires_grad=True))

        k2 = sum(1 for i in range(self._layers -3) for n in range(i+1))
        self.glob_arc_weight_list = []
        self.glob_arc_weight_list.append(Variable(1e-1 * torch.randn(k2).cuda().sigmoid_(), requires_grad=True))

    def copy_rand_glob_arc_weight(self):
        self.rand_glob_arc = Variable(1e-1 * torch.randn(self._layers-3).cuda().sigmoid_(), requires_grad=True)
        every_pair_num = [sum(range(0, i + 1)) for i in range(self._layers - 3)]
        self.rand_idex = [random.randint(0, i) for i in range(self._layers - 3)]
        self.rand_glob_id = [x + y for x, y in zip(every_pair_num, self.rand_idex)]

        for i ,id  in enumerate(self.rand_glob_id):
            self.rand_glob_arc[i].data.copy_(self.glob_arc_weight_list[0][id].data)

    def write_back_rand_glob_arc_weight(self):
        for i ,id  in enumerate(self.rand_glob_id):
            self.glob_arc_weight_list[0][id].data.copy_(self.rand_glob_arc[i].data)


    def get_arc_weight(self):
        if self.args.glob_arc:
            return self.arc_weight_list+self.glob_arc_weight_list
        else:
            return self.arc_weight_list

    def get_operation_num(self):
        operation_num = 0  # operation 的总数

        for i in range(self._steps):
            for j in range(2 + i):
                operation_num = operation_num + 1

        return operation_num

    def get_rand_opration_matrix(self):
        net_rand = []
        for i in range(self._layers):
            net_rand.append([random.randint(0, len(PRIMITIVES) - 1) for _ in range(self.operation_num)])

        self.net_rand = net_rand
        return net_rand

    def get_rand_opration_weight(self, net_rand):
        weight = []
        for i, j in zip(net_rand, self.arc_weight_list):
            weight.append([jj[ii] for ii, jj in zip(i, j)])

        weight2 = []
        for i in weight:
            t1 = Variable(torch.randn(len(i)).cuda(), requires_grad=True)
            for j in range(len(i)):
                t1[j].data.copy_(i[j].data)
            weight2.append(t1)
        self.net_rand_arc_weight = weight2
        return weight2

    def update_arc_weight(self):
        for i in range(len(self.net_rand)):
            for j in range(len(self.net_rand[i])):
                self.arc_weight_list[i][j][self.net_rand[i][j]].data.copy_(self.net_rand_arc_weight[i][j].data)

    def new_SD_model(self):
        net_rand = self.get_rand_opration_matrix()  # father model 产生operation 随机矩阵
        self.get_rand_opration_weight(net_rand)  # farther model 产生对应opration 的weight
        self.SD_model = SD_Network(self)

        self.SD_model.copy_Fmodel_para()  # 复制对应参数，数据不共享相同内存
        self.SD_model = self.SD_model.cuda()


class SD_Network(nn.Module):
    def __init__(self, args, model, steps=4, multiplier=4, stem_multiplier=3):
        super(SD_Network, self).__init__()
        self.F_model = model

        self.args = args

        self.arc_weight_constrain_L2 = args.arc_weight_constrain
        self.arc_weight_constrain_L2_up_down = args.arc_weight_constrain_up_down

        self.auxiliary = self.F_model.auxiliary
        self.auxiliary_weight = self.F_model.auxiliary_weight
        self._C = self.F_model._C
        self._num_classes = self.F_model._num_classes
        self._layers = self.F_model._layers
        self._criterion = self.F_model._criterion
        self._steps = self.F_model._steps

        self.operation_num = self.F_model.operation_num

        self._multiplier = self.F_model._multiplier

        C_curr = stem_multiplier * self.F_model._C
        self.CR_stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self._layers):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if i>2:
                cell = SD_Cell(self.F_model.net_rand[i], steps, multiplier,int(C_prev_prev*1.25), C_prev, C_curr, reduction,
                           reduction_prev)
            else:
                cell = SD_Cell(self.F_model.net_rand[i], steps, multiplier, C_prev_prev , C_prev, C_curr,
                               reduction,
                               reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            if i == 2 * self._layers // 3:
                C_to_auxiliary = C_prev

        self.CR_global_pooling = nn.AdaptiveAvgPool2d(1)
        self.CR_classifier = nn.Linear(C_prev, self._num_classes)

        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, self._num_classes)

    def new(self):
        model_new = SD_Network(self.args, self.F_model, steps=4, multiplier=4, stem_multiplier=3).cuda()
        return model_new

    def forward(self, input):
        s0 = s1 = self.CR_stem(input)
        logits_aux = None
        mid_glob = []

        weights = self.F_model.net_rand_arc_weight

        for i in weights:
            for j in i:
                j.data = torch.sigmoid(j)

        for i, cell in enumerate(self.cells):

            if self.args.glob_arc and i>2:
                s0 = self.glob_process(s0 , mid_glob, i)

            s0, s1 = s1, cell(s0, s1, weights[i])
            mid_glob.append(s1)


            #print(s1.size())

            if i == 2 * self._layers // 3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.CR_global_pooling(s1)
        logits = self.CR_classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    def glob_process(self , s0 , mid_glob ,i):
        s0_ = mid_glob[self.F_model.rand_idex[i-3]]
        f1 = torch.nn.AvgPool3d((4,1,1) , stride=(4,1,1))
        f2 = torch.nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
        f3 = torch.nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4))
        if s0.size()[1] == s0_.size()[1]:
            m1 = f1(s0_.unsqueeze(0)).squeeze()
            m2 = torch.cat((s0 , m1) , dim=1)
            return m2*self.F_model.rand_glob_arc[i-3]

        elif s0.size()[1] == 2*s0_.size()[1]:
            m1 = f2(s0_.unsqueeze(0)).squeeze()
            m2 = torch.cat((s0, m1), dim=1)
            return m2 * self.F_model.rand_glob_arc[i - 3]
        elif s0.size()[1] == 4 * s0_.size()[1]:
            m1 = f3(s0_.unsqueeze(0)).squeeze()
            m2 = torch.cat((s0, m1), dim=1)
            return m2 * self.F_model.rand_glob_arc[i - 3]
        else:
            print('数据没对齐')
            print(s0.size()[1])
            print(s0_.size()[1])
            sys.exit()





    def _loss(self, input, target):
        # 原loss
        logits, logits_auxi = self(input)
        loss_ori = self._criterion(logits, target)
        if self.auxiliary:
            loss_aux = self._criterion(logits_auxi, target)
            loss_ori += self.auxiliary_weight * loss_aux

        # arc loss
        if self.arc_weight_constrain_L2_up_down:
            self.opration_constrain = get_para_num()
            loss_list = []
            for cell_weight, opration_id in zip(self.F_model.net_rand_arc_weight,
                                                self.F_model.net_rand):
                for op_weight, op_id in zip(cell_weight, opration_id):
                    loss_list.append(op_weight.mul(self.opration_constrain[PRIMITIVES[op_id]]))

            loss = torch.add(loss_list[0], loss_list[1])
            for i in range(2, len(loss_list)):
                loss = torch.add(loss, loss_list[i])


        if self.arc_weight_constrain_L2_up_down:

            return loss_ori + loss * self.arc_weight_constrain_L2

        else:
            return loss_ori

    def copy_Fmodel_para(self):
        # 复制cell 参数
        for i in range(len(self.F_model.net_rand)):
            for j in range(len(self.F_model.net_rand[i])):
                for ii, jj in zip(self.cells[i]._ops[j].parameters(),
                                  self.F_model.cells[i]._ops[j]._ops[self.F_model.net_rand[i][j]].parameters()):
                    ii.detach().copy_(jj)
        # 复制stem参数
        for i, j in zip(self.F_model.CR_stem.parameters(), self.CR_stem.parameters()):
            j.detach().copy_(i)

        # cell 中process
        for i, j in zip(self.cells, self.F_model.cells):
            for ii, jj in zip(i.preprocess0.parameters(), j.preprocess0.parameters()):
                ii.detach().copy_(jj)
            for ii, jj in zip(i.preprocess1.parameters(), j.preprocess1.parameters()):
                ii.detach().copy_(jj)
        # calssifar
        for i, j in zip(self.F_model.CR_classifier.parameters(), self.CR_classifier.parameters()):
            j.detach().copy_(i)

        # auxiliary
        for i, j in zip(self.F_model.auxiliary_head.parameters(), self.auxiliary_head.parameters()):
            j.detach().copy_(i)

    def write_back_Fmodel_para(self):
        # 复制cell 参数
        for i in range(len(self.F_model.net_rand)):
            for j in range(len(self.F_model.net_rand[i])):
                for ii, jj in zip(self.cells[i]._ops[j].parameters(),
                                  self.F_model.cells[i]._ops[j]._ops[self.F_model.net_rand[i][j]].parameters()):
                    jj.detach().copy_(ii)
        # 复制stem参数
        for i, j in zip(self.F_model.CR_stem.parameters(), self.CR_stem.parameters()):
            # print('ori_Fmodel_para:')
            # print(i.shape)
            # print(i)
            i.detach().copy_(j)
            # print('changed_Fmodel_para:')
            # print(i)
            # print(i.shape)
            # print('SD_para:')
            # print(j)
            # print(j.shape)

        # cell 中process
        for i, j in zip(self.cells, self.F_model.cells):
            for ii, jj in zip(i.preprocess0.parameters(), j.preprocess0.parameters()):
                jj.detach().copy_(ii)
            for ii, jj in zip(i.preprocess1.parameters(), j.preprocess1.parameters()):
                jj.detach().copy_(ii)
        # calssifar
        for i, j in zip(self.F_model.CR_classifier.parameters(), self.CR_classifier.parameters()):
            i.detach().copy_(j)

        # auxiliary
        for i, j in zip(self.F_model.auxiliary_head.parameters(), self.auxiliary_head.parameters()):
            i.detach().copy_(j)

    def get_SDnet_parameters(self, recurse=True):
        for name, param in self.get_SDnet_named_parameters(recurse=recurse):
            yield param

    def get_SDnet_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if 'F_model' not in elem[0]:
                yield elem

    def arch_parameters(self):
        if self.args.glob_arc:
            return self.F_model.net_rand_arc_weight + [self.F_model.rand_glob_arc]
        else:
            return self.F_model.net_rand_arc_weight

