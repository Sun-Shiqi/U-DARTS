import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from cnn.genotypes import PRIMITIVES
from SD_cnn.constrain_arch import  get_para_num


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.get_SDnet_parameters()).data#####PARA
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.get_SDnet_parameters()).mul_(self.network_momentum)  #####PARA
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.get_SDnet_parameters())).data + self.network_weight_decay*theta  #####PARA
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.get_SDnet_parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.get_SDnet_named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.get_SDnet_parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.get_SDnet_parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.get_SDnet_parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


'''
class Arc_constrain(object):

  def __init__(self, model, args):
    self.opration_constrain = get_para_num()

    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.args = args
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def step(self):
    self.optimizer.zero_grad()

    loss = self.loss_()
    loss.backward()

    self.optimizer.step()
    print('constrain is completed' )



  def loss_(self):
    loss_list=[]
    for cell_weight,opration_id in zip(self.model.F_model.net_rand_arc_weight,
                                       self.model.F_model.net_rand):
      for op_weight , op_id in zip(cell_weight , opration_id):
        loss_list.append(torch.pow( torch.sub(op_weight ,
                                         op_weight.mul(self.opration_constrain[PRIMITIVES[op_id]])),
                               2))

    loss = torch.add(loss_list[0] , loss_list [1])
    for i in range( 2 , len(loss_list)):
      loss = torch.add(loss , loss_list[i])

    loss  = loss ** 0.5

    #print(loss)

    return loss.mul(self.args.arc_constrain_weight)
'''

