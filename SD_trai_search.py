import os
import sys
import time
import glob
import numpy as np
import torch
from cnn import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from SD_cnn.space_divide import Network
#from SD_cnn.space import Network
#from cnn.model_search import Network
from SD_cnn.architect import Architect



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--rand_frenq', type=int, default=100, help='rand frequence')
parser.add_argument('--arc_weight_constrain', type=float, default=0.0175, help='weight for arc constrain loss')
parser.add_argument('--arc_weight_constrain_up_down', action='store_true', default=True, help='')


parser.add_argument('--glob_arc', action='store_true', default=True, help='')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler( 'log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)

  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args ,args.init_channels, CIFAR_CLASSES, args.layers, criterion , args.auxiliary,args.auxiliary_weight)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)


  step_num = 25000//args.batch_size

  for epoch in range(args.epochs):

    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    #genotype = model.genotype()
    #logging.info('genotype = %s', genotype)
    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr ,step_num)
    logging.info('train_acc %f', train_acc)

    archi_path = "exp_data" + str(epoch) + "_epoch_archi.pt"

    torch.save(model.get_arc_weight(), archi_path)

    # validation
    #valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #logging.info('valid_acc %f', valid_acc)

    scheduler.step()


def train(train_queue, valid_queue, model, architect,
          criterion, optimizer, lr,step_num ):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):

    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    #architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    #optimizer.zero_grad()


    if step%args.rand_frenq == 0:
      SD_model = new_SD_model(args, model)  # 创建新SDmodel
      SD_model.train()
      optimizerSD = torch.optim.SGD(
        SD_model.get_SDnet_parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)  # 创建优化器
      architect = Architect(SD_model, args)


    architect.step(input, target, input_search, target_search,
                   lr, optimizerSD, unrolled=args.unrolled)




    optimizerSD.zero_grad()

    #f= open('arc.txt' , 'w')
    #for name ,_  in SD_model.get_SDnet_named_parameters():
      #print(name , file=f)

    #f.close()

    logits ,  logits_aux = SD_model(input)
    loss = criterion(logits, target)

    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm_(SD_model.get_SDnet_parameters(), args.grad_clip)
    optimizerSD.step()  # 优化器step() ,完成梯度更新


    # 将SD_net 中参数写回
    if step % args.rand_frenq == args.rand_frenq-1 or step in [ step_num , step_num -1 ]:
        SD_model.write_back_Fmodel_para()
        SD_model.F_model.update_arc_weight()# 写回结构参数
        
        if args.glob_arc:
          SD_model.F_model.write_back_rand_glob_arc_weight()


    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    print(step)

    #if step == 0 :
      #f= open('arc.txt' , 'w')
      #for name , _ in model.named_parameters():
       # print(name , file=f)
      #f.close()

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


from SD_cnn.space_divide import SD_Network

def new_SD_model(args,model):
  net_rand = model.get_rand_opration_matrix()  # father model 产生operation 随机矩阵
  model.get_rand_opration_weight(net_rand)  # farther model 产生对应opration 的weight
  if args.glob_arc:
    model.copy_rand_glob_arc_weight()
    
  SD_model = SD_Network(args,model)
  SD_model.copy_Fmodel_para()  # 复制对应参数，数据不共享相同内存
  SD_model = SD_model.cuda()

  return SD_model


if __name__ == '__main__':
  main()

