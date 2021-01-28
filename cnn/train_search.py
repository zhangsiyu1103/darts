import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from datasets import TrainDataset
from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--grow_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--learning_rate_middle', type=float, default=0.05, help='min learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=60, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--grow_portion', type=float, default=1.0, help='portion of training data for grow')
parser.add_argument('--grow_freq', type=int, default=5, help='frequency of growing')
parser.add_argument('--num_grow', type=int, default=5, help='number of edges activated each time grows')

parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--darts', action='store_true', default=False, help='use original darts code')
parser.add_argument('--pc', action='store_true', default=False, help='whether to use partial channell')
parser.add_argument('--sample', action='store_true', default=False, help='whether use sampled dataset')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

t_record={"train":0, "grow":0, "grow_search":0}

CIFAR_CLASSES = 10

train_time = 0
grow_time = 0


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
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, darts = args.darts, pc = args.pc)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.sample:
    train_data = TrainDataset(root = "./new_cifar.pth", transform = train_transform, sample = True)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))
  train_grow = int(np.floor(args.grow_portion * split))
  valid_grow = int(np.floor(args.grow_portion * (num_train - split)))

  train_indices = indices[:split]
  valid_indices = indices[split:]

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
      pin_memory=True, num_workers=2)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)



  architect = Architect(model, args)

  for epoch in range(args.epochs):
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    if not args.darts:
      alphas_reduce = torch.where(model.alphas_reduce==0,torch.FloatTensor([float("-inf")]).cuda(),model.alphas_reduce)
      alphas_normal = torch.where(model.alphas_normal==0,torch.FloatTensor([float("-inf")]).cuda(),model.alphas_normal)
      logging.info(F.softmax(alphas_normal, dim=-1))
      logging.info(F.softmax(alphas_reduce, dim=-1))
    else:
      logging.info(F.softmax(model.alphas_normal, dim=-1))
      logging.info(F.softmax(model.alphas_reduce, dim=-1))


   # print("post grow")
   # alphas_reduce = torch.where(model.alphas_reduce==0,torch.FloatTensor([float("-inf")]).cuda(),model.alphas_reduce)
   # alphas_normal = torch.where(model.alphas_normal==0,torch.FloatTensor([float("-inf")]).cuda(),model.alphas_normal)
   # print(F.softmax(alphas_normal, dim=-1))
   # print(F.softmax(alphas_reduce, dim=-1))


    # training
    train_s = time.time()
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    logging.info('train_acc %f', train_acc)
    train_e = time.time()
    t_record["train"]+=(train_e-train_s)
    if not args.darts and epoch > args.grow_freq:
        architect.print_arch_grad()

    #scheduler update
    scheduler.step()
    #if architect.scheduler is not None:
    #  architect.scheduler.step()



    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)



    if not args.darts and epoch % args.grow_freq == 0 and epoch < args.epochs-10 and not epoch == 0:
      train_indices_grow = np.random.choice(train_indices, train_grow, replace = False)
      valid_indices_grow = np.random.choice(valid_indices, valid_grow, replace = False)

      train_grow_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.grow_batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices_grow),
          pin_memory=True, num_workers=2)

      valid_grow_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.grow_batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices_grow),
          pin_memory=True, num_workers=2)

      grow_s = time.time()
      grow(train_grow_queue, valid_grow_queue, model, architect, criterion, optimizer, lr, args.num_grow)
      grow_e = time.time()
      t_record["grow"]+=(grow_e-grow_s)
      if epoch > 0:
        for param_group in optimizer.param_groups:
          param_group["lr"] = args.learning_rate_middle
          param_group["initial_lr"] = args.learning_rate_middle
        optimizer.defaults["lr"] = args.learning_rate_middle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, args.grow_freq - 1, eta_min=args.learning_rate_min)

    if not args.darts and epoch == args.epochs-10:
      for param_group in optimizer.param_groups:
        param_group["lr"] = 0.01
        param_group["initial_lr"] = 0.01
      optimizer.defaults["lr"]=0.01
      #scheduler = None
      #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      #    optimizer, 10.0, eta_min=args.learning_rate_min)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                 optimizer, 10, eta_min=args.learning_rate_min)


      #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      #    optimizer, 10.0, eta_min=args.learning_rate_min)

      #for param_group in architect.optimizer.param_groups:
      #  param_group["lr"] = architect.lr
      #architect.scheduler = torch.optim.lr_scheduler.StepLR(architect.optimizer, step_size = 1, gamma=0.9)

    torch.save(t_record, "time_record.pt")

    utils.save(model, os.path.join(args.save, 'weights.pt'))
  logging.info("total train: %f", t_record["train"])
  logging.info("total grow: %f", t_record["grow"])
  logging.info("total grow search: %f", t_record["grow_search"])




def grow(train_queue, valid_queue, model, architect, criterion, optimizer, lr, num_grow):
  print("grow start")
  model.train()
  grow_time = 0
  counter = 0
  for step, (input, target) in enumerate(train_queue):
    start = time.time()
    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    architect.grow_step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    optimizer.zero_grad()
    end = time.time()
    grow_time += (end-start)
    counter += 1
  logging.info("grow time per batch: %f ", grow_time/counter)
  search_s = time.time()
  architect.grow(num_grow)
  search_e = time.time()
  t_record["grow_search"] += (search_e-search_s)
  logging.info("grow search grad : %f", search_e-search_s)

  print("grow done")

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  train_time = 0
  counter = 0
  for step, (input, target) in enumerate(train_queue):
    start = time.time()
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()
    if epoch > args.grow_freq:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, darts = args.darts)
    #train_time
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    #for g in optimizer.param_groups:
    #    for p in g["params"]:
    #        if p.grad is None:
    #            print("here")
    #            continue
    #        #else:
    #        #    print(p.grad)
    #        #print(p.grad.device)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    #t_record["arch"]+=(train_start-arch_start)
    #t_record["train"]+=(end-train_start)
    #model.random_activate()

    #torch.save(t_record, "train_time.pth")
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    end = time.time()
    train_time += (end-start)
    counter += 1

  logging.info('train time per batch : %f', train_time/counter)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      #input = Variable(input, volatile=True).cuda()
      #target = Variable(target, volatile=True).cuda(async=True)
      input = input.cuda()
      target = target.cuda(non_blocking = True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

