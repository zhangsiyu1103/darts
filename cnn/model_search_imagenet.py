import torch
import random
import math
import copy
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_G,PRIMITIVES_D
from genotypes import Genotype
from utils import indicator, mask_softmax

t_record={}
t_record["forward"] = 0
t_record["forward_soft_max"] = 0

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride, darts, pc):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    self.k = 2
    self.C = C
    self.stride = stride
    self.darts = darts
    self.pc = pc
    if self.darts:
      PRIMITIVES = PRIMITIVES_D
      OPS = OPS_D
    else:
      PRIMITIVES = PRIMITIVES_G
      OPS = OPS_G

    for primitive in PRIMITIVES:
      if pc:
        op = OPS[primitive](C//self.k, stride, False)
      else:
        op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        if pc:
          op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
        else:
          op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
      #self._ops.append(None)

  def forward(self, x, weights, grow = False):
    #s = time.time()
    if self.pc:
      dim_2 = x.shape[1]
      xtemp = x[ : , :  dim_2//self.k, :, :]
      xtemp2 = x[ : ,  dim_2//self.k:, :, :]
      x = xtemp
    if self.darts:
      ret =  sum(w * op(x) for w, op in zip(weights, self._ops))
    else:
      PRIMITIVES = PRIMITIVES_G
      OPS = OPS_G
      ret = 0
      for i in range(len(weights)):
        if weights[i] == 0 and not grow:
          self._ops[i] = self._ops[i].cpu()
          continue
        self._ops[i] = self._ops[i].to(x.device)
        ret += weights[i]*self._ops[i](x)
    #for i in range(len(weights)):
    #  self._ops[i] = self._ops[i].to(x.device)
    if self.pc:
      if ret.shape[2] == xtemp2.shape[2]:
        ret = torch.cat([ret, xtemp2], dim=1)
      else:
        ret = torch.cat([ret, self.mp(xtemp2)], dim=1)
      ret = channel_shuffle(ret, self.k)
    #e = time.time()
    return ret

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, darts, pc):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.darts = darts
    self.pc = pc
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
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, darts, pc)
        self._ops.append(op)

  def forward(self, s0, s1, weights, grow = False, weights2 = None):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      start = time.time()
      if weights2 is not None:
        s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j], grow) for j, h in enumerate(states))
      else:
        s = sum(self._ops[offset+j](h, weights[offset+j], grow) for j, h in enumerate(states))
      end = time.time()
      t_record["forward"] += (end-start)
      offset += len(states)
      states.append(s)
    #output = []

    # append zero if no output
    #for x in states[-self._multiplier:]:
    #  if torch.is_tensor(x):
    #    zero = torch.zeros_like(x)
    #for x in states[-self._multiplier:]:
    #  if torch.is_tensor(x):
    #    output.append(x)
    #  else:
    #    output.append(zero)
    return torch.cat(states[-self._multiplier:], dim=1)
    #return torch.cat(output, dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, darts = False, pc =False):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.darts = darts
    self.pc = pc

    C_curr = stem_multiplier*C


    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )
    self.stem1 = nn.Sequential(
      nn.ReLU(inplace = True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )




    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, darts, pc)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    #model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    #for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
    #    with torch.no_grad():
    #        x.set_(y.data)
    #model_new.normal_indicaor = self.normal_indicator
    #model_new.reduce_indicaor = self.reduce_indicator
    model_new = copy.deepcopy(self)
    return model_new

  def forward(self, input, grow = False):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        if self.darts:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = mask_softmax(self.alphas_reduce, self.reduce_indicator, dim=-1)
        if self.pc:
          n = 3
          start = 2
          weights2 = F.softmax(self.beta_reduce[0:2], dim = -1)
          for j in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.beta_reduce[start:end], dim = -1)
            start = end
            n += 1
            weights2 = torch.cat([weights2, tw2], dim = -1)
      else:
        if self.darts:
          weights = F.softmax(self.alphas_normal, dim=-1)
        else:
          weights = mask_softmax(self.alphas_normal, self.normal_indicator, dim=-1)
        if self.pc:
          n = 3
          start = 2
          weights2 = F.softmax(self.beta_normal[0:2], dim = -1)
          for j in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.beta_normal[start:end], dim = -1)
            start = end
            n += 1
            weights2 = torch.cat([weights2, tw2], dim = -1)
      if self.pc:
        s0, s1 = s1, cell(s0, s1, weights, grow, weights2 = weights2)
      else:
        s0, s1 = s1, cell(s0, s1, weights, grow)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    torch.save(t_record, "critical.pth")
    return logits

  def _loss(self, input, target, grow=False):
    logits = self(input, grow)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
  # initialize two connection per row for weight update to be useful
    #num of connection
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    #num of operation per connection
    if self.darts:
      PRIMITIVES = PRIMITIVES_D
    else:
      PRIMITIVES = PRIMITIVES_G
    num_ops = len(PRIMITIVES)
    all_ops = k * num_ops
    if self.darts:
      self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
      self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    else:
      #self.active = all_ops
      self.active = k
      self.normal_indicator = torch.zeros([k, num_ops]).cuda()
      #self.normal_idx = []
      for i in range(k):
          idxs = np.random.choice(num_ops, size = 2, replace = False)
          #self.normal_idx.append([i,idx])
          self.normal_indicator[i,idxs[0]]=1
          self.normal_indicator[i,idxs[1]]=1
      self.reduce_indicator = torch.zeros([k, num_ops]).cuda()
      #self.reduce_idx = []
      for i in range(k):
          idxs = np.random.choice(num_ops, size = 2, replace = False)
          #self.normal_idx.append([i,idx])
          self.reduce_indicator[i,idxs[0]]=1
          self.reduce_indicator[i,idxs[1]]=1


      self.alphas_normal = 1e-3*torch.randn(k, num_ops).cuda()*self.normal_indicator.cuda()
      self.alphas_reduce = 1e-3*torch.randn(k, num_ops).cuda()*self.reduce_indicator.cuda()

      self.alphas_normal.requires_grad_()
      self.alphas_reduce.requires_grad_()
    if self.pc:
      self.beta_normal = 1e-3*torch.randn(k).cuda()
      self.beta_reduce = 1e-3*torch.randn(k).cuda()
      self.beta_normal.requires_grad_()
      self.beta_reduce.requires_grad_()
      self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
        self.beta_normal,
        self.beta_reduce,
      ]
    else:
      self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
      ]


  # random activation, not in use
  def random_activate(self):
    n_row = self.normal_indicator.size(0)
    n_col = self.normal_indicator.size(1)

    r_normal_idx = random.randint(0, (n_row - 1))
    c_normal_idx = random.randint(0, n_col - 1)

    #new_reduce = np.random.choice(n_row*n_col, size = 1, replace = False)
    r_reduce_idx = random.randint(0, (n_row - 1))
    c_reduce_idx = random.randint(0, n_col - 1)

     #make sure we don't activate active ops
    if self.reduce_indicator[r_reduce_idx, c_reduce_idx]==1 or self.normal_indicator[r_normal_idx, c_normal_idx]==1:
        self.random_activate()
        return
    with torch.no_grad():
        self.normal_indicator[r_normal_idx,c_normal_idx] = 1
        self.alphas_normal[r_normal_idx,c_normal_idx] = math.log(0.001)
        self.reduce_indicator[r_reduce_idx,c_reduce_idx] = 1
        self.alphas_reduce[r_reduce_idx,c_reduce_idx] = math.log(0.001)
    self.active+=1

  def activate(self, normal_idx, reduce_idx):
    with torch.no_grad():
      if normal_idx is not None:
        for idx in normal_idx:
          self.normal_indicator[idx] = 1
          self.alphas_normal[idx] = math.log(0.001)
      if reduce_idx is not None:
        for idx in reduce_idx:
          self.reduce_indicator[idx] = 1
          self.alphas_reduce[idx] = math.log(0.001)


  def deactivate(self):
    with torch.no_grad():
      n_row = self.normal_indicator.size(0)
      n_col = self.normal_indicator.size(1)
      real_weight = mask_softmax(self.alphas_normal, self.normal_indicator, dim = -1)
      for i in range(n_row):
        for j in range(n_col):
          if real_weight[i,j] <= 0.001:
            self.normal_indicator[i,j] = 0
            self.alphas_normal[i,j] = 0
      n_row = self.reduce_indicator.size(0)
      n_col = self.reduce_indicator.size(1)
      real_weight = mask_softmax(self.alphas_reduce, self.reduce_indicator, dim = -1)
      for i in range(n_row):
        for j in range(n_col):
          if real_weight[i,j] <= 0.001:
            self.reduce_indicator[i,j] = 0
            self.alphas_reduce[i,j] = 0



  def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        continue
      if isinstance(param, Parameter):
        param = param.data
      own_state[name].copy_(param)


  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    if self.darts:
      PRIMITIVES = PRIMITIVES_D
      OPS = OPS_D
    else:
      PRIMITIVES = PRIMITIVES_G
      OPS = OPS_G

    def _parse(weights, weights2 = None):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        if self.pc:
          W2 = weights2[start:end].copy()
          for j in range(n):
            W[j, :] = W[i, :] * W2[j]
        if self.darts:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        else:
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) ))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if (self.darts and k != PRIMITIVES.index('none')) or not self.darts:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    if self.pc:
      n = 3
      start = 2
      weightsn2 = F.softmax(self.beta_normal[0:2], dim = -1)
      weightsr2 = F.softmax(self.beta_reduce[0:2], dim = -1)
      for j in range(self._steps-1):
        end = start + n
        tw2 = F.softmax(self.beta_reduce[start:end], dim = -1)
        tn2 = F.softmax(self.beta_normal[start:end], dim = -1)
        start = end
        n += 1
        weightsr2 = torch.cat([weightsr2, tw2], dim = -1)
        weightsn2 = torch.cat([weightsn2, tn2], dim = -1)

    alphas_normal = torch.where(self.alphas_normal == 0,torch.FloatTensor([float("-inf")]).cuda(),self.alphas_normal)
    alphas_reduce = torch.where(self.alphas_reduce == 0, torch.FloatTensor([float("-inf")]).cuda(), self.alphas_reduce)
    if self.pc:
      gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
      gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())
    else:
      gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
      gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

