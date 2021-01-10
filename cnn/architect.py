import torch
import numpy as np
import torch.nn as nn
import time
import logging
from torch.autograd import Variable
from utils import mask_softmax

def _concat(xs, idx = None):
  if idx == None:
    return torch.cat([x.view(-1) for x in xs])
  else:
    return torch.cat([x.view(-1) for i, x in enumerate(xs) if i in idx])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer, darts, grow = False):
    loss = self.model._loss(input, target, grow)
    if darts:
      grads_all = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
      idx_use = None
      theta = _concat(self.model.parameters()).data
    else:
      grads_all = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
      idx_use = tuple(i for i in range(len(grads_all)) if grads_all[i] is not None )
      theta = _concat(self.model.parameters(),idx_use).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for i,v in enumerate(self.model.parameters()) if i in idx_use).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(grads_all, idx_use).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(moment+dtheta, alpha = eta),idx_use)
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, darts = False):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, darts)
    else:
        self._backward_step(input_valid, target_valid)
    #logging.info("normal_alphas grad: ")
    #logging.info(self.model.alphas_reduce.grad)
    #logging.info("reduce_alphas grad: ")
    #logging.info(self.model.alphas_reduce.grad)

    # eliminate unwanted grad noise
    self.model.alphas_normal.grad *= self.model.normal_indicator
    self.model.alphas_reduce.grad *= self.model.reduce_indicator
    #print([v.grad for v in self.optimizer.param_groups[0]["params"]])
    self.optimizer.step()

  def grow_step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, darts = False):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, darts, grow=True)
    else:
        self._backward_step(input_valid, target_valid, grow = True)
    #if not grow:
    # grow normal
    if not hasattr(self,"normal_grad") or self.normal_grad is None:
        self.normal_grad = self.model.alphas_normal.grad.clone()
    else:
        self.normal_grad+=self.model.alphas_normal.grad
    if not hasattr(self,"reduce_grad") or self.reduce_grad is None:
        self.reduce_grad = self.model.alphas_reduce.grad.clone()
    else:
        self.reduce_grad+=self.model.alphas_reduce.grad

  def grow(self):
    n_row = self.model.normal_indicator.size(0)
    n_col = self.model.normal_indicator.size(1)
    max_grad = 0
    normal_loc = None
    for i in range(n_row):
        for j in range(n_col):
            if self.model.normal_indicator[i,j]==0:
                cur_grad = self.normal_grad[i,j]
                if abs(cur_grad) > max_grad:
                    max_grad = cur_grad
                    normal_loc = (i,j)


    n_row = self.model.reduce_indicator.size(0)
    n_col = self.model.reduce_indicator.size(1)
    max_grad = 0
    reduce_loc = None
    for i in range(n_row):
        for j in range(n_col):
            if self.model.reduce_indicator[i,j]==0:
                cur_grad = self.reduce_grad[i,j]
                if abs(cur_grad) > max_grad:
                    max_grad = cur_grad
                    reduce_loc = (i,j)

    logging.info("normal_alphas grad: ")
    logging.info(self.normal_grad)
    logging.info("reduce_alphas grad: ")
    logging.info(self.reduce_grad)
    logging.info("activated normal_idx: ")
    logging.info(normal_loc)
    logging.info("activated reduce_idx: ")
    logging.info(reduce_loc)
    self.model.activate(normal_loc, reduce_loc)
    #logging.info(self.model.alphas_normal)
    #logging.info(self.model.alphas_reduce)
    self.normal_grad = None
    self.reduce_grad = None


  def _backward_step(self, input_valid, target_valid, grow = False):
    loss = self.model._loss(input_valid, target_valid, grow)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, darts, grow=False):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer, darts, grow)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid, grow)
    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    if darts:
      vector = [v.grad.data for v in unrolled_model.parameters()]
    else:
      vector = []
      for i,v in enumerate(unrolled_model.parameters()):
        if v.grad is not None:
          vector.append(v.grad.data)
        else:
          vector.append(None)

    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, darts=darts, grow=grow)
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(ig.data, alpha = eta)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = g.data
      else:
        v.grad.data.copy_(g.data)


  def _construct_model_from_theta(self, theta, idx):
    model_new = self.model.new()

    params, offset = {}, 0
    if idx is None:
      model_dict = self.model.state_dict()
      for i, (k, v) in enumerate(self.model.named_parameters()):
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length
      assert offset == len(theta)
      model_dict.update(params)
      model_new.load_state_dict(model_dict)
    else:
      for i, (k, v) in enumerate(self.model.named_parameters()):
        if i in idx:
          v_length = np.prod(v.size())
          params[k] = theta[offset: offset+v_length].view(v.size())
          offset += v_length
      assert offset == len(theta)
      #use self defined function to accelerate
      model_new.load_my_state_dict(params)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2, darts=False, grow = False):
    if darts:
      vector_concat = vector
    else:
      vector_concat = list(filter(lambda x: x is not None, vector))
    R = r / _concat(vector_concat).norm()
    for p, v in zip(self.model.parameters(), vector):
      if v is not None:
        p.data.add_(v, alpha = R)
    loss = self.model._loss(input, target, grow)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused = True)

    for p, v in zip(self.model.parameters(), vector):
      if v is not None:
        p.data.sub_(v, alpha = 2*R)
    loss = self.model._loss(input, target, grow)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused = True)

    for p, v in zip(self.model.parameters(), vector):
      if v is not None:
        p.data.add_(v, alpha = R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

