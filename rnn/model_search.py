import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel
import time

record={"ch":0, "unweight":0, "sum":0}

class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    probs = F.softmax(self.weights, dim=-1)
    #print(probs)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      start_ch = time.time()
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      end_ch = time.time()
      record["ch"]+=(end_ch-start_ch)

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unw_s = time.time()
        unweighted = torch.zeros_like(states)
        sum_cur = 0
        for j,s in enumerate(states):
            if not probs[offset+j,k]==0:
                #print(fn(h[j]).shape)
                #print(states[j].shape)
                #print(c.shape)
                #print(unweighted[j].shape)
                unweighted[j] = states[j] + c[0]*(fn(h[j])-states[j])

        #unweighted = states + c * (fn(h) - states)
        unw_e = time.time()
        #print("cur connect ", i)
        #print(probs[offset:offset+i+1, k])
        print(unweighted.shape)
        s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
        print(s.shape)
        sum_e = time.time()
        record["unweight"]+=(unw_e-unw_s)
        record["sum"]+=(sum_e-unw_e)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    torch.save(record,"f_record.pt")
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self, darts):
      k = sum(i for i in range(1, STEPS+1))
      num_ops = len(PRIMITIVES)
      if darts:
        self.weights = torch.randn(k, num_ops).mul_(1e-3)
      else:
        self.indicator = torch.zeros([k, num_ops]).cuda()
        for i in range(k):
          idx = np.random.choice(num_ops-1, size = 1, replace = False)
          self.indicator[i, idx[0]]=1
          self.weights = torch.randn(k, num_ops).mul_(1e-3)*self.indicator
      self.weights.requires_grad_()


      #self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self._arch_parameters = [self.weights]
      for rnn in self.rnns:
        rnn.weights = self.weights

    def arch_parameters(self):
      return self._arch_parameters

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def load_my_state_dict(self, state_dict):
      own_state = self.state_dict()
      for name, param in state_dict.items():
        if name not in own_state:
          continue
        if isinstance(param, Parameter):
          param = param.data
        own_state[name].copy_(param)





    def genotype(self):

      def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype

