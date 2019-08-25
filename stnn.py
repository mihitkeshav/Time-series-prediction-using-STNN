import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from module import MLP
from utils import identity


class SaptioTemporalNN(nn.Module):
    def __init__(self, relations, nx, nt, nd, nz, mode=None, nhid=1, nlayers=2, dropout_f=0., dropout_d=0.,
                 activation='identity', periode=1):
        super(SaptioTemporalNN, self).__init__()
        assert (nhid > 0 and nlayers > 1) or (nhid == 0 and nlayers == 1)
        # attributes
        self.nt = nt
        self.nx = nx
        self.nz = nz
        self.mode = mode
        # kernel
        self.activation = identity if activation == 'identity' else None
        device = relations.device
        self.relations = torch.cat((torch.eye(nx).to(device).unsqueeze(1), relations), 1)
    
        self.nr = self.relations.size(1)
        # modules
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.Tensor(nt, nx, nz))
        self.dynamic = MLP(nz * self.nr, nhid, nz, nlayers, dropout_d)
        self.decoder = nn.Linear(nz, nd, bias=False)
        # init
        self._init_weights(periode)

    def _init_weights(self, periode):
        initrange = 0.1
        if periode >= self.nt:
            self.factors.data.uniform_(-initrange, initrange)
        else:
            timesteps = torch.arange(self.factors.size(0)).long()
            for t in range(periode):
                idx = timesteps % periode == t
                idx_data = idx.view(-1, 1, 1).expand_as(self.factors)
                init = torch.Tensor(self.nx, self.nz).uniform_(-initrange, initrange).repeat(idx.sum().item(), 1, 1)
            self.factors.data.masked_scatter_(idx_data, init.view(-1))
     

    def get_relations(self):
        if self.mode is None:
            return self.relations
        else:
            weights = F.hardtanh(self.rel_weights, 0, 1)
            return torch.cat((intra, inter), 1)

    def update_z(self, z):
        z_context = self.get_relations().matmul(z).view(-1, self.nr * self.nz)
        z_next = self.dynamic(z_context)
        return self.activation(z_next)

    def decode_z(self, z):
        x_rec = self.decoder(z)
        return x_rec

    def dec_closure(self, t_idx, x_idx):
        z_inf = self.factors[t_idx, x_idx]
        x_rec = self.decoder(z_inf)
        return x_rec

    def dyn_closure(self, t_idx, x_idx):
        rels = self.get_relations()
        z_input = self.factors[t_idx]
        z_context = rels[x_idx].matmul(z_input).view(-1, self.nr * self.nz)
        z_gen = self.dynamic(z_context)
        return self.activation(z_gen)

    def generate(self, nsteps):
        z = self.factors[-1]
        z_gen = []
        for t in range(nsteps):
            z = self.update_z(z)
            z_gen.append(z)
        z_gen = torch.stack(z_gen)
        x_gen = self.decode_z(z_gen)
        return x_gen, z_gen

    def factors_parameters(self):
        yield self.factors

    def rel_parameters(self):
        assert self.mode is not None
        yield self.rel_weights
