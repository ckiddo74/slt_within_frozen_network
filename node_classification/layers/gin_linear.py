import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from layers.supermask_linear import SupermaskLinear

class GINLinear(nn.Module):
    def __init__(
            self, in_features, out_features, 
            aggr_type, init_eps=0, learn_eps=False,
            sparsity=None, algo='local_ep', scale_method=None, bias=False, device=None, dtype=None
            ):
        super().__init__()
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError(f'Aggregator type {aggr_type} not recognized.')
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        self.msg_mask = fn.src_mul_edge('h', 'mask', 'm')

        if algo == 'dense':
            self.linear = nn.Linear(
                in_features=in_features, out_features=out_features, bias=bias)
        else:
            self.linear = SupermaskLinear(
                in_features=in_features, out_features=out_features, 
                sparsity=sparsity, algo=algo, scale_method=scale_method, bias=bias)

    def forward(self, g, h):
        g = g.local_var()
        g.ndata['h'] = h
        ### pruning edges by cutting message passing process
        g.update_all(self.msg_mask, self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']     
        return self.linear(h) 