# based on https://github.com/TienjinHuang/UGTs-LoG

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.supermask_linear import SupermaskLinear
from layers.gin_linear import GINLinear

class GINBasedNet(nn.Module):
    def __init__(
            self, 
            num_classes, in_dim, hidden_dim, n_layers, 
            use_bias,
            norm_type, use_affine_bn, 
            dropout_p,
            graph, algo, sparsity, scale_method, width_factor
            ):
        super().__init__()
        self.dropout_p     = dropout_p
        self.n_layers      = n_layers
        self.edge_num      = graph.all_edges()[0].numel()
        init_eps           = 0
        learn_eps          = False  
        neighbor_aggr_type = 'mean' 
        
        self.norm_type = norm_type

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        if norm_type != None:
            self.norms = torch.nn.ModuleList()

        for layer in range(n_layers):
            if layer == n_layers - 1:
                out_dim = num_classes
            else:
                out_dim = int(hidden_dim * width_factor)
            self.ginlayers.append(
                GINLinear(
                    in_features=in_dim, out_features=out_dim,
                    aggr_type=neighbor_aggr_type, 
                    init_eps=init_eps, learn_eps=learn_eps,
                    sparsity=sparsity, algo=algo, scale_method=scale_method, bias=use_bias
                    )
                )
            if norm_type != None:
                if norm_type == 'bn':
                    self.norms.append(nn.BatchNorm1d(out_dim, affine=use_affine_bn))
                else:
                    raise NotImplementedError
            in_dim = out_dim

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score

        if algo == 'dense':
            self.linears_prediction = nn.Linear(int(hidden_dim * width_factor), num_classes, bias=use_bias)
        else:
            self.linears_prediction = SupermaskLinear(
                int(hidden_dim * width_factor), num_classes, sparsity, algo, scale_method, bias=use_bias)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

    def forward(self, g, edge_index, h):
        g.edata['mask'] = self.adj_mask2_fixed
        hidden_rep = []

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            if self.norm_type != None:
                h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout_p, training=self.training)
            hidden_rep.append(h)

        score_over_layer = (self.linears_prediction(hidden_rep[-2]) + hidden_rep[-1]) / 2

        return score_over_layer