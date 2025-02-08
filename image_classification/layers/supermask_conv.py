import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.utils.get_supermask import GetSupermask, get_supermask
from layers.utils.get_scale import get_scale

class SupermaskConv2d(nn.Conv2d):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernel_size, 
            sparsity,
            algo           = 'local_ep',
            scale_method   = None,
            stride         = 1,
            padding        = 0,
            dilation       = 1,
            groups         = 1,
            bias           = False,
            padding_mode   = 'zeros',
            device         = None,
            dtype          = None
            ):
        super(SupermaskConv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.algo         = algo
        self.scale_method = scale_method
        self.score = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('mask',                torch.Tensor(self.weight.size()))
        self.register_buffer('ternary_frozen_mask', torch.zeros_like(self.weight))
        self.register_buffer('local_sparsity',      torch.Tensor([sparsity]))
        self.register_buffer('global_sparsity',     torch.Tensor([sparsity]))
        self.register_buffer('scale',               torch.Tensor([1]))
        self.register_buffer('kthvalue',            torch.Tensor([0]))

        self.weight.requires_grad = False

    def forward(self, input):
        mask = GetSupermask.apply(
                self.score.abs(), self.kthvalue.data[0], self.ternary_frozen_mask)
        self.mask = mask
        with torch.no_grad():
            self.local_sparsity = 1 - (mask.sum() / mask.numel())
            self.scale = get_scale(self.local_sparsity, self.global_sparsity, self.scale_method)
        masked_weight = mask * self.weight * self.scale
        return F.conv2d(input, masked_weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)