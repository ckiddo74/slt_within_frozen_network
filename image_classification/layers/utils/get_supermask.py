from torch.autograd import Function
import torch

class GetSupermask(Function):
    @staticmethod
    def forward(ctx, score, kthvalue, ternary_frozen_mask):
        ones   = torch.ones_like(score).to(score.device)
        zeros  =  0 * ones
        m_ones = -1 * ones

        mask  = torch.gt(score, kthvalue).to(score.device)

        mask  = torch.where(
            ternary_frozen_mask == m_ones, zeros, mask)
        mask  = torch.where(
            ternary_frozen_mask == ones, ones, mask)
        
        return mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None
    
def get_supermask(mask, ternary_frozen_mask):
    ones   = torch.ones_like(mask).to(mask.device)
    zeros  =  0 * ones
    m_ones = -1 * ones

    mask  = torch.where(
        ternary_frozen_mask == m_ones, zeros, mask)
    mask  = torch.where(
        ternary_frozen_mask == ones,  ones,  mask)
    
    return mask