import torch

def get_scale(local_sparsity, global_sparsity, scale_method):
    scale = torch.tensor([1], device=local_sparsity.device)
    if scale_method == 'static_scaled':
        scale = 1 / ((1 - global_sparsity)**(1/2))
    elif scale_method == 'dynamic_scaled':
        scale = 1 / ((1 - local_sparsity)**(1/2))
    return scale