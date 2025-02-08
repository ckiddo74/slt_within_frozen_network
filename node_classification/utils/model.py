import torch
import torch.nn as nn

import models.gin_based_net as gin_based_net
from utils.sparsity_distribution    import get_sparsity_distribution
from layers.supermask_linear        import SupermaskLinear
from layers.utils.set_initial_value import set_initial_value

def apply_ternary_frozen_mask_to_model(model, flatten_mask):
    count = 0
    for m in model.modules():
        if isinstance(m, SupermaskLinear):
            m.ternary_frozen_mask.data = flatten_mask[count:count + m.mask.numel()].reshape(m.mask.size())
            count += m.mask.numel()

def apply_score_to_model(model, score):
    count = 0
    for m in model.modules():
        if isinstance(m, SupermaskLinear):
            m.score.data = score[count:count + m.score.numel()].reshape(m.score.size())
            count += m.score.numel()

def get_model(model_name, dataset_name):
    gnn_models = {
        'gin_based_net': gin_based_net.GINBasedNet,
    }
    return gnn_models[model_name]

    
def initialize_params(
        model, w_init_method, s_init_method=None, m_init_method=None, 
        p_ratio=0, r_ratio=0, r_method = 'sparsity_distribution', mask_file=None,
        nonlinearity='relu'):
    if mask_file != None:
        print(f'Use {mask_file}.')
        mask_data = torch.load(mask_file)
        total = 0
        count = 0
        for n, m in model.named_modules():
            if isinstance(m, SupermaskLinear):
                total += 1
                if not n + '.weight' in mask_data:
                    n = 'module.' + n
                if mask_data[n + '.weight'].numel() == m.weight.numel():
                    count += 1
        if total == count:
            assert m_init_method != None
            frozen_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - (p_ratio+r_ratio))
            is_same_size = True
            frozen_dist_count = 0
        elif count == 0:  
            is_same_size = False
        else:
            raise ValueError
        
        p_dist = []
        r_dist = []
        for n, m in model.named_modules():
            if isinstance(m, SupermaskLinear):
                if is_same_size:
                    if not n + '.weight' in mask_data:
                        n = 'module.' + n
                    mask_ones_rate  = ((mask_data[n + '.mask'] == 1).sum()/m.weight.numel()).to(frozen_dist[0].device)
                    mask_zeros_rate = 1 - mask_ones_rate
                    print(f'{mask_zeros_rate=}, {mask_ones_rate=}')
                    free_rate = (1 - frozen_dist[frozen_dist_count])
                    free_ones_rate  = free_rate / 2
                    free_zeros_rate = free_rate / 2
                    print(f'0: {free_zeros_rate=}, {free_ones_rate=}')
                    if free_ones_rate > mask_ones_rate:
                        free_zeros_rate += free_ones_rate - mask_ones_rate
                        free_ones_rate = mask_ones_rate
                        print(f'1: {free_zeros_rate=}, {free_ones_rate=}')
                    if free_zeros_rate > mask_zeros_rate:
                        free_ones_rate += free_zeros_rate - mask_zeros_rate
                        free_zeros_rate = mask_zeros_rate
                        print(f'2: {free_zeros_rate=}, {free_ones_rate=}')
                    p_dist.append(mask_zeros_rate - free_zeros_rate)
                    r_dist.append(mask_ones_rate - free_ones_rate)
                    print(f'{p_dist[-1]=}, {p_dist[-1]=}')
                    eps = 1e-4
                    assert abs(free_ones_rate + free_zeros_rate - free_rate) < eps,          f'{free_ones_rate} + {free_zeros_rate} != {free_rate}.'
                    assert (p_dist[-1] + r_dist[-1] - frozen_dist[frozen_dist_count]) < eps, f'{p_dist[-1]} + {r_dist[-1]} != {frozen_dist[frozen_dist_count]}.'
                    frozen_dist_count += 1
                else:
                    raise NotImplementedError
        p_dist = torch.tensor(p_dist)
        r_dist = torch.tensor(r_dist)
    elif m_init_method != None:
        assert p_ratio + r_ratio <= 1
        if r_method == 'density_distribution':
            print(f'{r_method=}')
            p_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio)
            r_dist = 1 - get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=r_ratio)
        elif r_method == 'sparsity_distribution':
            print(f'{r_method=}')
            frozen_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - (p_ratio+r_ratio))
            p_dist      = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio)
            r_dist      = frozen_dist - p_dist
            assert torch.logical_and(r_dist >= 0, r_dist <= 1).all()
        else: 
            raise ValueError
        
    module_count = 0

    for m in model.modules():
        if isinstance(m, SupermaskLinear):
            print('initializing SupermaskLinear')
            set_initial_value(m.weight, init_method=w_init_method, nonlinearity=nonlinearity)
            set_initial_value(m.score,  init_method=s_init_method, nonlinearity=nonlinearity)
            if m.bias != None:
                raise NotImplementedError
            
            if m_init_method != None:
                assert p_dist[module_count] + r_dist[module_count] <= 1
                rand_indices = torch.randperm(m.weight.numel(), device=m.weight.device)
                # n_frozen_params = (m.weight.numel() * p_dist[module_count]).int()
                # n_p_params = (n_frozen_params * p_ratio).int()
                # n_r_params = n_frozen_params - n_p_params
                n_p_params   = (m.weight.numel() * p_dist[module_count]).int()
                n_r_params   = (m.weight.numel() * r_dist[module_count]).int()

                ternary_frozen_mask = m.ternary_frozen_mask.flatten()
                ternary_frozen_mask[rand_indices[:n_p_params]] = -1
                ternary_frozen_mask[rand_indices[n_p_params:n_p_params+n_r_params]] = 1
                m.ternary_frozen_mask = ternary_frozen_mask.reshape(m.ternary_frozen_mask.size()).clone().detach()
                module_count += 1

        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            print('initializing nn.Linear/nn.Conv2d')
            set_initial_value(m.weight, init_method=w_init_method, nonlinearity=nonlinearity)
            if m.bias != None:
                set_initial_value(m.bias, init_method=w_init_method, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            print('initializing nn.BatchNorm2d')
            if m.weight:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            print('initializing BatchNorm1d')
            if m.weight:
                raise NotImplementedError