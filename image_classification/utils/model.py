import torch
import torch.nn as nn
import torch.distributed as dist

import models.resnet_cifar    as resnet_cifar
import models.convs_cifar     as convs_cifar
import models.resnet_imagenet as resnet_imagenet
from utils.sparsity_distribution    import get_sparsity_distribution
from layers.supermask_conv          import SupermaskConv2d
from layers.supermask_linear        import SupermaskLinear
from layers.utils.set_initial_value import set_initial_value

def apply_ternary_frozen_mask_to_model(model, flatten_mask):
    count = 0
    for m in model.modules():
        if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
            m.ternary_frozen_mask.data = flatten_mask[count:count + m.mask.numel()].reshape(m.mask.size())
            count += m.mask.numel()

def apply_score_to_model(model, score):
    count = 0
    for m in model.modules():
        if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
            m.score.data = score[count:count + m.score.numel()].reshape(m.score.size())
            count += m.score.numel()

def get_model(model_name, dataset_name):
    cifar_models = {
        'conv2':    convs_cifar.conv2,
        'conv4':    convs_cifar.conv4,
        'conv6':    convs_cifar.conv6,
        'conv8':    convs_cifar.conv8,
        'resnet18': resnet_cifar.resnet18,
        'resnet34': resnet_cifar.resnet34
    }

    imagenet_models = {
        'resnet18': resnet_imagenet.resnet18,
        'resnet34': resnet_imagenet.resnet34,
        'resnet50': resnet_imagenet.resnet50,
        'resnet101': resnet_imagenet.resnet101,
        'wide_resnet50_2': resnet_imagenet.wide_resnet50_2
    }

    if dataset_name in ['cifar10', 'cifar100']:
        return cifar_models[model_name]
    elif dataset_name == 'imagenet':
        return imagenet_models[model_name]
    else:
        raise NotImplementedError
    
def initialize_params(
        model, w_init_method, s_init_method=None, m_init_method=None, 
        p_ratio=0, r_ratio=0, r_method = 'sparsity_distribution', mask_file=None,
        nonlinearity='relu', algo=None):
    if mask_file != None:
        print(f'Use {mask_file}.')
        mask_data = torch.load(mask_file)
        total = 0
        count = 0
        for n, m in model.named_modules():
            if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
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
            if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
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
        if r_method == None or r_method == 'density_distribution':
            print('r_method: density_distribution')
            p_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio)
            r_dist = 1 - get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=r_ratio)
        elif r_method == 'sparsity_distribution':
            print('r_method: sparsity_distribution')
            frozen_dist = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - (p_ratio+r_ratio))
            p_dist      = get_sparsity_distribution(model=model, s_dist_method=m_init_method, density=1 - p_ratio)
            r_dist      = frozen_dist - p_dist
            assert torch.logical_and(r_dist >= 0, r_dist <= 1).all()
        else: 
            raise ValueError

    module_count = 0

    for m in model.modules():
        if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
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
            set_initial_value(m.weight, init_method=w_init_method, nonlinearity=nonlinearity)
            if m.bias != None:
                set_initial_value(m.bias, init_method=w_init_method, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight != None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)