import copy

import numpy as np
import torch
from layers.supermask_conv          import SupermaskConv2d
from layers.supermask_linear        import SupermaskLinear


def get_sparsity_distribution(
    model, loss=None, dataloader=None, s_dist_method=None, density=0):

    masked_parameters = []
    for m in filter(lambda x: isinstance(x, (SupermaskLinear, SupermaskConv2d)), model.modules()):
        # masked_parameters.append(m.ternary_frozen_mask)
        masked_parameters.append(m.weight)

    print(f'Sparsity distribution : {s_dist_method}')
    if s_dist_method == 'snip':
        raise NotImplementedError
        edges = get_snip(
            model=model, loss=loss, dataloader=dataloader, 
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'grasp':
        raise NotImplementedError
        edges = get_grasp(
            model=model, loss=loss, dataloader=dataloader, 
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'synflow':
        raise NotImplementedError
        edges = get_synflow(
            model=model, loss=loss, dataloader=dataloader, 
            masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'erk':
        sparsities, edges = get_erk(masked_parameters=masked_parameters, density=density)
    elif s_dist_method == 'igq':
        sparsities, edges = get_igq(masked_parameters=masked_parameters, density=density) 
    elif s_dist_method == 'epl':
        sparsities, edges = get_epl(masked_parameters=masked_parameters, density=density) 
    else:
        raise ValueError
    
    return sparsities



# def get_snip(model, loss, dataloader, masked_parameters, density):
#     for _, p in masked_parameters:
#         device = p.device
#         break
#     pruner = SNIP(masked_parameters=[])
#     pruner.masked_parameters = masked_parameters
#     pruner.score(
#             model=model, loss=loss, dataloader=dataloader, device=device)
#     pruner.mask(density, 'global')
#     edges = []
#     for m, p in pruner.masked_parameters:
#         edges.append(m.sum())
#     return torch.tensor(edges)


# def get_grasp(model, loss, dataloader, masked_parameters, density):
#     for _, p in masked_parameters:
#         device = p.device
#         break
#     pruner = GraSP(masked_parameters=[])
#     pruner.masked_parameters = masked_parameters
#     pruner.score(
#             model=model, loss=loss, dataloader=dataloader, device=device)
#     pruner.mask(density, 'global')
#     edges = []
#     for m, p in pruner.masked_parameters:
#         edges.append(m.sum())
#     return torch.tensor(edges)

# def get_synflow(model, loss, dataloader, masked_parameters, density):
#     # epoch 100, pruning_schedule exponential
#     for _, p in masked_parameters:
#         device = p.device
#         break
#     pruner = SynFlow(masked_parameters=[])
#     pruner.masked_parameters = masked_parameters
#     epochs = 100
#     for epoch in tqdm(range(epochs)):
#         pruner.score(
#                 model=model, loss=loss, dataloader=dataloader, device=device)
#         dense = density**((epoch + 1) / epochs)
#         pruner.mask(dense, 'global')
#     edges = []
#     for m, p in pruner.masked_parameters:
#         edges.append(m.sum())
#     return torch.tensor(edges)

def get_erk(masked_parameters, density):
    # We have to enforce custom sparsities and then find the correct scaling
    # factor.
    is_eps_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_eps_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
        divisor = 0
        rhs     = 0
        raw_probabilities = {}
        print(f'Loop : ')
        for p in masked_parameters:
            n_param = p.numel()
            n_zeros = int(p.numel() * (1 - density))
            if id(p) in dense_layers:
                print(f'{id(p)} is a dense layer')
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                n_ones = n_param - n_zeros
                rhs += n_ones
                assert id(p) not in raw_probabilities 
                print(f'{id(p)} raw_prob : {(sum(p.size()) / p.numel())}')
                raw_probabilities[id(p)] = (sum(p.size()) / p.numel())
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[id(p)] * n_param
        print()
        # All layer is dense
        if divisor == 0:
            is_eps_valid = True
            break
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        eps = rhs / divisor
        print(f'eps : {rhs} / {divisor} = {eps}')
        # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob     = np.max(list(raw_probabilities.values()))
        print(f'raw_prob_list : {list(raw_probabilities.values())}')
        print(f'max_prob : {max_prob}')
        max_prob_one = max_prob * eps
        print(f'max_prob_one : {max_prob_one}')
        if max_prob_one >= 1:
            is_eps_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f'Sparsity of layer {mask_name} had to be set to 0')
                    dense_layers.add(mask_name)
            print()
        else:
            is_eps_valid = True

    sparsities = []
    edges      = []
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for p in masked_parameters:
        n_param = p.numel()
        if id(p) in dense_layers:
            sparsities.append(0.)
            edges.append(p.numel())
        else:
            probability_one = eps * raw_probabilities[id(p)]
            sparsities.append(1. - probability_one)
            print(f'{p.numel()} * {probability_one} = {p.numel() * probability_one}')
            edges.append(int(p.numel() * probability_one))

    return torch.tensor(sparsities), torch.tensor(edges)


def get_epl(masked_parameters, density):
    layers = set(range(len(masked_parameters)))
    n_params_lst = []
    edges = []
    for p in masked_parameters:
        n_params_lst.append(p.numel())
        edges.append(0)
    total = sum(n_params_lst) * density
    dense_layers = set()

    while total != 0:
        for k in layers:
            edges[k] += total / len(layers)

        total = 0
        for k in layers:
            if edges[k] > n_params_lst[k]:
                total += edges[k] - n_params_lst[k]
                edges[k] = n_params_lst[k]
                dense_layers.add(k)
        layers = layers - dense_layers

    sparsities = []
    for i in range(len(edges)):
        edges[i] = int(edges[i])
        density = edges[i] / n_params_lst[i]
        sparsities.append(1 - density)

    return torch.tensor(sparsities), torch.tensor(edges)


def get_igq(masked_parameters, density):
    def bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high):
        lengths_low          = [Length / (f_low / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_low = 1 - sum(lengths_low) / sum(Lengths)
        if abs(overall_sparsity_low - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_low, Lengths)]
        
        lengths_high          = [Length / (f_high / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_high = 1 - sum(lengths_high) / sum(Lengths)
        if abs(overall_sparsity_high - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_high, Lengths)]
            
        force            = float(f_low + f_high) / 2
        lengths          = [Length / (force / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity = 1 - sum(lengths) / sum(Lengths)
        f_low            = force if overall_sparsity < target_sparsity else f_low
        f_high           = force if overall_sparsity > target_sparsity else f_high
        return bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high)
    
    edges  = []
    counts = []
    for p in masked_parameters:
        counts.append(p.numel())
    tolerance = 1./sum(counts)
    areas     = [1./count for count in counts]
    sparsities = bs_force_igq(
        areas=areas, Lengths=counts, target_sparsity=1-density, 
        tolerance=tolerance, f_low=0, f_high=1e20)
    for i, p in enumerate(masked_parameters):
        edges.append(int(p.numel() * (1 - sparsities[i])))

    return torch.tensor(sparsities), torch.tensor(edges)
