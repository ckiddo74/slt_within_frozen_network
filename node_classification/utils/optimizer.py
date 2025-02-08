import torch.optim as optim


def get_optimizer(optimizer_name, lr, momentum, weight_decay, model):
    update_params = []
    print(f'Params to update:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}')
            update_params.append(param)

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(update_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(update_params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer