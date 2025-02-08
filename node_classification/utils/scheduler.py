import torch.optim as optim

def get_scheduler(scheduler_name, optimizer, milestones, gamma, max_epoch):
    if scheduler_name == 'multi_step_lr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == 'cosine_lr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=max_epoch
        )
    else:
        raise NotImplementedError
    
    return scheduler