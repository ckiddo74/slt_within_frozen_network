import torch
from tqdm.auto import tqdm
from layers.utils.set_kthvalue import set_kthvalue

def train(
        model, device, train_loader, optimizer, criterion, algo,
        model_ema, mixup_fn, use_non_blocking,
        use_amp, loss_scaler, max_norm=0):
    non_blocking = True if use_non_blocking else False
    model.train()
    for _, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = images.to(device=device, non_blocking=non_blocking), labels.to(device=device, non_blocking=non_blocking)        
        optimizer.zero_grad()
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        set_kthvalue(model, algo, device)

def evaluate(
        model, device, eval_loader, 
        use_non_blocking, use_amp, only_update_mask=False):
    non_blocking = True if use_non_blocking else False
    model.eval()
    with torch.no_grad():
        acc_count = torch.tensor([0], device=device)
        total     = torch.tensor([0], device=device)
        for _, (images, labels) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            images, labels = images.to(device=device, non_blocking=non_blocking), labels.to(device=device, non_blocking=non_blocking)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            acc_count += (predicted == labels).sum().item()
            if only_update_mask:
                break
    return acc_count, total