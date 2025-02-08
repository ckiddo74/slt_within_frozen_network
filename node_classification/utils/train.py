import torch

from layers.utils.set_kthvalue import set_kthvalue

def train(model, g, edge_index, features, labels, idx_train, optimizer, criterion, algo, device):
    model.train()
    optimizer.zero_grad()
    outputs = model(g, edge_index, features)
    loss = criterion(outputs[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    set_kthvalue(model, algo, device)

def evaluate(model, g, edge_index, features, labels, idx_eval, device):
    model.eval()
    with torch.no_grad():
        acc_count = torch.tensor([0], device=device)
        total     = torch.tensor([0], device=device)

        outputs = model(g, edge_index, features)
        _, predicted = torch.max(outputs.data, 1)
        acc_count += (predicted[idx_eval] == labels[idx_eval]).sum().item()
        total     += predicted[idx_eval].numel()
    return acc_count, total