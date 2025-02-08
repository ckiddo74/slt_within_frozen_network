# based on https://leimao.github.io/blog/PyTorch-Distributed-Training/
import os
import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
import dgl
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_sparse.tensor import from_scipy
from tqdm.auto             import tqdm

from utils.seed      import set_random_seeds
from utils.parser    import get_parser
from utils.dataset   import get_datasets, get_dataset_dim
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.train     import train, evaluate
from utils.model     import (
    get_model, 
    initialize_params, 
    apply_score_to_model, 
    apply_ternary_frozen_mask_to_model
    )
from layers.utils.set_kthvalue      import set_kthvalue
from layers.supermask_linear        import SupermaskLinear
from layers.utils.set_initial_value import set_initial_value



def main():
    parser = get_parser()
    args   = parser.parse_args()

    device   = torch.device(f'cuda:{0}')
    save_dir = pathlib.Path(args.save_dir)
    set_random_seeds(seed=args.seed)

    num_classes = get_dataset_dim(args.dataset_name) 

    # data, split_idx = get_datasets(
    g, features, edge_index, labels, idx_train, idx_valid, idx_test = get_datasets(
        model_name      = args.model_name,
        dataset_name    = args.dataset_name, 
        dataset_dir     = args.dataset_dir, 
        sampling        = args.sampling,
        samplingtype    = args.samplingtype
        )
    if g != None and edge_index != None:
        print(f'This is OGBN-ArXiv exp')
        g = g.to(device)
        edge_index = edge_index.to(device)
    elif g != None:
        print(f'This is DGL model exp')
        g = g.to(device)
    elif edge_index != None:
        print(f'This is PyG model exp')
        edge_index = edge_index.to(device)
    features = features.to(device)
    labels   = labels.to(device)
    # idx_train = split_idx['train']
    # idx_valid = split_idx['valid']
    # idx_test  = split_idx['test']

    # node_num  = data.x.size(0)
    # adj       = to_scipy_sparse_matrix(data.edge_index).tocoo()
    # g         = dgl.DGLGraph()
    # g.add_nodes(node_num)
    # g.add_edges(adj.row,adj.col)
    # g         = g.to(device)
    # features   = data.x.to(device)
    # edge_index = from_scipy(to_scipy_sparse_matrix(data.edge_index)).to(device)
    # labels    = data.y.squeeze().to(device)
    model = get_model(args.model_name, args.dataset_name)(
        num_classes=num_classes, in_dim=args.in_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        use_bias=args.use_bias,
        norm_type=args.norm_type, use_affine_bn=args.use_affine_bn,
        dropout_p=args.dropout_p,
        graph=g, algo=args.algo, sparsity=args.sparsity, scale_method=args.scale_method, width_factor=args.width_factor)
    initialize_params(
        model, w_init_method=args.w_init_method, s_init_method=args.s_init_method, 
        m_init_method=args.m_init_method, p_ratio=args.p_ratio, r_ratio=args.r_ratio, r_method=args.r_method, mask_file=args.mask_file,
        nonlinearity=args.nonlinearity)
    model = model.to(device)
    if args.algo != 'dense':
        ternary_frozen_masks = []
        for m in model.modules():
            if isinstance(m, SupermaskLinear):
                ternary_frozen_masks.append(m.ternary_frozen_mask.clone().detach().flatten())
        ternary_frozen_mask = torch.cat(ternary_frozen_masks)

    set_kthvalue(model, args.algo, device)

    print(model)
    if args.algo != 'dense':
        p_count = 0
        r_count = 0
        total   = 0
        with torch.no_grad():
            for m in filter(lambda x: isinstance(x, SupermaskLinear), model.modules()):
                p_count += (m.ternary_frozen_mask == -1).sum()
                r_count += (m.ternary_frozen_mask ==  1).sum()
                total   += m.ternary_frozen_mask.numel()
            print(f'Pruning ratio :   {p_count/total}')
            print(f'Retaining ratio : {r_count/total}')
            print(f'Total :           {(p_count + r_count)/total}')



    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        optimizer_name = args.optimizer_name,
        lr             = args.lr,
        momentum       = args.momentum,
        weight_decay   = args.weight_decay,
        model          = model
        )
    scheduler = get_scheduler(
        scheduler_name = args.scheduler_name,
        optimizer      = optimizer,
        milestones     = args.milestones,
        gamma          = args.gamma,
        max_epoch      = args.max_epoch
    )

    val_acc_count,  val_total  = evaluate(
        model=model, g=g, edge_index=edge_index, features=features, labels=labels, 
        idx_eval=idx_valid, device=device)
    test_acc_count, test_total = evaluate(
        model=model, g=g, edge_index=edge_index, features=features, labels=labels, 
        idx_eval=idx_test, device=device)

    val_accs  = []
    test_accs = []
    epochs    = []
    ckpt_name = pathlib.Path(f'epoch_{0}.pt')
    torch.save(model.state_dict(), save_dir / ckpt_name)
    print(f"{'-' * 75}\nEpoch: {0}, Val acc.: {(val_acc_count/val_total*100).item()} [%]\n{'-' * 75}")
    val_accs.append((val_acc_count/val_total).item())
    test_accs.append((test_acc_count/test_total).item())
    epochs.append(0)
    best_val_acc = (val_acc_count/val_total).item()


    n_divisions = 1 if args.n_divisions == None else args.n_divisions
    n_rewinds   = 1 if args.n_rewinds   == None else args.n_rewinds
    for n in range(n_rewinds):
        if n_divisions != 1:
            raise NotImplementedError

        for epoch in tqdm(range(1, args.max_epoch+1)):
            print(f"Epoch: {epoch}; lr = {scheduler.get_last_lr()[0]:.5f}")

            train(
                model=model, g=g, edge_index=edge_index, features=features, labels=labels, idx_train=idx_train,
                optimizer=optimizer, criterion=criterion, algo=args.algo, device=device)
                
            scheduler.step()

            if epoch % args.eval_interval == 0:
                val_acc_count,  val_total  = evaluate(
                    model=model, g=g, edge_index=edge_index, features=features, labels=labels, 
                    idx_eval=idx_valid, device=device)
                test_acc_count, test_total = evaluate(
                    model=model, g=g, edge_index=edge_index, features=features, labels=labels, 
                    idx_eval=idx_test, device=device)

                # Save and evaluate model routinely
                if epoch % args.save_interval == 0:
                    ckpt_name = pathlib.Path(f'epoch_{epoch}.pt')
                    torch.save(model.state_dict(), save_dir / ckpt_name)

                print(f"{'-' * 75}\nEpoch: {epoch}, Val acc.: {(val_acc_count/val_total*100).item()} [%]\n{'-' * 75}")
                print(f"{'-' * 75}\nEpoch: {epoch}, Test acc.: {(test_acc_count/test_total*100).item()} [%]\n{'-' * 75}")
                val_accs.append((val_acc_count/val_total).item())
                test_accs.append((test_acc_count/test_total).item())
                epochs.append(epoch + args.max_epoch * n)
                if (val_acc_count/val_total).item() > best_val_acc:
                    print(f'best val acc updated: {best_val_acc} -> {(val_acc_count/val_total).item()}')
                    best_val_ckpt_name = pathlib.Path(f'best_val_ckpt.pt')
                    best_val_acc = (val_acc_count/val_total).item()
                    torch.save(model.state_dict(), save_dir / best_val_ckpt_name)

    df = pd.DataFrame(
        {
            'epoch':    epochs,
            'val_acc':  val_accs,
            'test_acc': test_accs
            }
            )
    df.to_csv(args.save_dir + '/accuracy.csv')

if __name__ == "__main__":
    main()