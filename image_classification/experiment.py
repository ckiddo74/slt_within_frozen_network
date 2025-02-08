# based on https://leimao.github.io/blog/PyTorch-Distributed-Training/
import os
import pathlib

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
from torch.utils.data             import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto                    import tqdm
from timm.data                    import Mixup
from timm.utils                   import ModelEma, NativeScaler
from timm.loss                    import SoftTargetCrossEntropy

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
from utils.sampler import RASampler
from layers.utils.set_kthvalue      import set_kthvalue
from layers.supermask_conv          import SupermaskConv2d
from layers.supermask_linear        import SupermaskLinear
from layers.utils.set_initial_value import set_initial_value



def main():
    parser = get_parser()
    args   = parser.parse_args()

    local_rank      = int(os.environ["LOCAL_RANK"])
    device          = torch.device(f'cuda:{local_rank}')
    save_dir        = pathlib.Path(args.save_dir)
    set_random_seeds(seed=args.seed)

    dist.init_process_group(backend="nccl")

    input_shape, num_classes = get_dataset_dim(args.dataset_name) 
    mixup_fn = None
    if args.mixup == None and args.cutmix == None:
        mixup_active = False
    else:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
    model = get_model(args.model_name, args.dataset_name)(
        num_classes=num_classes, use_affine_bn=args.use_affine_bn, 
        algo=args.algo, sparsity=args.sparsity, scale_method=args.scale_method, width_factor=args.width_factor,
        drop_rate=args.drop, drop_path_rate=args.drop_path)

    initialize_params(
        model, w_init_method=args.w_init_method, s_init_method=args.s_init_method, 
        m_init_method=args.m_init_method, p_ratio=args.p_ratio, r_ratio=args.r_ratio, r_method=args.r_method, mask_file=args.mask_file,
        nonlinearity=args.nonlinearity, algo=args.algo)
    model = model.to(device)
    if args.algo != 'dense':
        ternary_frozen_masks = []
        for m in model.modules():
            if isinstance(m, (SupermaskLinear, SupermaskConv2d)):
                ternary_frozen_masks.append(m.ternary_frozen_mask.clone().detach().flatten())
        ternary_frozen_mask = torch.cat(ternary_frozen_masks)

    set_kthvalue(model, args.algo, device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)

    ddp_model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper <- Is this true?
        ddp_model_ema = ModelEma(
            ddp_model,
            decay=args.model_ema_decay,
            device=device,
            resume='')

    if local_rank == 0:
        print(ddp_model)
        if args.algo != 'dense':
            p_count = 0
            r_count = 0
            total   = 0
            with torch.no_grad():
                for m in filter(lambda x: isinstance(x, (SupermaskLinear, SupermaskConv2d)), ddp_model.modules()):
                    p_count += (m.ternary_frozen_mask == -1).sum()
                    r_count += (m.ternary_frozen_mask ==  1).sum()
                    total   += m.ternary_frozen_mask.numel()
                print(f'Pruning ratio :   {p_count/total}')
                print(f'Retaining ratio : {r_count/total}')
                print(f'Total :           {(p_count + r_count)/total}')

    train_set, val_set, test_set = get_datasets(
        dataset_name         = args.dataset_name, 
        dataset_dir          = args.dataset_dir, 
        train_val_ratio      = args.train_val_ratio,
        seed                 = args.seed,
        use_simple_transform = args.use_simple_transform
        )

    # Restricts data loading to a subset of the dataset exclusive to the current process
    if args.repeated_aug:
        assert args.dataset_name == 'imagenet', '--repeated_aug option is available only for ImageNet exp.'
        train_sampler = RASampler(dataset=train_set, shuffle=True)
    else:
        train_sampler = DistributedSampler(dataset=train_set, seed=args.seed, shuffle=True,  drop_last=True)
    val_sampler   = DistributedSampler(dataset=val_set,   seed=args.seed, shuffle=False, drop_last=True)
    test_sampler  = DistributedSampler(dataset=test_set,  seed=args.seed, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        dataset     = train_set, 
        batch_size  = args.batch_size_per_gpu, 
        sampler     = train_sampler, 
        num_workers = args.n_worker,
        pin_memory  = True
        )
    val_loader = DataLoader(
        dataset     = val_set, 
        batch_size  = args.batch_size_val_per_gpu, 
        sampler     = val_sampler, 
        num_workers = args.n_worker_eval,
        pin_memory  = True
        )
    test_loader = DataLoader(
        dataset     = test_set, 
        batch_size  = args.batch_size_test_per_gpu, 
        sampler     = test_sampler, 
        num_workers = args.n_worker_test,
        pin_memory  = True
        )

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        optimizer_name     = args.optimizer_name,
        lr                 = args.lr,
        momentum           = args.momentum,
        weight_decay       = args.weight_decay,
        model              = ddp_model,
        filter_bias_and_bn = args.filter_bias_and_bn
        )
    
    loss_scaler = None
    if args.use_amp:
        loss_scaler = NativeScaler()

    scheduler = get_scheduler(
        scheduler_name = args.scheduler_name,
        optimizer      = optimizer,
        milestones     = args.milestones,
        gamma          = args.gamma,
        max_epoch      = args.max_epoch,
        min_lr         = args.min_lr,
        warmup_lr_init = args.warmup_lr_init,
        warmup_t       = args.warmup_t,
        warmup_prefix  = args.warmup_prefix
    )

    val_acc_count,  val_total  = evaluate(
        model=ddp_model, device=device, eval_loader=val_loader, 
        use_non_blocking=args.use_non_blocking, use_amp=args.use_amp)
    test_acc_count, test_total = evaluate(
        model=ddp_model, device=device, eval_loader=test_loader,
        use_non_blocking=args.use_non_blocking, use_amp=args.use_amp)
    dist.all_reduce(val_acc_count,  op=dist.ReduceOp.SUM)
    dist.all_reduce(val_total,      op=dist.ReduceOp.SUM)
    dist.all_reduce(test_acc_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_total,     op=dist.ReduceOp.SUM)

    if local_rank == 0:
        
        val_accs  = []
        test_accs = []
        # if args.model_ema:
        #     ema_val_accs  = []
        #     ema_test_accs = []
        epochs    = []

        ckpt_name = pathlib.Path(f'epoch_{0}.pt')
        torch.save(ddp_model.state_dict(), save_dir / ckpt_name)
        print(f"{'-' * 75}\nEpoch: {0}, Val acc.: {(val_acc_count/val_total*100).item()} [%]\n{'-' * 75}")
        val_accs.append((val_acc_count/val_total).item())
        test_accs.append((test_acc_count/test_total).item())
        # if args.model_ema:
        #     ema_val_accs.append(np.nan)
        #     ema_test_accs.append(np.nan)
        epochs.append(0)
        best_val_acc = (val_acc_count/val_total).item()
        # best_ema_val_acc = -1
    dist.barrier()

    n_divisions = 1 if args.n_divisions == None else args.n_divisions
    n_rewinds   = 1 if args.n_rewinds   == None else args.n_rewinds
    for n in range(n_rewinds):
        if n_divisions != 1:
            raise NotImplementedError

        for epoch in tqdm(range(1, args.max_epoch+1), disable= (local_rank != 0)):
            if hasattr(scheduler, 'get_last_lr'):
                print(f"[rank {local_rank}] Epoch: {epoch}; lr = {scheduler.get_last_lr()[0]:.5f}")
            else:
                print(f"[rank {local_rank}] Epoch: {epoch}; lr = {optimizer.param_groups[0]['lr']:.8f}")
                


            train_sampler.set_epoch(epoch)
            train(model=ddp_model, device=device, train_loader=train_loader,
                optimizer=optimizer, criterion=criterion, algo=args.algo,
                model_ema=ddp_model_ema, mixup_fn=mixup_fn, use_non_blocking=args.use_non_blocking,
                use_amp=args.use_amp, loss_scaler=loss_scaler, max_norm=args.clip_grad)
            
            if args.scheduler_name == 'cosine_lr_warmup':
                # exception handling only for scheduler of timm library
                scheduler.step(epoch-1)
            else:
                scheduler.step()

            if epoch % args.eval_interval == 0:
                val_acc_count, val_total   = evaluate(
                    model=ddp_model, device=device, eval_loader=val_loader, 
                    use_non_blocking=args.use_non_blocking, use_amp=args.use_amp)
                test_acc_count, test_total = evaluate(
                    model=ddp_model, device=device, eval_loader=test_loader,
                    use_non_blocking=args.use_non_blocking, use_amp=args.use_amp)
                dist.all_reduce(val_acc_count,  op=dist.ReduceOp.SUM)
                dist.all_reduce(val_total,      op=dist.ReduceOp.SUM)
                dist.all_reduce(test_acc_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(test_total,     op=dist.ReduceOp.SUM)
                # if args.model_ema:
                #     set_kthvalue(ddp_model_ema.ema, args.algo, device)
                #     evaluate(
                #         model=ddp_model_ema.ema, device=device, eval_loader=test_loader,
                #         use_non_blocking=args.use_non_blocking, use_amp=args.use_amp, 
                #         only_update_mask=True)

                # Save and evaluate model routinely
                if local_rank == 0:
                    if epoch % args.save_interval == 0:
                        ckpt_name = pathlib.Path(f'epoch_{epoch}.pt')
                        torch.save(ddp_model.state_dict(), save_dir / ckpt_name)
                        # if args.model_ema:
                        #     ckpt_name = pathlib.Path(f'epoch_{epoch}_ema.pt')
                        #     torch.save(ddp_model_ema.ema.state_dict(), save_dir / ckpt_name)

                    print(f"{'-' * 75}\nEpoch: {epoch}, Val acc.: {(val_acc_count/val_total*100).item()} [%]\n{'-' * 75}")
                    print(f"{'-' * 75}\nEpoch: {epoch}, Test acc.: {(test_acc_count/test_total*100).item()} [%]\n{'-' * 75}")
                    val_accs.append((val_acc_count/val_total).item())
                    test_accs.append((test_acc_count/test_total).item())
                    # if args.model_ema:
                    #     ema_val_accs.append((ema_val_acc_count/ema_val_total).item())
                    #     ema_test_accs.append((ema_test_acc_count/ema_test_total).item())

                    epochs.append(epoch + args.max_epoch * n)
                    if (val_acc_count/val_total).item() > best_val_acc:
                        print(f'best val acc updated: {best_val_acc} -> {(val_acc_count/val_total).item()}')
                        best_val_ckpt_name = pathlib.Path(f'best_val_ckpt.pt')
                        best_val_acc = (val_acc_count/val_total).item()
                        torch.save(ddp_model.state_dict(), save_dir / best_val_ckpt_name)
                        if args.model_ema:
                            best_ema_val_ckpt_name = pathlib.Path(f'best_ema_val_ckpt.pt')
                            torch.save(ddp_model_ema.ema.state_dict(), save_dir / best_ema_val_ckpt_name)
            dist.barrier()

    if local_rank == 0:
        results_dict = {
                'epoch':    epochs,
                'val_acc':  val_accs,
                'test_acc': test_accs
                }
        # if args.model_ema:
        #     results_dict['ema_val_acc']  = ema_val_accs
        #     results_dict['ema_test_acc'] = ema_test_accs
        df = pd.DataFrame(results_dict)
        df.to_csv(args.save_dir + '/accuracy.csv')

if __name__ == "__main__":
    main()