import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True)

    #experiment setting
    parser.add_argument('--gpu_ids', nargs='*', type=int)
    parser.add_argument('--omp_num_threads',    type=int)
    parser.add_argument('--save_dir',           type=str)
    parser.add_argument('--seed',               type=int)

    # training/inference setting
    parser.add_argument('--optimizer_name',        type=str)
    parser.add_argument('--lr',                    type=float)
    parser.add_argument('--momentum',              type=float)
    parser.add_argument('--weight_decay',          type=float)
    parser.add_argument('--scheduler_name',        type=str)
    parser.add_argument('--milestones', nargs='*', type=int)
    parser.add_argument('--gamma',                 type=float)
    parser.add_argument('--max_epoch',             type=int)
    parser.add_argument('--eval_interval',         type=int)
    parser.add_argument('--save_interval',         type=int)


    # model setting
    parser.add_argument('--model_name',    type=str)
    parser.add_argument('--in_dim',        type=int)
    parser.add_argument('--hidden_dim',    type=int)
    parser.add_argument('--n_layers',      type=int)
    parser.add_argument('--use_bias',      action='store_true')
    parser.add_argument('--nonlinearity',  type=str)
    parser.add_argument('--dropout_p',     type=float)
    parser.add_argument('--norm_type',     type=str)
    parser.add_argument('--use_affine_bn', action='store_true')
    parser.add_argument('--width_factor',  type=float)
    parser.add_argument('--algo',          type=str)
    parser.add_argument('--sparsity',      type=float)
    parser.add_argument('--scale_method',  type=str)
    parser.add_argument('--w_init_method', type=str)
    parser.add_argument('--s_init_method', type=str)
    parser.add_argument('--m_init_method', type=str)
    parser.add_argument('--p_ratio',       type=float)
    parser.add_argument('--r_ratio',       type=float)
    parser.add_argument('--r_method',      type=str)
    parser.add_argument('--n_divisions',   type=int)
    parser.add_argument('--n_rewinds',     type=int)
    parser.add_argument('--fitness',       type=str)
    parser.add_argument('--mask_file',     type=str)

    # dataset setting
    parser.add_argument('--dataset_name',    type=str)
    parser.add_argument('--dataset_dir',     type=str)
    parser.add_argument('--sampling',        type=float)
    parser.add_argument('--samplingtype',      type=str)
    parser.add_argument('--train_val_ratio', type=float)
    parser.add_argument('--n_worker',        type=int)

    return parser