import os
import subprocess
import sys

import yaml

# from utils.dataset import get_datasets
from utils.parser  import get_parser

def get_config(args):
    def argv_to_vars(argv):
        var_names = []
        for arg in argv:
            if arg.startswith("-") and arg_to_varname(arg) != "config":
                var_names.append(arg_to_varname(arg))
        return var_names
    
    def arg_to_varname(st: str):
        st = trim_preceding_hyphens(st)
        st = st.replace("-", "_")
        return st.split("=")[0]
    
    def trim_preceding_hyphens(st):
        i = 0
        while st[i] == "-":
            i += 1
        return st[i:]

    # get commands from command line
    override_args = argv_to_vars(sys.argv)
    # load yaml file
    with open(args.config) as f:
        loaded_yaml = yaml.safe_load(f)
    for v in override_args:
        if getattr(args, v) != None:
            loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}\n")
    args.__dict__.update(loaded_yaml)
    
def show_args(args):
    print('-----Exp setting-----')
    for k, v in args.__dict__.items():
        print(f'{k} : {v}')
    print()

# def check_n_data(dataset_name, dataset_dir, n_gpu, batch_size_per_gpu, train_val_ratio, seed):
#     print(f'Check number of data...')
#     train_set, val_set, test_set = get_datasets(
#         dataset_name=dataset_name, dataset_dir=dataset_dir, train_val_ratio=train_val_ratio, seed=seed)
#     for name, dataset, in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
#         if (len(dataset) % (n_gpu * batch_size_per_gpu)) % n_gpu != 0:
#             raise ValueError(f'{name}: Unable to use all data.')
#     print(f'Done.\n')


def main():
    parser = get_parser()
    args = parser.parse_args()

    get_config(args)
    show_args(args)

    # check_n_data(
    #     dataset_name       = args.dataset_name, 
    #     dataset_dir        = args.dataset_dir, 
    #     n_gpu              = len(args.gpu_ids), 
    #     batch_size_per_gpu = args.batch_size_per_gpu, 
    #     train_val_ratio    = args.train_val_ratio,
    #     seed               = args.seed
    #     )
    if args.width_factor != None and args.width_factor != 1:
        args.save_dir += f'/{args.dataset_name}-{args.model_name}-{args.width_factor}-{args.algo}'
    else:
        args.save_dir += f'/{args.dataset_name}-{args.model_name}-{args.algo}'
    if args.algo in ['local_ep', 'global_ep']:
        args.save_dir += f'-{args.sparsity}'
    if args.m_init_method != None:
        assert args.p_ratio != None and args.r_ratio != None
        if args.mask_file != None:
            args.save_dir += f'-{args.m_init_method}-oracle-{args.p_ratio+args.r_ratio}'
        else:
            args.save_dir += f'-{args.m_init_method}-{args.p_ratio}-{args.r_ratio}'
    if args.r_method != None and args.r_method != 'density_distribution':
        args.save_dir += f'-{args.r_method}'
    if args.n_divisions != None:
        args.save_dir += f'-greedy-{args.n_divisions}'
    if args.n_rewinds != None:
        args.save_dir += f'-rewind-{args.n_rewinds}'
    args.save_dir += f'/seed_{args.seed}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.save_dir + '/args.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
    try:
        envs = os.environ.copy()
        envs['CUDA_VISIBLE_DEVICES']    = ','.join([str(i) for i in args.gpu_ids])
        envs['OMP_NUM_THREADS']         = str(args.omp_num_threads)
        envs['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        cmds = [
            'python',
            'experiment.py' 
        ]

        for k, v in args.__dict__.items():
            if isinstance(v, list):
                cmds.append(f'--{k}')
                cmds.extend([str(i) for i in v])
            elif isinstance(v, (int, float, str)):
                if isinstance(v, bool):
                    if v:
                        cmds.append(f'--{k}')
                else:
                    cmds.append(f'--{k}')
                    cmds.append(str(v))
            elif v == None:
                pass
            else:
                raise ValueError(f'Value of {k} is {type(v)}.')

        print(f'Experiment command is \n{" ".join(cmds)}')
        result = subprocess.run(cmds, check=True, env=envs)
        print(f'Experiment Finished.')

    except subprocess.CalledProcessError as e:
        print(e)


if __name__ == "__main__":
    main()