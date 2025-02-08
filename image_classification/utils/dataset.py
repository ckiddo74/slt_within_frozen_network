import torch
import torchvision
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data           import create_transform

def get_dataset_dim(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def get_datasets(
        dataset_name, dataset_dir, train_val_ratio, seed, use_simple_transform=False
        # color_jitter, aa, train_interpolation, reprob, remode, recount, eval_crop_ratio
        ):
    if dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        train_transform    = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        val_test_transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        trainval_set         = torchvision.datasets.MNIST(dataset_dir, train=True,  download=True)
        transformed_test_set = torchvision.datasets.MNIST(dataset_dir, train=False, download=True, transform=val_test_transform)
    if dataset_name == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        train_transform    = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=True)
        val_test_transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=False)
        trainval_set         = torchvision.datasets.CIFAR10(dataset_dir, train=True,  download=True) 
        transformed_test_set = torchvision.datasets.CIFAR10(dataset_dir, train=False, download=True, transform=val_test_transform) 
    if dataset_name == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        train_transform    = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=True)
        val_test_transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=False)
        trainval_set         = torchvision.datasets.CIFAR100(dataset_dir, train=True,  download=True) 
        transformed_test_set = torchvision.datasets.CIFAR100(dataset_dir, train=False, download=True, transform=val_test_transform) 
    
    # based on https://github.com/facebookresearch/deit/blob/main/datasets.py
    if dataset_name == 'imagenet':
        input_shape, _ = get_dataset_dim(dataset_name)
        # train_transform = create_transform(
        #     input_size=input_shape[1],
        #     is_training=True,
        #     color_jitter=color_jitter,
        #     auto_augment=aa,
        #     interpolation=train_interpolation,
        #     re_prob=reprob,
        #     re_mode=remode,
        #     re_count=recount,
        # )
        if use_simple_transform:
            print('[DEBUG]: Use simple transform')
            # https://github.com/allenai/hidden-networks/blob/master/data/imagenet.py
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ]
                )
            val_test_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ]
                )
            trainval_set         = torchvision.datasets.ImageFolder(dataset_dir + '/train')
            transformed_test_set = torchvision.datasets.ImageFolder(
                dataset_dir + '/val', transform=val_test_transform)
        else:
            train_transform = create_transform(
                input_size=input_shape[1],
                is_training=True,
                color_jitter=0.3,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )
            if not (input_shape[1] > 32):
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                train_transform.transforms[0] = transforms.RandomCrop(input_shape[1], padding=4)
            
            t = []
            if (input_shape[1] > 32):
                # size = int(input_shape[1] / eval_crop_ratio)
                size = int(input_shape[1] / 0.875)

                t.append(
                    transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(input_shape[1]))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            val_test_transform = transforms.Compose(t)

            trainval_set         = torchvision.datasets.ImageFolder(dataset_dir + '/train')
            transformed_test_set = torchvision.datasets.ImageFolder(
                dataset_dir + '/val', transform=val_test_transform)

    n_data  = len(trainval_set)
    n_train = int(n_data * train_val_ratio)
    n_val   = n_data - n_train

    g = torch.Generator()
    g.manual_seed(seed)

    train_set, val_set = torch.utils.data.random_split(
        trainval_set, [n_train, n_val], generator=g)

    transformed_train_set = Subset(train_set, transform=train_transform)
    transformed_val_set   = Subset(val_set,   transform=val_test_transform)

    print(f'-----Dataset-----')
    print(f'train: {len(transformed_train_set)}')
    print(f'val:   {len(transformed_val_set)}')
    print(f'test:  {len(transformed_test_set)}\n')

    return transformed_train_set, transformed_val_set, transformed_test_set

class Subset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)