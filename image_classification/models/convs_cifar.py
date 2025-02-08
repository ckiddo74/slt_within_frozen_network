import torch.nn as nn
import torch.nn.functional as F

from layers.supermask_conv          import SupermaskConv2d
from layers.supermask_linear        import SupermaskLinear

class Conv2(nn.Module):
    def __init__(self, algo, sparsity, scale_method, width_factor):
        super(Conv2, self).__init__()
        if width_factor == None:
            width_factor = 1
        if algo == 'dense':
            self.convs = nn.Sequential(
                nn.Conv2d(
                    3, int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.linears = nn.Sequential(
                nn.Linear(int(64*width_factor) * 16 * 16, int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), 10, bias=False)
            )
        else:
            self.convs = nn.Sequential(
                SupermaskConv2d(
                    3, int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.linears = nn.Sequential(
                SupermaskLinear(int(64*width_factor) * 16 * 16, int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), 10, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False)
            )


    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out

class Conv4(nn.Module):
    def __init__(self, algo, sparsity, scale_method, width_factor):
        super(Conv4, self).__init__()
        if width_factor == None:
            width_factor = 1
        if algo == 'dense':
            self.convs = nn.Sequential(
                nn.Conv2d(
                    3, int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            )
            self.linears = nn.Sequential(
                nn.Linear(int(128*width_factor) * 16 * 16, int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), 10, bias=False)
            )
        else:
            self.convs = nn.Sequential(
                SupermaskConv2d(
                    3, int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.linears = nn.Sequential(
                SupermaskLinear(int(128*width_factor) * 16 * 16, int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), 10, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False)
            )


    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out


class Conv6(nn.Module):
    def __init__(self, algo, sparsity, scale_method, width_factor):
        super(Conv6, self).__init__()
        if width_factor == None:
            width_factor = 1
        if algo == 'dense':
            self.convs = nn.Sequential(
                nn.Conv2d(
                    3, int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(128*width_factor), int(256*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(256*width_factor), int(256*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            )
            self.linears = nn.Sequential(
                nn.Linear(int(256*width_factor) * 4 * 4, int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), 10, bias=False)
            )
        else:
            self.convs = nn.Sequential(
                SupermaskConv2d(
                    3, int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(128*width_factor), int(256*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(256*width_factor), int(256*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.linears = nn.Sequential(
                SupermaskLinear(int(256*width_factor) * 4 * 4, int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), 10, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False)
            )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out


class Conv8(nn.Module):
    def __init__(self, algo, sparsity, scale_method, width_factor):
        super(Conv8, self).__init__()
        if width_factor == None:
            width_factor = 1
        if algo == 'dense':
            self.convs = nn.Sequential(
                nn.Conv2d(
                    3, int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(128*width_factor), int(256*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(256*width_factor), int(256*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(
                    int(256*width_factor), int(512*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(
                    int(512*width_factor), int(512*width_factor), kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            )
            self.linears = nn.Sequential(
                nn.Linear(int(512*width_factor) * 2 * 2, int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), int(256*width_factor), bias=False),
                nn.ReLU(),
                nn.Linear(int(256*width_factor), 10, bias=False)
            )
        else:
            self.convs = nn.Sequential(
                SupermaskConv2d(
                    3, int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(64*width_factor), int(64*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(64*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(128*width_factor), int(128*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(128*width_factor), int(256*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(256*width_factor), int(256*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                SupermaskConv2d(
                    int(256*width_factor), int(512*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                SupermaskConv2d(
                    int(512*width_factor), int(512*width_factor), kernel_size=3, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.linears = nn.Sequential(
                SupermaskLinear(int(512*width_factor) * 2 * 2, int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), int(256*width_factor), sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False),
                nn.ReLU(),
                SupermaskLinear(int(256*width_factor), 10, sparsity=sparsity, 
                    algo=algo, scale_method=scale_method, bias=False)
            )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.linears(out)
        return out



def conv2(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return Conv2(algo, sparsity, scale_method, width_factor)

def conv4(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return Conv4(algo, sparsity, scale_method, width_factor)

def conv6(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return Conv6(algo, sparsity, scale_method, width_factor)

def conv8(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return Conv8(algo, sparsity, scale_method, width_factor)