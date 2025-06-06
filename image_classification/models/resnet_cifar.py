# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.supermask_conv          import SupermaskConv2d
from layers.supermask_linear        import SupermaskLinear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, 
        use_affine_bn=None, sparsity=None, algo=None, scale_method=None):
        super(BasicBlock, self).__init__()
        if algo == 'dense':
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = SupermaskConv2d(
                in_planes, planes, kernel_size=3, 
                sparsity=sparsity, algo=algo, scale_method=scale_method,
                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=use_affine_bn)
        if algo == 'dense':
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        else:
            self.conv2 = SupermaskConv2d(
                planes, planes, kernel_size=3,
                sparsity=sparsity, algo=algo, scale_method=scale_method,                
                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=use_affine_bn)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if algo == 'dense':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes, affine=use_affine_bn)
                )
            else:
                self.shortcut = nn.Sequential(
                    SupermaskConv2d(
                        in_planes, self.expansion*planes, kernel_size=1, 
                        sparsity=sparsity, algo=algo, scale_method=scale_method,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes, affine=use_affine_bn)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, 
        use_affine_bn=None, sparsity=None, algo=None, scale_method=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self, block, num_blocks,
            num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor):
        super(ResNet, self).__init__()

        if width_factor == None:
            width_factor = 1
            
        self.in_planes = int(64*width_factor)

        if algo == 'dense':
            self.conv1 = nn.Conv2d(
                3, int(64*width_factor), kernel_size=3, stride=1, padding=1, bias=False)
        else: 
            self.conv1 = SupermaskConv2d(
                3, int(64*width_factor), kernel_size=3, sparsity=sparsity, algo=algo, scale_method=scale_method, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*width_factor), affine=use_affine_bn)
        
        self.layer1 = self._make_layer(
            block, int(64*width_factor), num_blocks[0], stride=1,
            use_affine_bn=use_affine_bn, sparsity=sparsity, algo=algo, scale_method=scale_method)
        self.layer2 = self._make_layer(
            block, int(128*width_factor), num_blocks[1], stride=2,
            use_affine_bn=use_affine_bn, sparsity=sparsity, algo=algo, scale_method=scale_method)
        self.layer3 = self._make_layer(
            block, int(256*width_factor), num_blocks[2], stride=2,
            use_affine_bn=use_affine_bn, sparsity=sparsity, algo=algo, scale_method=scale_method)
        self.layer4 = self._make_layer(
            block, int(512*width_factor), num_blocks[3], stride=2,
            use_affine_bn=use_affine_bn, sparsity=sparsity, algo=algo, scale_method=scale_method)
        
        if algo == 'dense':
            self.linear = nn.Linear(int(512*width_factor)*block.expansion, num_classes, bias=False)
        else:
            self.linear = SupermaskLinear(
                int(512*width_factor)*block.expansion, num_classes, 
                sparsity=sparsity, algo=algo, scale_method=scale_method, bias=False)

    def _make_layer(
            self, block, planes, num_blocks, stride,
            use_affine_bn, sparsity, algo, scale_method):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, stride,
                    use_affine_bn=use_affine_bn, sparsity=sparsity, algo=algo, scale_method=scale_method
                    )
                )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return ResNet(
        BasicBlock, [2, 2, 2, 2],
        num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor)


def resnet34(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    return ResNet(
        BasicBlock, [3, 4, 6, 3],
        num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor)


def resnet50(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    raise NotImplementedError
    return ResNet(
        Bottleneck, [3, 4, 6, 3],
        num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor)


def resnet101(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    raise NotImplementedError
    return ResNet(
        Bottleneck, [3, 4, 23, 3],
        num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor)


def resnet152(num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor, **kwargs):
    raise NotImplementedError
    return ResNet(
        Bottleneck, [3, 8, 36, 3],
        num_classes, use_affine_bn, algo, sparsity, scale_method, width_factor)