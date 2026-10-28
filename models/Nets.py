#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        feature_dim = 512 * 1 * 1  # 512, for 32x32 inputs

        self.classifier = nn.Sequential(
            # 14
            nn.Linear(feature_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.fc = nn.Linear(4096, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.fc(out)
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result


def ResNet50(args):
    num_block = [3, 4, 6, 3]
    return ResNet4(args, ImprovedBottleneck, num_block, num_classes=args.num_classes)


class ResNet4(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet4, self).__init__()
        # More initial channels
        self.inplanes = 32  # increased from 16 to 32

        # Stronger initial feature extraction
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # More channels
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # 32 channels
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # 64 channels
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # 128 channels
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)  # 256 channels



        self.bn_final = nn.BatchNorm2d(256 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)

        # Improved classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Add dropout against overfitting
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        result = {}

        # Backbone
        x = self.layer1(x)
        result['activation1'] = x

        x = self.layer2(x)
        result['activation2'] = x

        x = self.layer3(x)
        result['activation3'] = x

        x = self.layer4(x)
        result['activation4'] = x

        # Final processing
        x = self.bn_final(x)
        x = self.relu_final(x)

        # Classification head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result['representation'] = x
        x = self.dropout(x)
        x = self.fc(x)
        result['output'] = x

        return result


class ImprovedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ImprovedBottleneck, self).__init__()
        # Bottleneck layer 1: 1x1 convolution (channel reduction)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Bottleneck layer 2: 3x3 grouped convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(planes)

        # Bottleneck layer 3: 1x1 convolution (channel expansion)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)



        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 grouped convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 convolution
        out = self.conv3(out)
        out = self.bn3(out)


        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """Standard ResNet-18/34 basic block (He et al. 2016) -- two plain 3x3 convs,
    no bottleneck/channel-reduction, no grouping (unlike ImprovedBottleneck above,
    which is specific to the Bottleneck-style architectures this repo's ResNet50
    uses). Added so ResNet18 below can reuse ResNet4's stem/bn_final/dropout/fc
    exactly as-is, isolating "block type + depth" as the only thing that changes
    relative to this repo's own ResNet50."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def ResNet18(args):
    num_block = [2, 2, 2, 2]
    return ResNet4(args, BasicBlock, num_block, num_classes=args.num_classes)


class CifarResNet(nn.Module):
    """Canonical CIFAR ResNet (BasicBlock), kept bit-for-bit architecturally
    identical to `decfus/models/resnet_small.py` in the decfus-reproduced repo so
    that cross-codebase experiments isolate the DFL algorithm rather than the
    network. Deliberately NOT built on ResNet4 above: ResNet4 carries this repo's
    own customisations (32-channel stem, extra bn_final/relu_final, dropout
    before fc) which the other codebase does not have.

    Returns the same dict interface the rest of this repo expects ('output').
    """

    def __init__(self, args, stage_blocks, stage_channels, num_classes=10):
        super(CifarResNet, self).__init__()
        assert len(stage_blocks) == len(stage_channels)
        self.stage_blocks = tuple(stage_blocks)
        self.stage_channels = tuple(stage_channels)

        self.inplanes = stage_channels[0]
        self.conv1 = nn.Conv2d(args.num_channels, stage_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stage_channels[0])
        self.relu = nn.ReLU(inplace=True)

        for k, (blocks, planes) in enumerate(zip(stage_blocks, stage_channels)):
            stride = 1 if k == 0 else 2
            setattr(self, 'layer{}'.format(k + 1), self._make_layer(planes, blocks, stride))
        self.num_stages = len(stage_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[-1] * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        out_planes = planes * BasicBlock.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = out_planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        result = {}
        x = self.relu(self.bn1(self.conv1(x)))
        for k in range(1, self.num_stages + 1):
            x = getattr(self, 'layer{}'.format(k))(x)
            result['activation{}'.format(k)] = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result['representation'] = x
        x = self.fc(x)
        result['output'] = x
        return result


def ResNet8(args):
    return CifarResNet(args, stage_blocks=(1, 1, 1), stage_channels=(16, 32, 64),
                       num_classes=args.num_classes)


def ResNet18Cifar(args):
    return CifarResNet(args, stage_blocks=(2, 2, 2, 2), stage_channels=(64, 128, 256, 512),
                       num_classes=args.num_classes)
