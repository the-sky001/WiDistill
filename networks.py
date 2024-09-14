import torch.nn as nn
import torch.nn.functional as F
import torch
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
class mmfi_CNN(nn.Module):
    def __init__(self, num_classes):
        super(mmfi_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道数为1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.flatten_size = self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 342, 350)  # 确保输入为单通道
            x = self.features(x)
            flatten_size = x.view(1, -1).size(1)
        return flatten_size
    
    def forward(self, x):
        # 在传递给卷积层之前增加一个通道维度
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
class mmfi_MLP(nn.Module):
    def __init__(self, num_classes):
        super(mmfi_MLP,self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(22*20*20,1024),
            # nn.Linear(270*1000,1024),
            nn.Linear(3*114*350,256),

            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        # x = x.view(-1,270*1000)
        # x = x.view(-1,270*500)
        # print("1",x.shape)
        
        x = x.view(-1,114*350*3)

        # print("2",x.shape)

        x = self.fc(x)
        # print("3",x.shape)
        return x
class mmfi_ResidualBlock(nn.Module):
    expansion = 1  # 添加 expansion 属性

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(mmfi_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class mmfi_Bottleneck(nn.Module):
    expansion = 4  # 添加 expansion 属性

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(mmfi_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class mmfi_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(mmfi_ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.mmfi_make_layer(block, 64, layers[0])
        self.layer2 = self.mmfi_make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.mmfi_make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.mmfi_make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def mmfi_make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input shape is [batch_size, 1, 342, 350]
        if x.dim() == 3:  # If input is [batch_size, 342, 350]
            x = x.unsqueeze(1)  # Convert to [batch_size, 1, 342, 350]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def mmfi_resnet18(num_classes=1000):
    return mmfi_ResNet(mmfi_ResidualBlock, [2, 2, 2, 2], num_classes)
class xrf_MLP(nn.Module):
    def __init__(self, num_classes):
        super(xrf_MLP,self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(22*20*20,1024),
            # nn.Linear(270*1000,1024),
            nn.Linear(270*1000,1024),

            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,270*1000)
        # x = x.view(-1,270*500)
        # x = x.view(-1,90*100)


        x = self.fc(x)
        return x
def conv3x3(in_planes, out_planes, stride=1, group=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, group=group)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



def xrf_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def xrf_conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)
class xrf_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(xrf_BasicBlock, self).__init__()
        self.conv1 = xrf_conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = xrf_conv3x3(planes, planes, stride=1, groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)
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
class xrf_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(xrf_Bottleneck, self).__init__()
        self.conv1 = xrf_conv1x1(inplanes, planes, stride=1, groups=groups)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = xrf_conv3x3(planes, planes, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = xrf_conv1x1(planes, planes * self.expansion, stride=1, groups=groups)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class xrf_ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=270, num_classes=55):
        super(xrf_ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, groups=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, groups=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, groups=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, groups=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                xrf_conv1x1(self.inplanes, planes * block.expansion, stride, groups=groups),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:  # 适用于 [batch_size, 1, in_channels, sequence_length] 的输入
            x = x.squeeze(1)  # 删除维度 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def xrf_resnet18(num_classes=55):
    """ return a ResNet 18 object """
    return xrf_ResNet(xrf_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
def xrf_resnet50(num_classes=55):
    """ return a ResNet 50 object """
    return xrf_ResNet(xrf_Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
def xrf_resnet101(num_classes=55):
    """ return a ResNet 101 object """
    return xrf_ResNet(xrf_Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
class xrf_CNN(nn.Module):
    def __init__(self, num_classes=55):
        super(xrf_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(270, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 计算卷积层输出的特征图尺寸
        self.flatten_size = self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def _get_flatten_size(self):
        # 创建一个示例输入，用于计算扁平化后的尺寸
        with torch.no_grad():
            x = torch.zeros(1, 270, 1000)
            x = self.features(x)
            flatten_size = x.view(1, -1).size(1)
        return flatten_size

    def forward(self, x):
        if x.dim() == 4:  # 适用于 [batch_size, 1, in_channels, sequence_length] 的输入
            x = x.squeeze(1)  # 删除维度 1
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22*20*20,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,22*20*20)
        x = self.fc(x)
        return x

class Widar_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(Widar_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (22,20,20)
            nn.Conv2d(22,32,6,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=1),
            nn.ReLU(True),
            nn.Conv2d(64,96,3,stride=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*4)
        out = self.fc(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, i_downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
def widar_conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

def widar_conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)

class widar_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, downsample=None):
        super(widar_BasicBlock, self).__init__()
        self.conv1 = widar_conv3x3(inplanes, planes, stride, groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = widar_conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
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

class widar_ResNet2D(nn.Module):
    def __init__(self, block, layers, in_channels=22, num_classes=6):
        super(widar_ResNet2D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, groups=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, groups=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, groups=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, groups=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                widar_conv1x1(self.inplanes, planes * block.expansion, stride, groups=groups),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def widar_resnet18(num_classes=6):
    """ return a ResNet 18 object for 2D input """
    return widar_ResNet2D(widar_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        
        


class Widar_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(Widar_CNN, self).__init__()
        self.conv1 = nn.Conv2d(22, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 2 * 2)  # 展平操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Widar_RNN(nn.Module):
    def __init__(self,num_classes):
        super(Widar_RNN,self).__init__()
        self.rnn = nn.RNN(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

class Widar_GRU(nn.Module):
    def __init__(self,num_classes):
        super(Widar_GRU,self).__init__()
        self.gru = nn.GRU(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,22,400)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs

class Widar_LSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_LSTM,self).__init__()
        self.lstm = nn.LSTM(400,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    # def forward(self,x):
    #     x = x.view(-1,22,400)
    #     x = x.permute(1,0,2)
    #     _, (ht,ct) = self.lstm(x)
    #     outputs = self.fc(ht[-1])
    #     return outputs
    def forward(self, x):
        # 将输入张量x调整形状并转置
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)

        # 禁用CuDNN进行LSTM的前向和反向传播
        with torch.backends.cudnn.flags(enabled=False):
            _, (ht, ct) = self.lstm(x)
        
        # 应用全连接层计算最终输出
        outputs = self.fc(ht[-1])
        return outputs

class Widar_BiLSTM(nn.Module):
    def __init__(self,num_classes):
        super(Widar_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(400,64,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(64,num_classes)
    # def forward(self,x):
    #     x = x.view(-1,22,400)
    #     x = x.permute(1,0,2)
    #     _, (ht,ct) = self.lstm(x)
    #     outputs = self.fc(ht[-1])
    #     return outputs
    def forward(self, x):
        # 将输入张量x调整形状并转置
        x = x.view(-1, 22, 400)
        x = x.permute(1, 0, 2)

        # 禁用CuDNN进行LSTM的前向和反向传播
        with torch.backends.cudnn.flags(enabled=False):
            _, (ht, ct) = self.lstm(x)
        
        # 应用全连接层计算最终输出
        outputs = self.fc(ht[-1])
        return outputs

class Widar_CNN_GRU(nn.Module):
    def __init__(self,num_classes):
        super(Widar_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,6,2),
            nn.ReLU(),
            nn.Conv2d(8,16,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*3*3,64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        batch_size = len(x)
        # batch x 22 x 20 x 20
        x = x.view(batch_size*22,1,20,20)
        # 22*batch x 1 x 20 x 20
        x = self.encoder(x)
        # 22*batch x 16 x 3 x 3
        x = x.view(-1,16*3*3)
        x = self.fc(x)
        # 22*batch x 64
        x = x.view(-1,22,64)
        x = x.permute(1,0,2)
        # 22 x batch x 64
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 2, patch_size_h = 40, emb_size = 2*40, img_size = 22*400):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
    
    def forward(self, x):
        x = x.view(-1,1,22,400)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 80, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 80,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
        
class Widar_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 2,
                patch_size_h = 40,
                emb_size = 80,
                img_size = 22*400,
                depth = 1,
                *,
                num_classes,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


    def __init__(self, num_classes):
        super(MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22*20*20,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,22*20*20)
        x = self.fc(x)
        return x


''' widar_mlp '''        
class Widar_MLP(nn.Module):
    def __init__(self, num_classes):
        super(Widar_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(22*20*20,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = x.view(-1,22*20*20)
        x = self.fc(x)
        return x

