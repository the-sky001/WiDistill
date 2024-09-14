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


class ResNet(nn.Module):

    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNet, self).__init__()
        # B*270*1000
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)
        

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        output = self.avg_pool(c4)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class ResNetLargeBert3(nn.Module):
    '''bert pooling: 1024->55'''
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super(ResNetLargeBert3, self).__init__()
        # B*270*1000
        self.inplanes = 256
        self.conv1 = nn.Conv1d(inchannel, 256, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(1024, 1024, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024 * block.expansion, activity_num)
       

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, group, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        output = self.avg_pool(c4)
        output_bert = output.view(output.size(0), -1)
        output = self.fc(output_bert)

        return output, output_bert


       
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet18_mutual():
    """ return a ResNet 18 object
    """
    return ResNetLargeBert3(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet34_mutual():
    """ return a ResNet 34 object
    """
    return ResNetLargeBert3(BasicBlock, [3, 4, 6, 3])


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet50_mutual():
    return ResNetLargeBert4(Bottleneck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet101_mutual():
    """ return a ResNet 101 object
    """
    return ResNetLargeBert4(Bottleneck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])
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
class xrf_UNet(nn.Module):
    def __init__(self, num_classes=55):
        super(xrf_UNet, self).__init__()

        self.encoder1 = self.conv_block(270, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv1d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        if dec4.size(2) != enc4.size(2):
            diff = enc4.size(2) - dec4.size(2)
            dec4 = nn.functional.pad(dec4, (diff // 2, diff - diff // 2))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.size(2) != enc3.size(2):
            diff = enc3.size(2) - dec3.size(2)
            dec3 = nn.functional.pad(dec3, (diff // 2, diff - diff // 2))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.size(2) != enc2.size(2):
            diff = enc2.size(2) - dec2.size(2)
            dec2 = nn.functional.pad(dec2, (diff // 2, diff - diff // 2))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.size(2) != enc1.size(2):
            diff = enc1.size(2) - dec1.size(2)
            dec1 = nn.functional.pad(dec1, (diff // 2, diff - diff // 2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)   
class UNet1D(nn.Module):
    def __init__(self, in_channels=270, out_channels=1):
        super(UNet1D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool1d(kernel_size=2, stride=2)(enc1))
        enc3 = self.encoder3(nn.MaxPool1d(kernel_size=2, stride=2)(enc2))
        enc4 = self.encoder4(nn.MaxPool1d(kernel_size=2, stride=2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool1d(kernel_size=2, stride=2)(enc4))
        
        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

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

        
        
# class Widar_ResNet(nn.Module):
#     def __init__(self, ResBlock, layer_list, num_classes):
#         super(Widar_ResNet, self).__init__()
#         self.reshape = nn.Sequential(
#             nn.ConvTranspose2d(22,3,7,stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(3,3,kernel_size=7,stride=1),
#             nn.ReLU()
#         )
#         self.in_channels = 64
        
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
#         self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
#         self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
#         self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
#     def forward(self, x):
#         x = self.reshape(x)
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         x = self.max_pool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
        
#         return x
        
#     def _make_layer(self, ResBlock, blocks, planes, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != planes * ResBlock.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * ResBlock.expansion)
#             )

#         layers = []
#         layers.append(ResBlock(self.in_channels, planes, stride, downsample))
#         self.in_channels = planes * ResBlock.expansion

#         for i in range(1, blocks):
#             layers.append(ResBlock(self.in_channels, planes))

#         return nn.Sequential(*layers)

    
# def Widar_ResNet18(num_classes):
#     return Widar_ResNet(Block, [2,2,2,2], num_classes = num_classes)
# def Widar_ResNet50(num_classes):
#     return Widar_ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes)
    
# def Widar_ResNet101(num_classes):
#     return Widar_ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes)

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

''' MLP '''
class MLP(nn.Module):
    # def __init__(self, channel, num_classes):
    #     super(MLP, self).__init__()
    #     self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
    #     self.fc_2 = nn.Linear(128, 128)
    #     self.fc_3 = nn.Linear(128, num_classes)

    # def forward(self, x):
    #     out = x.view(x.size(0), -1)
    #     out = F.relu(self.fc_1(out))
    #     out = F.relu(self.fc_2(out))
    #     out = self.fc_3(out)
    #     return out
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

''' resnet18_widar '''
class resnet18_widar(nn.Module):
# class Widar_CNN_GRU(nn.Module):
    def __init__(self,num_classes):
        super(resnet18_widar,self).__init__()
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
''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNetGAP(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNetGAP, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feat[0], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
#         super(BasicBlock, self).__init__()
#         self.norm = norm
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
#         super(Bottleneck, self).__init__()
#         self.norm = norm
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.norm = norm

#         self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.classifier = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self.norm))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out


# class ResNetImageNet(nn.Module):
#     def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
#         super(ResNetImageNet, self).__init__()
#         self.in_planes = 64
#         self.norm = norm

#         self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self.norm))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         # out = F.avg_pool2d(out, 4)
#         # out = out.view(out.size(0), -1)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out


# def ResNet18BN(channel, num_classes):
#     return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

# def ResNet18(channel, num_classes):
#     return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

# def ResNet34(channel, num_classes):
#     return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

# def ResNet50(channel, num_classes):
#     return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

# def ResNet101(channel, num_classes):
#     return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

# def ResNet152(channel, num_classes):
#     return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

# def ResNet18ImageNet(channel, num_classes):
#     return ResNetImageNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

# def ResNet6ImageNet(channel, num_classes):
#     return ResNetImageNet(BasicBlock, [1,1,1,1], channel=channel, num_classes=num_classes)
