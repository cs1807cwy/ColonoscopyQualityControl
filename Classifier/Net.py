import torch
import torch.nn as nn

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


class Bottleneck(nn.Module):
    expansion = 4  # 通道数扩大倍数

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)  # 1*1卷积
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # 3*3卷积
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1,
                               bias=False)  # 1*1卷积
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x  # 保存残差连接输入
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out = self.conv1(x)  # 1*1卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 3*3卷积
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1*1卷积
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block_type, block_num, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)  # H/2 W/2 C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/2 W/2 C:64
        self.layer1 = self._make_layer(block_type, 64, block_num[0], 1)  # H/2 W/2 C:256
        self.layer2 = self._make_layer(block_type, 128, block_num[1], 2)  # H/2 W/2 C:512
        self.layer3 = self._make_layer(block_type, 256, block_num[2], 2)  # H/2 W/2 C:1024
        self.layer4 = self._make_layer(block_type, 512, block_num[3], 2)  # H/2 W/2 C:2048

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)  # 全连接层

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block_type, out_channels, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block_type.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block_type.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * block_type.expansion)
            )
        layers = [block_type(self.in_channels, out_channels, stride, down_sample)]
        self.in_channels = out_channels * block_type.expansion
        for _ in range(1, block_num):
            layers.append(block_type(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet50(ResNet):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class ResNet101(ResNet):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


class ResNet152(ResNet):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test_resnet50():
    net = ResNet50(3)
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y.size())
    print(net)


if __name__ == "__main__":
    test_resnet50()
