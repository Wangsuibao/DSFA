'''
不同组件的代码块：
    1、inception
    2、Resnet
    3、单卷积块
    4、双卷积块
    5、

'''


class InceptionA(nn.Module):
    '''
        注意不同分支的数组大小不同，cat
    '''
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = nn.Conv2d

        self.branch1x1 = conv_block(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 24, kernel_size=1)
        self.branch5x5_2 = conv_block(24, 32, kernel_size=5, padding=2)  # 这个应用是pool实现

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 32, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(32, 32, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]  # 数组在维度上不等。
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        outputs = torch.cat(outputs,1)
        outputs = F.max_pool2d(outputs, kernel_size=3, stride=2, padding=1)
        return outputs


class ResnextBlock(nn.Module):
    '''分组卷积->并联链接->depthwise
    in_planes: 输入深度; cardinality=32:基数(分组数); bottleneck_width=4:每一个分组的输出深度
    垂向上是1*1卷积减少参数，纵向上是分组卷积形成类似并联+resnet
    in_channel(32) --> out_channel(64)
    '''
    expansion = 2  # 1*1卷积缩减的通道数
    def __init__(self, in_planes, cardinality=32, bottleneck_width=1, stride=1):
        super(ResnextBlock, self).__init__()
        group_width = cardinality * bottleneck_width  # 分组基数*每一组的输出数=总通道数  32*2 = 64
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        # in_channel=128,out_channel=128,kernel_size=3,stride=2,padding=1, groups=32
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, 
                                stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        # res链接的需要和分组卷积一致。
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width))

    def forward(self, x):
        # 卷积+bn+激活
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.max_pool2d(out,kernel_size=2,stride=2)
        out = F.relu(out)

        return out