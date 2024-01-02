# migration form BasicBlack4GEM to  BasicBlock
## previous params
- in_planes -> inplanes  输入维度
- planes -> planes  输出维度
- stride -> stride 卷积核滑动时的步长
## newly added params
- downSample:  下采样器
  ``` python
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
  ```
  这个部分GEM直接写在Block里，助教的代码则是在ResNet里初始化并且通过downSample参数传给Block
- groups: 只能是1 采用默认值
- base_width: 只能是64 采用默认值
- dialation: 暂时还没看到使用
- norm_layer: 如果什么都不填，就会初始化为 `nn.BatchNorm2d(planes)` 正好就是GEM需要的layers

# migration from ResNet4GEM to Resnet
## previous params
- num_blocks -> layers 每个阶段包含的残差块数量，都是[2,2,2,2]
- nclasses -> num_classes 这个值在助教的代码中没有被用到，在GEM中作为最后Linear层的输出维度，应该是最后可能有哪些类别？
- 
## added params
- inplanes: 原来的框架直接赋值64，GEM直接赋值20
### 迁移时一些不好合并的差别
- 助教给出的ResNet中有这样一段：
  ``` python
    self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
  ```
  这比GEM多了一层ReLU
  GEM的代码是：
  ``` python
    self.conv1 = conv3x3(3, nf * 1)
    self.bn1 = nn.BatchNorm2d(nf * 1)
  ```
- 助教给出的代码添加了一个池化层
  ``` python
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
  ```
  GEM多了一个Linear层,并且通过内置的avg_pool2d来调用池化函数,助教的代码没有Linear，但是通过 features = torch.flatten(pooled, 1)进行了降维?
  ``` python 
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
  ```
