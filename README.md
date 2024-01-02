# LibContinual

A framework of Continual Learning

* 代码中的LUCIR、LWF的复现样例可以跑通，ICARL的复现样例未跑通
  * 可以参考LUCIR、LWF的复现，去复现GEM
    * 善用代码搜索功能，对比复现代码和论文代码
* 跑起来LCL框架需要修改的地方：
  * ./config/xxx.yaml里的data_root文件位置
  * ./run_trainer.py里main文件的Config
* 感想
  * 看懂GEM的代码流程对复现最重要,分清楚什么时候是训练，什么时候是测试，什么时候是前后向传播。LCL框架并不复杂
  * 有GEM现成代码后，大部分参数移植到框架里时需要魔改/删除


要求的复现环境配置

* 代码环境：Python3.7 及以上，PyTorch 库以及其他运行代码需要的库
* 数据集：CIFAR10、CIFAR100

# 复现需要做的事

（不全面，为必要条件）

1. 新增文件：选定好一个方法后，

   1. 在./config路径下新增一个.yaml文件用来满足需要的**参数设置**. 
   2. 在./model/replay 或者 ./model/下新增一个.py文件用来**实现训练算法**. 

2. 参数设置：

   1. 在 configuration/config.py （继承了./config）中设置实验所需参数，推荐复制样例后修改。比如：

      ```
      数据集 dataset：CIFAR10、CIFAR100
      数据集路径 dir：比如data_root= “/home/xyk/CIFAR100"
      保存旧样本的数量 memory_size，不使用内存需要将其置零
      模型 backbone：resnet18
      优化器相关参数：optimizer、lr_scheduler
      训练参数：训练 epoch 数目 n_epoch、学习率 lr、batch_size
      任务设置：init_cls_num，inc_cls_num 分别对应持续学习初始化的任务类别数和每个任务增量的类别数
      ```

3. 训练实现：对于.py文件，需要实现几个函数: <br>

   1. `def __init__()`:  用来**初始化各自算法需要的对象**

   2. `def observe(self, data)`:  **训练过程**中，面对到来的一个batch的样本完成**训练的损失计算以及参数更新**，**返回pred, acc, loss**:  预测结果，准确率，损失    <br>

      > 训练深度神经网络时，通常会将训练数据划分为多个 batch，并将每个 batch 依次输入到网络中进行训练。这样做的好处是可以利用高效的并行计算，提高训练的效率，并且可以在每个 batch 上进行梯度更新，从而更好地优化网络权重。
      >
      > 与 batch 相关的概念是 "epoch"。一个 epoch 表示将所有训练样本都经过一次前向传播和反向传播的过程。在一个 epoch 中，所有的训练样本都被用于更新网络的权重

   3. `def inference(self, data)`:   **推理过程**中，面对到来的一个batch的样本，完成预测，**返回pred, acc**   <br>

   4. `def before_task() / after_task()`:  <u>**可选**</u>，如果算法在**每个任务开始前后有额外的操作**，在这两个函数内完成   <br>

      1. before_task：负责在每个任务训练前，执行更换分类头、重置优化器、赋值变量等操作
      2. afer_task：在每个任务结束时更新内存中保存的样本（replay 模型）

4. 训练过程中需要不同的buffer以及更新策略，可以自己在'./cire/model/buffer'下仿照LinearBuffer新增文件，并反馈给助教.

   > "buffer" 是指用于存储和管理数据的缓冲区或缓存区域。它通常用于在训练过程中临时存储训练样本、特征向量、标签或其他相关数据。例如数据加载缓冲区、中间结果缓冲区




# 代码结构

1. data模块：'./core/data' **负责dataset的读取逻辑**，关于datasets所需要的transform暂时没有写好，大家在复现各自方法的时候，需要什么transform可以直接修改./core/data/dataloader.py，后续会逐渐完善. <br>

   1. 可能在./core/data/dataloader.py需要添加预处理transform逻辑

   > 数据集（dataset）的变换（transform）是指对原始数据集进行一系列的操作和转换，以生成经过预处理或增强的新数据集。这些操作可以包括数据的标准化、缩放、旋转、裁剪、翻转等，以及其他一些对数据进行处理的方法。

2. bakcbone模块：'./core/model/backbone' **负责backbone模型文件的定义**(不包含fc)，这里我是参考PyCIL(https://github.com/G-U-N/PyCIL).   建议大家在复现各自方法之前，先检查一下与论文代码中的模型结构是否一致。   <br>

   1. 都是resnet18，应该不需要修改

      > 在深度学习中，通常的网络架构是由 backbone 模型和fc全连接层（或其他类型的层）组成的。backbone 模型用于提取输入数据的特征表示，而fc全连接层则用于将这些特征映射到最终的输出结果。这种组合可以有效地进行特征提取和学习，以解决各种机器学习任务。
      >
      > backbone 模型通常由一系列的卷积层、池化层和激活函数组成，用于逐层提取和转换输入数据的特征。这些层级结构可以根据具体的网络架构而有所不同，但其目标都是通过层叠的非线性变换来捕捉输入数据的高级特征。

3. buffer模块： './core/model/buffer' 负责训练过程中**buffer的管理以及更新**。 目前只实现了LinearBuffer, 在每个任务开始前会把buffer样本与新样本拼接在一起.  buffer的更新策略，目前只支持了随机更新.  其他类型的Buffer以及更新策略后续会逐渐完善.  此外，Buffer的更新函数 def update的参数，大家在实现的时候可以先根据自己的需要随意设置，后续会考虑整合.  <br>

   1. 论文里没看到buffer，大概率不需要动

      > "buffer" 是指用于存储和管理数据样本的缓冲区或缓存区域。它通常用于在训练过程中临时存储训练样本、特征向量、标签或其他相关数据。例如数据加载缓冲区、中间结果缓冲区

4. logger模块：'./core/utils/logger.py' 负责训练过程中的**日志打印**。 此处选择直接hack 系统输出，因此大家在训练过程中不需要显示的调用logger.info等接口，  直接正常的print想要的信息，logger模块会自动的保存在日志文件中.  

   1. 大概率不需要动

5. trainer模块：'./core/trainer.py' **负责整个实验的流程控制**。 大家在复现各自方法的时候，如果流程上有暂时支持不了的，可以直接修改trainer.py来满足，并且可以反馈给我，后续我会对流程做进一步的完善.  <br>

   1. 可能需要动。负责流程控制

6. config模块：'./config/', 负责整个**训练过程的参数配置**。
   入口：run_trainer.py里的line：15填入各自方法对应的yaml配置文件路径。 为每个方法在./config/路径下新建一个*.yaml。 配置文件里面需要写入以下参数： <br>

   ```
   a/  includes:  仿照finetune.yaml照抄，用来填充一些默认的参数。   *.yaml里的参数会覆盖掉config/headers/里的参数  <br>
   b/  data_root: 所使用的数据集路径。数据集的摆放格式如下：
         data_root:  <br>
         | ---train  <br>
         | ------class1   <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class2  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class3  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ---test  <br>
         | ------class1  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class2  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
         | ------class3  <br>
         | ----------img1.jpg  <br>
         | ----------img2.jpg  <br>
   
   c/ save_path: log以及checkpoints存放路径，log文件存放在 save_path/xxx.log,  checkpoint保存功能还未完成.  <br>
   
   d/ init_cls_num, inc_cls_num, task_num: 第一个任务的类别个数、后续每个任务的类别个数、任务总数。 类别顺序是随机生成的 <br>
   
   e/ init_epoch, epoch:  分别对应持续学习初始化的任务类别数和每个任务增量的类别数，没设置init_epoch的情况下init_epoch = epoch  <br>
   
   f/ backbone:  参考finetune.yaml, 一般指明name即可， 其中args:datasets 是代码遗留问题，暂时先照抄，后续会修改掉.   <br>
   
   g/ optimizer, lr_scheduler:  可以仿照大家平常使用pytorch自带的optimizer与scheduler, 将名字与相应参数，仿照finetune.yaml的形式填入即可.   <br>
   
   h/ buffer:  与训练过程中使用的buffer相关，目前buffer的使用只支持将旧样本与新样本拼接在一起。buffer_size, batch_size, strategy： 旧样本数量，batch_size在linearBuffer下没用，strategy更新策略。
   
   i: classifier: name:对应各自实现的方法名，其他参数看各自需要什么，直接在里面加
   ```

   


# LibContinual-GEM
Coursework (GEM, LCL8) of Introduction to Machine Learning, Software Institute, Nanjng University


# CIFAR100 数据的细节
```python
print("i: {}, task_idx: {}".format(i, task_idx))
```
通过这行看到, 每2500个循环是一个task. 这和论文是匹配的.
[这篇文章](https://blog.csdn.net/nyist_yangguang/article/details/126077044)介绍了二进制的cifar100数据集.


# TODO
- [x] 能够运行代码
- [x] `n_task`变量转换成`task_num`, `t`变量转换成`task_idx`
- [x] 消除`life_experience()`函数
- [ ] 进一步地整合参数
- [ ] 重新考虑是否有办法不消除continuum, 而是让使用者感受不到它的存在, 比如说加到 `GEM.before_task()` 里面
- [ ] 将 `class GEM` 的 `__init__()` 函数和 `observe()` 函数中的数据存储部分抽象成RingBuffer类.
- [ ] 将 `class BasicBloc4GEM` 消除, 和上面的 `class BasicBlock` 合并. 类似地消除 `ResNet18`, 和上面的 `resnet18` 合并 (这一部分难度较大)


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

# 数据预处理

本次数据预处理工作主要改动了两个文件：

- 添加gem.yaml
- 修改dataloader.py



关于gem.yaml:

- 首先是yaml并没有严格的语法规则，可以理解为暂存常用变量的文件，所以yaml添加的都是在lcl-8-reproduction-main中的命令行参数（具体调用方法比如：n_task = config['task_num'],task_num为yaml文件中的声明task_num: 10）

- yaml中没有放入的参数：

  - 相关.pt文件路径，由于原项目中数据处理的逻辑整合在了一起，省去了文件之间相互调用.pt文件，故省略
  - cuda参数，后续移植可能遇到，由于只使用CIFAR数据集，默认为yes



关于dataloader.py:
- 原dataloader的逻辑不完善! 不完善! 不完善! 这就是为什么助教说

  >大家在复现各自方法的时候，需要什么transform可以直接修改./core/data/dataloader.py

  本来尝试将数据处理的逻辑嵌入到原本的dataloader.get_dataloader()中去,发现完全行不通(这也可能是为什么ICARL的复现样例未跑通)

  所以另建了一个get_data_in_gem的函数...

- 调取处理完的数据直接使用
  
  >dataloader.get_data_in_gem(config)

  该函数的具体返回值以及使用参照lcl-8-reproduction-main中main.py的217行