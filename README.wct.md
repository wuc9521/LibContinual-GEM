# LibContinual-GEM
Coursework (GEM, LCL8) of Introduction to Machine Learning, Software Institute, Nanjng University

# notes
> 比较混乱, 把能想到的先全都记下来.

finetune都是false, 所以这里忽略finetune

# 数据部分的细节

## CIFAR100
```python
print("i: {}, task_idx: {}".format(i, task_idx))
```
通过这行看到, 每2500个循环是一个task. 这和论文是匹配的. 实际上,

- `task_num`: 20
- `task_idx`: 0, 1, ..., 19
- `samples_per_task`: 2500

## `Continuum`
LCL-8代码中, `Continuum` 是一个可以迭代的对象, 存储了长度为 `epoch_size * task_num * samples_per_task` 的数据.

在LibContinual中, 数据的存储方式如下:
- 每一个任务有一个 `dataloader`, 参数是 `task_idx`: `dataloader = self.train_loader.get_loader(task_idx)`
- 每一个 `dataloader` 有一个 `dataset`: `datasets = dataloader.dataset`

## `SingleDataset`

> 在数据处理部分, LibContinual 的实现要比 LCL-8 好很多.

- `SingleDataset` 是一个 `torch.utils.data.Dataset` 的子类, 用于存储一个任务的数据. 对于LibContinual中的数据, SingleDataset存储了一个任务的所有数据, 并且不重复.

- 原本的 `Continuum` 存储了所有任务的全部数据, 并且包括 `epoch` 产生的重复数据. 

- 这里新写一个 `SingleContinuum` , 同样是一个 `torch.utils.data.Dataset` 的子类, 用于一个任务的数据, 并且不重复.

# `nn.Module` 的细节
- 模型类是不需要知道所有的数据的, 因为 `model.train()` 是在循环中调用的, 所以每次只需要知道当前的数据就可以了.


# TODO
- [x] 能够运行代码
- [x] `n_task`变量转换成`task_num`, `t`变量转换成`task_idx`
- [x] 消除`life_experience()`函数
- [ ] 进一步地整合参数
- [ ] 重新考虑是否有办法不消除continuum, 而是让使用者感受不到它的存在, 比如说加到 `GEM.before_task()` 里面
- [ ] 将 `class GEM` 的 `__init__()` 函数和 `observe()` 函数中的数据存储部分抽象成RingBuffer类.
- [ ] 将 `class BasicBloc4GEM` 消除, 和上面的 `class BasicBlock` 合并. 类似地消除 `ResNet18`, 和上面的 `resnet18` 合并 (这一部分难度较大)


# Appendix

1. [这篇文章](https://blog.csdn.net/nyist_yangguang/article/details/126077044)介绍了二进制的cifar100数据集.

2. [这篇文章](https://zhuanlan.zhihu.com/p/557253923)介绍了 pytorch `nn.Module` 模块的使用方法.

3. [这个链接](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 是 `nn.Module` 的官方文档.


# 可圈可点

1. 充分利用类的overload, 完全不改变原有方法的调用.
