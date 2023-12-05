# LibContinual-GEM
Coursework (GEM, LCL8) of Introduction to Machine Learning, Software Institute, Nanjng University

# notes
> 比较混乱, 把能想到的先全都记下来.

finetune都是false, 所以这里忽略finetune


# CIFAR100 数据的细节
```python
print("i: {}, task_idx: {}".format(i, task_idx))
```
通过这行看到, 每2500个循环是一个task. 这和论文是匹配的.


# TODO
- [x] 能够运行代码
- [x] `n_task`变量转换成`task_num`, `t`变量转换成`task_idx`
- [x] 消除`life_experience()`函数
- [ ] 重新考虑是否有办法不消除continuum, 而是让使用者感受不到它的存在
- [ ] 将class GEM的 `__init__()` 函数和 `observe()` 函数中的数据存储部分抽象成RingBuffer类.
