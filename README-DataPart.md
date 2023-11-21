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



