# LibContinual-GEM
Coursework (GEM, LCL8) of Introduction to Machine Learning, Software Institute, Nanjng University

# notes
> 比较混乱, 把能想到的先全都记下来.

finetune都是false, 所以这里忽略finetune

# records
## 202311271644
```shell
(base) *[main][~/work/LibContinual-GEM]$ python3 ./run_trainer.py
/home/featurize/work/LibContinual-GEM/core/model/backbone/__init__.py:16: SyntaxWarning: str indices must be integers or slices, not str; perhaps you missed a comma?
  emb_func = eval(config["backbone"['name']])(**kwargs)
{'data_root': '/home/featurize/work/data/cifar10', 'image_size': 32, 'pin_memory': False, 'augment': True, 'workers': 8, 'device_ids': 0, 'n_gpu': 1, 'seed': 1993, 'deterministic': True, 'epoch': 10, 'batch_size': 128, 'val_per_epoch': 10, 'optimzer': {'name': 'SGD', 'kwargs': {'lr': 0.1}}, 'lr_scheduler': {'name': 'StepLR', 'kwargs': {'gamma': 0.5, 'step_size': 10}}, 'warmup': 3, 'includes': ['headers/data.yaml', 'headers/device.yaml', 'headers/model.yaml', 'headers/optimizer.yaml', 'backbones/resnet12.yaml'], 'save_path': './', 'init_cls_num': 2, 'inc_cls_num': 2, 'task_num': 5, 'optimizer': {'name': 'SGD', 'kwargs': {'lr': 0.1}}, 'backbone': {'name': 'resnet18', 'kwargs': {'num_classes': 10, 'args': {'dataset': 'cifar10'}}}, 'buffer': {'name': 'LinearBuffer', 'kwargs': {'buffer_size': 500, 'batch_size': 32, 'strategy': 'random'}}, 'classifier': {'name': 'LUCIR', 'kwargs': {'num_class': 10, 'feat_dim': 512, 'init_cls_num': 2, 'inc_cls_num': 2, 'dist': 0.5, 'lamda': 10, 'K': 2, 'lw_mr': 1}}, 'rank': 0}
LUCIR(
  (backbone): ResNet(
    (conv1): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (classifier): CosineLinear()
  (loss_fn): CrossEntropyLoss()
)
Trainable params in the model: 11169857
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    initial_lr: 0.1
    lr: 0.03333333333333333
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
================Task 0 Start!================
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    initial_lr: 0.1
    lr: 0.03333333333333333
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)
================Task 0 Training!================
The training samples number: 10000
learning rate: [0.03333333333333333]
================ Train on the train set ================
  4%|█████▊                                                                                                                                    | 13/312 [00:17<05:45,  1.16s/it]  4%|█████▊                                                                                                                                    | 13/312 [00:18<06:57,  1.40s/it]
Traceback (most recent call last):
  File "/home/featurize/work/LibContinual-GEM/./run_trainer.py", line 23, in <module>
    main(0, config)
  File "/home/featurize/work/LibContinual-GEM/./run_trainer.py", line 14, in main
    trainer.train_loop()
  File "/home/featurize/work/LibContinual-GEM/core/trainer.py", line 251, in train_loop
    train_meter = self._train(epoch_idx, dataloader)
  File "/home/featurize/work/LibContinual-GEM/core/trainer.py", line 289, in _train
    for batch_idx, batch in enumerate(dataloader):
  File "/environment/miniconda3/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 633, in __next__
    data = self._next_data()
  File "/environment/miniconda3/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 677, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/environment/miniconda3/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/environment/miniconda3/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/featurize/work/LibContinual-GEM/core/data/dataset.py", line 57, in __getitem__
    image = PIL.Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
  File "/environment/miniconda3/lib/python3.10/site-packages/PIL/Image.py", line 3140, in open
    prefix = fp.read(16)
KeyboardInterrupt

```

# CIFAR100
```python
print("i: {}, task_idx: {}".format(i, task_idx))
```
通过这行看到, 每2500个循环是一个task. 这和论文是匹配的.