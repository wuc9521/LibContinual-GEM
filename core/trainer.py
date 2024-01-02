import datetime
import os
import sys
import time
import uuid

import torch
from tqdm import tqdm
from core.data import get_dataloader
from core.test import confusion_matrix
from core.utils import init_seed, AverageMeter, get_instance, GradualWarmupScheduler, count_parameters
from core.model.buffer import *
import core.model as arch
from core.model.buffer import *
from torch.utils.data import DataLoader
import numpy as np
from core.model.buffer import LinearBuffer, hearding_update, random_update
from core.utils import Logger, fmt_date_str, eval_tasks

from core.data.dataset import Continuum

class Trainer(object):
    """
    The Trainer.
    
    Build a trainer from config dict, set up optimizer, model, etc.
    """

    def __init__(self, rank, config):
        self.rank = rank
        self.config = config
        self.config['rank'] = rank
        self.distribute = self.config['n_gpu'] > 1  # 暂时不考虑分布式训练
        self.logger = self._init_logger(config)           
        self.device = self._init_device(config) 
        self.init_cls_num, self.inc_cls_num, self.task_num = self._init_data(config)
        self.is_binary = (config['classifier']['name'] == "GEM")

          # added by @wct: 如果是GEM模型, 使用的是二进制数据集
        # @wct: 这里的 _init_* 函数就相当于 Java 里的 new
        (
            self.train_loader,
            self.test_loader,
        ) = self._init_dataloader(config)

        self.model = self._init_model(config)  # modified by wct
        
        self.buffer = self._init_buffer(config)
        (
            self.init_epoch,
            self.inc_epoch,
            self.optimizer,
            self.scheduler,
        ) = self._init_optim(config)

        self.train_meter, self.test_meter = self._init_meter()

        self.val_per_epoch = config['val_per_epoch']

    def _init_logger(self, config, mode='train'):
        '''
        Init logger.

        Args:
            config (dict): Parsed config file.

        Returns:
            logger (Logger)
        '''

        save_path = config['save_path']
        log_path = os.path.join(save_path, "log")
        if not os.path.isdir(log_path):
            os.mkdir(log_path) 
        log_prefix = config['classifier']['name'] + "-" + config['backbone']['name'] + "-" + f"epoch{config['epoch']}"
        log_file = os.path.join(log_path, "{}-{}.log".format(log_prefix, fmt_date_str()))
        logger = Logger(log_file)

        # hack sys.stdout
        sys.stdout = logger

        return logger

    def _init_device(self, config):
        """"
        Init the devices from the config.
        
        Args:
            config(dict): Parsed config file.
            
        Returns:
            device: a device.
        """
        init_seed(config['seed'], config['deterministic'])
        device = torch.device("cuda:{}".format(config['device_ids']))
        return device


    def _init_files(self, config): 
        pass

    def _init_writer(self, config):
        pass

    def _init_meter(self, ):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
        )

        test_meter = [AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
        ) for _ in range(self.task_num)]

        return train_meter, test_meter

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler.
        """
        params_dict_list = {"params": self.model.parameters()}
    
        optimizer = get_instance(
            torch.optim, "optimizer", config, params=self.model.parameters()
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details
        init_epoch = config['init_epoch'] if 'init_epoch' in config.keys() else config['epoch']
        return init_epoch, config['epoch'], optimizer, scheduler

    def _init_data(self, config):
        return config['init_cls_num'], config['inc_cls_num'], config['task_num']

    # @wct: _init_model() 函数目前看起来没问题了.
    def _init_model(self, config):
        """
        Init model (backbone + classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        backbone = get_instance(arch, "backbone", config)
           # added by @wct
         # @wct: 这里添加了两个成员变量, 反正很丑就是了

         # 输入特征的数量
         # 输出类别的数量

        n_inputs = self.train_loader.n_inputs if hasattr(self.train_loader, "n_inputs") else 0
        n_outputs = self.train_loader.n_outputs if hasattr(self.train_loader, "n_outputs") else 0
            #load_datasets(config)) if config['classifier']['name'] == "GEM" else (None, None, None, None)
        # @wct: 这里写的很丑, 后面改

        dic = {
            "backbone": backbone, 
            "device": self.device,
            "n_inputs": n_inputs, # added by @wct
            "n_outputs": n_outputs, # added by @wct
            "task_num": self.task_num, # added by @wct
        }
        model = get_instance(arch, "classifier", config, **dic) 
        print(model)
        print("Trainable params in the model: {}".format(count_parameters(model)))
        return model.to(self.device) if config['classifier']['name'] != "GEM" else model # added by @wct
    
    def _init_dataloader(self, config):
        '''
        Init DataLoader

        Args:
            config (dict): Parsed config file.

        Returns:
            train_loaders (list): Each task's train dataloader.
            test_loaders (list): Each task's test dataloader.
        '''
        train_loaders = get_dataloader(
            config, 
            "train", 
            is_binary=self.is_binary
        )
        test_loaders = get_dataloader(
            config, 
            "test", 
            cls_map=(train_loaders.cls_map if config['classifier']['name'] != "GEM" else None), 
            is_binary=self.is_binary
        )
        return train_loaders, test_loaders
    
    def _init_buffer(self, config):
        '''
        Init Buffer
        
        Args:
            config (dict): Parsed config file.

        Returns:
            buffer (Buffer): a buffer for old samples.
        '''
        return get_instance(arch, "buffer", config)

    # 主要的函数
    def train_loop(self):
        """
        The norm train loop:  before_task, train, test, after_task
        """
        if self.config['classifier']['name'] == "GEM": # this "if-else" is added by @wct
            # set up continuum
            dataloader = self.train_loader

            # load model
            self.model.cuda()
            result_a = [] # modified by @wct: 这里原本是life_experience()函数
            result_t = []
            log_every = self.config['log_every']
            current_task = 0

            time_start = time.time()

            for task_idx in range(self.task_num):
                # print("================Task {} Start!================".format(task_idx))
                if hasattr(self.model, 'before_task'):
                    self.model.before_task(
                        task_idx, 
                    )


            for i in tqdm(range(self.task_num),"task_iteration:"):

                x, task_idx, y = dataloader.get_loader(i)
                print("================Task {} Start!================".format(task_idx))
                result_a.append(eval_tasks(self.model, self.test_loader))
                result_t.append(current_task)
                current_task = task_idx

                for j in tqdm(range(0, len(y), self.val_per_epoch), "interTask_iteration:"):
                    _x = x[j : j+ self.val_per_epoch]
                    _y = y[j : j+ self.val_per_epoch]

                    v_x = _x.view(_x.size(0), -1).cuda()
                    v_y = _y.long().cuda()

                    self.model.train()
                    self.model.observe(v_x, task_idx, v_y)

            time_end = time.time()

            result_a.append(eval_tasks(self.model, self.test_loader))
            result_t.append(current_task)

            (result_t, result_a) = (torch.Tensor(result_t), torch.Tensor(result_a))

            timespent = time_end - time_start


            stats = confusion_matrix(result_t, result_a)
            print(stats.get('fin'))
            print(stats.get('fwt'))
            print(stats.get('bwt'))
            one_liner = ''.join(["%.3f" % stat for stat in stats])
            print('result' + ': ' + one_liner + ' # ' + str(timespent))
            print("result_t: {}".format(result_t))
            print("result_a: {}".format(result_a))
        
        else:
            for task_idx in range(self.task_num):
                print("================Task {} Start!================".format(task_idx))
                if hasattr(self.model, 'before_task'):
                    self.model.before_task(
                        task_idx, 
                        self.buffer, 
                        self.train_loader.get_loader(task_idx), 
                        self.test_loader.get_loader(task_idx)
                    )
                (
                    _, __,
                    self.optimizer,
                    self.scheduler,
                ) = self._init_optim(self.config)

                self.buffer.total_classes += self.init_cls_num if task_idx == 0 else self.inc_cls_num

                dataloader = self.train_loader.get_loader(task_idx)

                if isinstance(self.buffer, LinearBuffer) and task_idx != 0:
                    datasets = dataloader.dataset
                    datasets.images.extend(self.buffer.images)
                    datasets.labels.extend(self.buffer.labels)
                    dataloader = DataLoader(
                        datasets,
                        shuffle = True,
                        batch_size = self.config['batch_size'],
                        drop_last = True
                    )
                
                print("================Task {} Training!================".format(task_idx))
                print("The training samples number: {}".format(len(dataloader.dataset)))

                best_acc = 0.
                for epoch_idx in range(self.init_epoch if task_idx == 0 else self.inc_epoch):
                    print("learning rate: {}".format(self.scheduler.get_last_lr()))
                    print("================ Train on the train set ================")
                    train_meter = self._train(epoch_idx, dataloader)
                    print("Epoch [{}/{}] |\tLoss: {:.3f} \tAverage Acc: {:.3f} ".format(epoch_idx, self.init_epoch if task_idx == 0 else self.inc_epoch, train_meter.avg('loss'), train_meter.avg("acc1")))

                    if (epoch_idx + 1) % self.val_per_epoch == 0 or (epoch_idx + 1) == self.inc_epoch:
                        print("================ Test on the test set ================")
                        test_acc = self._validate(task_idx)
                        best_acc = max(test_acc["avg_acc"], best_acc)
                        print(" * Average Acc: {:.3f} Best acc {:.3f}".format(test_acc["avg_acc"], best_acc))
                        print(" Per-Task Acc:{}".format(test_acc['per_task_acc']))
                
                    self.scheduler.step()

                if hasattr(self.model, 'after_task'):
                    self.model.after_task(task_idx, self.buffer, self.train_loader.get_loader(task_idx), self.test_loader.get_loader(task_idx))

                if self.buffer.strategy == 'herding':
                    hearding_update(self.train_loader.get_loader(task_idx).dataset, self.buffer, self.model.backbone, self.device)
                elif self.buffer.strategy == 'random':
                    random_update(self.train_loader.get_loader(task_idx).dataset, self.buffer)



    def _train(self, epoch_idx, dataloader):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index

        Returns:
            dict:  {"avg_acc": float}
        """
        self.model.train()
        meter = self.train_meter
        meter.reset()
        

        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                output, acc, loss = self.model.observe(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
                meter.update("acc1", acc)
        return meter



    def _validate(self, task_idx):
        dataloaders = self.test_loader.get_loader(task_idx)

        self.model.eval()
        meter = self.test_meter
        
        per_task_acc = []
        with torch.no_grad():
            for t, dataloader in enumerate(dataloaders):
                meter[t].reset()
                for batch_idx, batch in enumerate(dataloader):
                    output, acc = self.model.inference(batch)
                    meter[t].update("acc1", acc)

                per_task_acc.append(round(meter[t].avg("acc1"), 2))
        
        return {"avg_acc" : np.mean(per_task_acc), "per_task_acc" : per_task_acc}
