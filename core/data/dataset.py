from torch.utils.data import Dataset
import torch
import PIL
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from tqdm import trange



class ContinualDatasets:
    def __init__(self, mode, task_num, init_cls_num, inc_cls_num, data_root, cls_map, trfms):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.dataloaders = []

        self.create_loaders()

    def create_loaders(self):
        for i in range(self.task_num):
            start_idx = 0 if i == 0 else (self.init_cls_num + (i-1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if i ==0 else self.inc_cls_num)
            self.dataloaders.append(DataLoader(
                SingleDataseat(self.data_root, self.mode, self.cls_map, start_idx, end_idx, self.trfms),
                shuffle = True,
                batch_size = 32,
                drop_last = True
            ))

    def get_loader(self, task_idx):
        assert task_idx >= 0 and task_idx < self.task_num
        if self.mode == 'train':
            return self.dataloaders[task_idx]
        else:
            return self.dataloaders[:task_idx+1]
        


    
class SingleDataseat(Dataset):
    def __init__(self, data_root, mode, cls_map, start_idx, end_idx, trfms):
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.cls_map = cls_map
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.trfms = trfms
        self.images, self.labels = self._init_datalist()

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = PIL.Image.open(os.path.join(self.data_root, self.mode, img_path)).convert("RGB")
        image = self.trfms(image)
        return {"image": image, "label": label}
    
    def __len__(self,):
        return len(self.labels)

    def _init_datalist(self):
        imgs, labels = [], []
        for id in range(self.start_idx, self.end_idx):
            img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
            imgs.extend(img_list)
            labels.extend([id for _ in range(len(img_list))])

        return imgs, labels

    

class Continuum:
    """
    Continuum类, 用于迭代训练集中的每一个任务.

    Attributes:
        data: 包含有[训练集, 测试集, 输入的数量, 输出的数量, 训练集大小]的tuple.
        batch_size: 每个batch的大小.
        permutation: 一个list, 用于记录每个batch的数据的索引.
        length: permutation的长度.
        current: 当前迭代到的位置.
    """
    
    def __init__(
            self, 
            data, 
            batch_size, # added by @wct
            shuffle_tasks, # added by @wct
            samples_per_task, # added by @wct
            epoch, # added by @wct
            task_num # added by @wct
        ):
        print("===============================")
        print("epoch: ", epoch)
        print("===============================")
        self.data = data
        self.batch_size = batch_size  # batch_size 是每次传递给模型的样本数目
        task_permutation = torch.randperm(task_num).tolist() if shuffle_tasks == 'yes' else range(task_num)

        sample_permutations = []

        for t in trange(task_num, desc='Tasks', leave=True):
            N = data[t][1].size(0)
            n = N if samples_per_task <= 0 else min(samples_per_task, N)
            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        self.permutation = []
        for t in trange(task_num, desc='Tasks Permutation', leave=True):
            task_t = task_permutation[t]
            for _ in trange(epoch, desc='Epochs', leave=False):
                task_p = [[task_t, i] for i in sample_permutations[task_t]]
                random.shuffle(task_p)
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self): return self

    def next(self): return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            ti = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == ti) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[ti][1][j], ti, self.data[ti][2][j]