import os
import PIL
import torch
import random
import pprint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class ContinualDatasets:
    def __init__(
            self, mode, task_num, data_root, init_cls_num, 
            inc_cls_num=0, 
            cls_map=None, 
            trfms=None,
            samples_per_task=0,
            epoch=0,
            data=None, 
            n_inputs=0, 
        ):
        self.mode = mode
        self.task_num = task_num
        self.init_cls_num = init_cls_num
        self.inc_cls_num = inc_cls_num
        self.data_root = data_root
        self.cls_map = cls_map
        self.trfms = trfms
        self.dataloaders = []
        # =========== GEM ============
        self.samples_per_task = samples_per_task
        self.epoch = epoch
        self.data = data
        self.n_inputs = n_inputs
        # =========== GEM ============
        if self.samples_per_task == 0:
            self.create_loaders()
        else:
            self.batch_size = 10
            self.create_loaders(model="GEM")


    def create_loaders(self, model=None):
        for task_idx in range(self.task_num):
            start_idx = 0 if task_idx == 0 else (self.init_cls_num + (task_idx - 1) * self.inc_cls_num)
            end_idx = start_idx + (self.init_cls_num if task_idx ==0 else self.inc_cls_num)
            self.dataloaders.append(
                DataLoader(
                    SingleDataset(
                        self.data_root, 
                        self.mode, 
                        start_idx, 
                        end_idx, 
                        self.cls_map, 
                        self.trfms
                    ),
                    shuffle = True,
                    batch_size=32,
                    drop_last=True
                ) if model != "GEM" else
                DataLoader(
                    SingleContinuum(
                        data=self.data, 
                        batch=self.batch_size, 
                        task_idx=task_idx
                    ),
                    shuffle=True,
                    batch_size=self.batch_size,
                    drop_last=True
                )
            )    

    def get_loader(self, task_idx):
        assert task_idx >= 0 and task_idx < self.task_num
        return self.dataloaders[task_idx] if self.mode == 'train' else self.dataloaders[:task_idx + 1]
        

# SingleDataset指的是每个任务的数据集
class SingleDataset(Dataset):
    def __init__(
            self, data_root, mode, start_idx, end_idx, 
            cls_map=None, 
            trfms=None
        ):
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

    def _init_datalist(self, model=None):
        imgs, labels = [], []
        for id in range(self.start_idx, self.end_idx):
            img_list = [self.cls_map[id] + '/' + pic_path for pic_path in os.listdir(os.path.join(self.data_root, self.mode, self.cls_map[id]))]
            imgs.extend(img_list)
            labels.extend([id for _ in range(len(img_list))])
        return imgs, labels


# ==================== GEM ==================== # added by @wct
class SingleContinuumTimesEpoch(Dataset): 
    def __init__(self, data, batch, samples_per_task, epoch, task_idx) -> None:
        super().__init__()
        self.data = data
        self.batch = batch
        self.samples_per_task = samples_per_task
        self.epoch = epoch
        self.task_idx = task_idx
        self._init_datalist()

    def __getitem__(self, index): 
        j = torch.LongTensor([])
        return (
            self.data[self.task_idx][1][j], 
            self.data[self.task_idx][2][j]
        )

    def __len__(self,): return self.length

    def _init_datalist(self):
        data = self.data
        task_idx = self.task_idx
        samples_per_task = self.samples_per_task
        epoch = self.epoch

        N = data[task_idx][1].size(0)
        n = N if samples_per_task <= 0 else min(samples_per_task, N)
        p = torch.randperm(N)[0:n]
            

        # sample_permulations 总共有10个任务，每个任务是一个长度为 2500 的随机排列
        self.permutation = []
        for _ in range(epoch): # 10
            task_p = [[task_idx, i] for i in p] # 2500
            random.shuffle(task_p) # 随机打乱列表的元素顺序
            self.permutation += task_p

        self.length = len(self.permutation)

class SingleContinuum(Dataset): 
    def __init__(self, data, batch, task_idx) -> None:
        super().__init__()
        self.data = data
        self.batch = batch
        self.task_idx = task_idx
        self._init_datalist()

    def __getitem__(self, index): 
        return (
            self.data[self.task_idx][1][index], 
            self.data[self.task_idx][2][index]
        )

    def __len__(self,): return self.length

    # data 是长度为 task_num=20 的数组, 每个元素是长度为3的数组, 其中第二个和第三个元素分别是数据和标签, 长度都是2500
    def _init_datalist(self):
        task_idx = self.task_idx
        N = self.data[task_idx][1].size(0)
        p = torch.randperm(N) # p 是一个长度为 2500 的随机排列
        self.length = len(p)

class Continuum:
    def __init__(
            self, data, 
            batch_size, # added by @wct
            samples_per_task, # added by @wct
            epoch, # added by @wct
            task_num # added by @wct
        ):
        self.data = data
        self.batch_size = batch_size  # batch_size 是每次传递给模型的样本数目
        sample_permutations = [] # 生成任务数据的随机排列

        for task_idx in range(task_num):
            N = data[task_idx][1].size(0)
            n = N if samples_per_task <= 0 else min(samples_per_task, N)
            p = torch.randperm(N)[0:n]
            sample_permutations.append(p)

        # sample_permulations 总共有10个任务，每个任务是一个长度为 2500 的随机排列
        self.permutation = []
        for task_idx in range(task_num): # 10
            for _ in range(epoch): # 10
                task_p = [[task_idx, i] for i in sample_permutations[task_idx]] # 2500
                random.shuffle(task_p) # 随机打乱列表的元素顺序
                self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self): return self

    def next(self): return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            task_idx = self.permutation[self.current][0]
            j = []
            i = 0
            while (
                ((self.current + i) < self.length) and
                (self.permutation[self.current + i][0] == task_idx) and
                (i < self.batch_size)
            ):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            return self.data[task_idx][1][j], task_idx, self.data[task_idx][2][j]