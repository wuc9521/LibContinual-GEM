import os
import pickle
import numpy as np
from torchvision import transforms
from .augments import *
from .dataset import ContinualDatasets
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]


def get_dataloader(config, mode, cls_map=None, model=None):
    task_num = config['task_num']
    data_root = config['data_root']
    init_cls_num = config['init_cls_num']
    inc_cls_num = config['inc_cls_num']
    trfms_list = get_augment_method(config, mode)
    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    trfms = transforms.Compose(trfms_list)

    if model == "GEM":
        assert init_cls_num == inc_cls_num
        d_tr, n_inputs = load_train_data(config, init_cls_num)
        return ContinualDatasets(
            mode, task_num, data_root, init_cls_num,
            data=d_tr, 
            epoch=config['epoch'],
            samples_per_task=config['samples_per_task'],
            n_inputs=n_inputs
        )
    
    if cls_map is None:
        if config['classifier']['name'] != "GEM": 
            cls_list = os.listdir(os.path.join(data_root, mode))
            perm = np.random.permutation(len(cls_list))
            cls_map = dict()
            for label, ori_label in enumerate(perm):
                cls_map[label] = cls_list[ori_label]
    return ContinualDatasets(
        mode, task_num, data_root, init_cls_num, 
        inc_cls_num, 
        cls_map, # TODO: solve class map in binary dataset
        trfms
    )



# added by @ycy
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    


def load_train_data(config, init_cls_num):
    tasks_tr = []
    cifar100_train = unpickle(os.path.join(config['data_root'], "train"))
    x_tr = torch.from_numpy(cifar100_train[b'data']) #训练集
    y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0  # 255是图片的像素值范围，将其缩小到 1 - 0
    
    torch.manual_seed(config['seed'])
    for task_idx in range(config['task_num']):
        c1 = task_idx * init_cls_num
        c2 = (task_idx + 1) * init_cls_num
        i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
        tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    d_tr = tasks_tr # 用于训练的数据集
    n_inputs = d_tr[0][1].size(1) # 输入特征的数量
    return d_tr, n_inputs


def load_test_data(config, init_cls_num):
    tasks_te = []
    cifar100_test = unpickle(os.path.join(config['data_root'], "test"))
    x_te = torch.from_numpy(cifar100_test[b'data'])
    y_te = torch.LongTensor(cifar100_test[b'fine_labels'])
    x_te = x_te.float().view(x_te.size(0), -1) / 255.0
    torch.manual_seed(config['seed'])
    init_cls_num = int(100 / config['task_num'])
    for task_idx in range(config['task_num']):
        c1 = task_idx * init_cls_num
        c2 = (task_idx + 1) * init_cls_num
        i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
        tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])
    d_te = tasks_te # 用于测试的数据集            
    n_outputs = 0 # 输出类别的数量
    d_tr, _ = load_train_data(config, init_cls_num)
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())  
    return d_te, n_outputs
