import pickle

from torchvision import transforms
from .augments import *
import os
import numpy as np
from .dataset import ContinualDatasets
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

def get_dataloader(config, mode, cls_map=None, is_binary=False):
    '''
    Initialize the dataloaders for Continual Learning.

    Args:
        config (dict): Parsed config dict.
        mode (string): 'train' or 'test'.
        cls_map (dict): record the map between class and labels.
    
    Returns:
        Dataloaders (list): a list of dataloaders
    '''

    task_num = config['task_num']
    init_cls_num = config['init_cls_num']
    inc_cls_num = config['inc_cls_num']

    data_root = config['data_root']

    trfms_list = get_augment_method(config, mode)
    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    trfms = transforms.Compose(trfms_list)

    if is_binary: # added by @wct
        return None
    else:
        if cls_map is None:
            cls_list = os.listdir(os.path.join(data_root, mode))
            perm = np.random.permutation(len(cls_list))
            cls_map = dict()
            for label, ori_label in enumerate(perm):
                cls_map[label] = cls_list[ori_label]
        return ContinualDatasets(
            mode, 
            task_num, 
            init_cls_num, 
            inc_cls_num, 
            data_root, 
            cls_map, 
            trfms
        )

# added by @ycy
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict    


def load_datasets(config):
    data_root = config['data_root']
    cifar100_train = unpickle(os.path.join(data_root, "train"))
    cifar100_test = unpickle(os.path.join(data_root, "test"))

    x_tr = torch.from_numpy(cifar100_train[b'data']) #训练集
    y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
    x_te = torch.from_numpy(cifar100_test[b'data'])
    y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

    torch.manual_seed(config['seed'])
    
    x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0  # 255是图片的像素值范围，将其缩小到 1 - 0
    x_te = x_te.float().view(x_te.size(0), -1) / 255.0

    cpt = int(100 / config['task_num'])

    tasks_tr = []
    tasks_te = []

    for t in range(config['task_num']):
        c1 = t * cpt
        c2 = (t + 1) * cpt
        i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
        i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
        tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
        tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])


    d_tr = tasks_tr # 用于训练的数据集
    d_te = tasks_te # 用于测试的数据集
    n_inputs = d_tr[0][1].size(1) # 输入特征的数量
    n_outputs = 0 # 输出类别的数量
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max().item())
        n_outputs = max(n_outputs, d_te[i][2].max().item())
    print("task num: ", len(d_tr))        
    return d_tr, d_te, n_inputs, n_outputs + 1