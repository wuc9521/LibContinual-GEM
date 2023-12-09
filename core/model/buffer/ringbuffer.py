# 将GEM中的memory_data和memory_labs抽象出来, 作为一个RingBuffer类
import torch
class RingBuffer:
    def __init__(self, task_num, n_memories, n_inputs):
        self.memory_data = torch.FloatTensor(task_num, n_memories, n_inputs).cuda()
        self.memory_labs = torch.LongTensor(task_num, n_memories).cuda()
        self.mem_cnt = 0
        self.n_memories = n_memories

    def update(self, x, task_idx, y):
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[task_idx, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[task_idx, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[task_idx, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

    def get_memory_data(self, task_idx):
        return self.memory_data[task_idx]

    def get_memory_labels(self, task_idx):
        return self.memory_labs[task_idx]