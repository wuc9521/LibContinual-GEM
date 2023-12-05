import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import quadprog

# from .common import MLP, ResNet18
from core.model.backbone.resnet import ResNet18
        

# Auxiliary functions useful for GEM's inner optimization.
def compute_offsets(task, nc_per_task):
    offset1 = task * nc_per_task
    offset2 = (task + 1) * nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(nn.Module):
    def __init__(
            self, 
            backbone, # added by @wct
            device, # added by @wct
            memory_strength, # added by @wct
            n_memories, # added by @wct
            lr, # added by @wct
            n_inputs, # 输入特征的数量
            n_outputs, # 输出类别的数量
            task_num, # 任务的数量: 从设计的角度说, 实际上模型不应该知道任务的数量(?)
        ):
        super(GEM, self).__init__()
        self.margin = memory_strength

        self.net = ResNet18(n_outputs)
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = optim.SGD(self.parameters(), lr)
        self.n_memories = n_memories
        self.gpu = True

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(task_num, self.n_memories, n_inputs) # 作为buffer, 存储样本
        self.memory_labs = torch.LongTensor(task_num, self.n_memories) # 作为buffer, 存储标签
        self.memory_data = self.memory_data.cuda() # modified by @wct
        self.memory_labs = self.memory_labs.cuda() # modified by @wct

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), task_num).cuda() # modified by @wct

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = int(n_outputs / task_num)

    def before_task(self, task_idx):
        # 不如在before_task这个地方把Continuum的数据加载进来
        pass

    def forward(self, x, t):
        output = self.net(x)
        offset1 = int(t * self.nc_per_task)
        offset2 = int((t + 1) * self.nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, task_idx, y):
        # update memory
        if task_idx != self.old_task:
            self.observed_tasks.append(task_idx)
            self.old_task = task_idx

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[task_idx, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[task_idx, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[task_idx, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad() # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(
                    past_task, 
                    self.nc_per_task
                )
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(task_idx, self.nc_per_task)
        loss = self.ce(self.forward(x, task_idx)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, task_idx)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, task_idx].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, task_idx].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, task_idx],
                               self.grad_dims)
        self.opt.step()