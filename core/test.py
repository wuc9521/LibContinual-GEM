from __future__ import print_function
import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t
    return n_tasks, changes


def confusion_matrix(result_t, result_a, fname=None):
    nt, changes = task_changes(result_t)
    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[changes]
    acc = result.diag()
    fin = result[nt - 1]
    bwt = result[nt - 1] - acc
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]
    print('Final Accuracy: %.4f' % fin.mean())
    print('Backward: %.4f' % bwt.mean())
    print('Forward:  %.4f' % fwt.mean())

    stats = {}
    # stats.append(acc.mean())
    stats['fin'] = fin.mean()
    stats['bwt'] = bwt.mean()
    stats['fwt'] = fwt.mean()

    return stats
