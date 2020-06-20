import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175]).cuda()
    return torch.dot(torch.sum(torch.abs(inp - targ), axis=0) / torch.sum(targ, axis=0), W)
