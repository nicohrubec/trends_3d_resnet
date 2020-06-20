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


def weighted_nae_npy(y_true, y_pred):
    weights = [.3, .175, .175, .175, .175]
    domain_losses = []

    score = 0.
    for i, weight in enumerate(weights):
        target_score = np.mean(np.sum(np.abs(y_true[:,i] - y_pred[:,i]), axis=0) / np.sum(y_true[:,i], axis=0))
        domain_losses.append(target_score)
        score += weight * target_score

    return score, domain_losses
