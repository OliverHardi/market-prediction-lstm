import torch
from modules import constants

def quantile_loss(preds, target):
    """
        preds: [batch_size, 3] (num quantiles)
        target: [batch_size]
    """

    losses = []
    for i, q in enumerate(constants.QUANTILES):
        errors = target - preds[:, i]
        loss = torch.max((q-1)*errors, q*errors) # pinball loss/quantile loss
        losses.append(loss.unsqueeze(1))
    
    return torch.mean(torch.cat(losses, dim=1))
