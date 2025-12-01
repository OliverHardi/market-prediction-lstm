import torch

def loss(outputs, y):
    p_logits = outputs[:, 0]     # raw logits (NOT sigmoid)
    e_pred   = outputs[:, 1]

    # targets
    p_true = y[:, 0]             # p_revert 0/1
    e_true = y[:, 1]             # e_return

    # binary cross entropy (with logits = more stable)
    loss_p = torch.nn.functional.binary_cross_entropy_with_logits(
        p_logits, p_true
    )

    # mask: only count e_return when p_revert = 1
    mask = p_true   # shape (batch,), 1 where revert, 0 otherwise

    mse_raw = (e_pred - e_true)**2
    mse_masked = (mse_raw * mask).sum() / (mask.sum() + 1e-6)

    # Optional scale multiplier to balance losses
    loss = loss_p + mse_masked * 10.0

    return loss, loss_p.item(), mse_masked.item()