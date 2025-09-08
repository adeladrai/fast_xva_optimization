import torch

def pinball(yhat: torch.Tensor, y: torch.Tensor, q: float | torch.Tensor):
    # yhat, y: broadcastable; q can be scalar or vector
    e = y - yhat
    if not torch.is_tensor(q):
        q = torch.tensor(q, dtype=e.dtype, device=e.device)
    return torch.mean(torch.maximum(q * e, (q - 1) * e))