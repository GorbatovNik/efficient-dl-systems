"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn


@torch.compile
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward."""
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    normalized = x * rsqrt
    scale = 1.0 + weight.float()
    output = normalized * scale
    return output.to(input_dtype), rsqrt


@torch.compile
def rmsnorm_backward(grad_output, x, weight, rsqrt):
    """Zero-Centered RMSNorm backward."""
    grad_output = grad_output.float()
    x = x.float()
    scale = 1.0 + weight.float()

    normalized = x * rsqrt

    grad_weight = (grad_output * normalized).sum(dim=tuple(range(grad_output.ndim - 1)))

    grad_normalized = grad_output * scale

    D = x.shape[-1]
    grad_x = rsqrt * (grad_normalized - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True))

    return grad_x, grad_weight


class RMSNormFunction(torch.autograd.Function):
    """
    Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        output, rsqrt = rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, rsqrt)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rsqrt = ctx.saved_tensors
        grad_x, grad_weight = rmsnorm_backward(grad_output, x, weight, rsqrt)
        return grad_x.to(x.dtype), grad_weight, None


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
