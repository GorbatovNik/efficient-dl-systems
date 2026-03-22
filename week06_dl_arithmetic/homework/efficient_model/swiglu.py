"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    a_clamped = tl.minimum(a_row, 7.0)
    b_clamped = tl.maximum(tl.minimum(b_row, 7.0), -7.0)

    sig = tl.sigmoid(a_clamped * 1.702)
    glu = a_clamped * sig
    c_row = (b_clamped + 1.0) * glu

    tl.store(c_ptr + col_offsets, c_row.to(b_row.dtype), mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row_orig = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row_orig = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_row_orig = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    dc_row = dc_row_orig.to(tl.float32)
    a_row = a_row_orig.to(tl.float32)
    b_row = b_row_orig.to(tl.float32)

    a_clamped = tl.minimum(a_row, 7.0)
    b_clamped = tl.maximum(tl.minimum(b_row, 7.0), -7.0)

    sig = tl.sigmoid(a_clamped * 1.702)
    glu = a_clamped * sig
    activation_out = (b_clamped + 1.0) * glu

    tl.store(c_ptr + col_offsets, activation_out.to(dc_row_orig.dtype), mask=mask)

    db_clamped = dc_row * glu
    b_in_range = (b_row > -7.0) & (b_row < 7.0)
    db_row = tl.where(b_in_range, db_clamped, 0.0)

    da_clamped = dc_row * (b_clamped + 1.0) * sig * (1.0 + a_clamped * 1.702 * (1.0 - sig))
    a_in_range = a_row < 7.0
    da_row = tl.where(a_in_range, da_clamped, 0.0)

    tl.store(a_ptr + col_offsets, da_row.to(a_row_orig.dtype), mask=mask)
    tl.store(b_ptr + col_offsets, db_row.to(b_row_orig.dtype), mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward_triton(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    c_recomputed = torch.empty_like(dc)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        c_recomputed,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape), c_recomputed.view(*ori_shape)


def swiglu_backward_pytorch(gate, up, dc):
    gate_clamped = gate.clamp(max=7.0)
    up_clamped = up.clamp(min=-7.0, max=7.0)

    sig = torch.sigmoid(gate_clamped * 1.702)
    glu = gate_clamped * sig
    activation_out = (up_clamped + 1.0) * glu

    db_clamped = dc * glu
    b_in_range = (up > -7.0) & (up < 7.0)
    d_up = torch.where(b_in_range, db_clamped, torch.zeros_like(db_clamped))

    da_clamped = dc * (up_clamped + 1.0) * sig * (1.0 + gate_clamped * 1.702 * (1.0 - sig))
    a_in_range = gate < 7.0
    d_gate = torch.where(a_in_range, da_clamped, torch.zeros_like(da_clamped))

    return d_gate, d_up, activation_out


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """

    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)

        _gate, _up, activation_out = swiglu_forward(gate, up)

        out = F.linear(activation_out, w_down)

        ctx.save_for_backward(x, w_gate, w_up, w_down)
        ctx.alpha = alpha
        ctx.limit = limit

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_gate, w_up, w_down = ctx.saved_tensors
        ori_shape = grad_output.shape
        compute_dtype = grad_output.dtype

        x = x.to(compute_dtype)
        w_gate = w_gate.to(compute_dtype)
        w_up = w_up.to(compute_dtype)
        w_down = w_down.to(compute_dtype)

        gate = F.linear(x, w_gate)
        up = F.linear(x, w_up)

        d_activation = grad_output @ w_down

        d_gate, d_up, activation_out_recomputed = swiglu_backward_triton(
            gate.reshape(-1, gate.shape[-1]).contiguous(),
            up.reshape(-1, up.shape[-1]).contiguous(),
            d_activation.reshape(-1, d_activation.shape[-1]),
        )
        del d_activation
        gate_shape = (*ori_shape[:-1], d_gate.shape[-1])
        d_gate = d_gate.view(gate_shape)
        d_up = d_up.view(gate_shape)

        grad_output_2d = grad_output.view(-1, ori_shape[-1])
        d_w_down = grad_output_2d.T @ activation_out_recomputed.view(-1, activation_out_recomputed.shape[-1])
        del activation_out_recomputed

        d_x = d_gate @ w_gate + d_up @ w_up

        x_2d = x.view(-1, x.shape[-1])
        d_gate_2d = d_gate.view(-1, d_gate.shape[-1])
        d_up_2d = d_up.view(-1, d_up.shape[-1])
        d_w_gate = d_gate_2d.T @ x_2d
        d_w_up = d_up_2d.T @ x_2d

        return d_x, d_w_gate, d_w_up, d_w_down, None, None


class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.

    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
