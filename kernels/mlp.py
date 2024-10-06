# needs some work for numerical accuracy
# i believe this issue is resolved

import torch

import triton
import triton.language as tl

@triton.jit
def _relu(accumulator):
    return tl.where(accumulator > 0.0, accumulator, 0.0)

configs1 = [
    triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": BC, "BLOCK_SIZE_H": BH}, num_warps=w) for BC in [32, 64, 128] for BH in [32, 64, 128, 256] for w in [4, 8]
]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": 32, "BLOCK_SIZE_H": 256}, num_warps=4)],
    # configs=configs1,
    key=["BT", "C", "H"]
)
@triton.jit
def _mlp_kernel1(
    x_ptr, stride_x_bt, stride_x_c,
    fc1_ptr, stride_fc1_h, stride_fc1_c,
    out_ptr, stride_out_bt, stride_out_h,
    BT, C, H,
    ACTIVATION,
    BLOCK_SIZE_BT: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    group_row = tl.program_id(0)
    group_col = tl.program_id(1)

    offsets_bt = group_row * BLOCK_SIZE_BT + tl.arange(0, BLOCK_SIZE_BT)
    offsets_h = group_col * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    accumulator = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_H), dtype=tl.float32)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (offsets_bt[:, None] < BT) & (offsets_c[None, :] < C)
        offsets_x = offsets_bt[:, None] * stride_x_bt + offsets_c[None, :] * stride_x_c

        mask_fc1 = (offsets_h[None, :] < H) & (offsets_c[:, None] < C) # reverse for transpose
        offsets_fc1 = offsets_h[None, :] * stride_fc1_h + offsets_c[:, None] * stride_fc1_c # reverse for transpose

        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)
        fc1 = tl.load(fc1_ptr + offsets_fc1, mask=mask_fc1, other=0.0)
        accumulator = tl.dot(x, fc1, accumulator, out_dtype=tl.float32)

        offsets_c += BLOCK_SIZE_C

    if ACTIVATION == 1:
        accumulator = _relu(accumulator)

    mask_out = (offsets_bt[:, None] < BT) & (offsets_h[None, :] < H)
    offsets_out = offsets_bt[:, None] * stride_out_bt + offsets_h[None, :] * stride_out_h
    tl.store(out_ptr + offsets_out, accumulator, mask=mask_out)

configs2 = [
    triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": BC, "BLOCK_SIZE_H": BH}, num_warps=w) for BC in [32, 64, 128] for BH in [32, 64, 128, 256] for w in [4, 8]
]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": 128, "BLOCK_SIZE_H": 64}, num_warps=8)],
    # configs=configs2,
    key=["BT", "C", "H"]
)
@triton.jit
def _mlp_kernel2(
    x_ptr, stride_x_bt, stride_x_h,
    fc2_ptr, stride_fc2_c, stride_fc2_h,
    out_ptr, stride_out_bt, stride_out_c,
    BT, C, H,
    BLOCK_SIZE_BT: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    group_row = tl.program_id(0)
    group_col = tl.program_id(1)

    offsets_bt = group_row * BLOCK_SIZE_BT + tl.arange(0, BLOCK_SIZE_BT)
    offsets_c = group_col * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offsets_h = tl.arange(0, BLOCK_SIZE_H)

    accumulator = tl.zeros((BLOCK_SIZE_BT, BLOCK_SIZE_C), dtype=tl.float32)
    for _ in range(0, tl.cdiv(H, BLOCK_SIZE_H)):
        mask_x = (offsets_bt[:, None] < BT) & (offsets_h[None, :] < H)
        offsets_x = offsets_bt[:, None] * stride_x_bt + offsets_h[None, :] * stride_x_h

        mask_fc2 = (offsets_h[:, None] < H) & (offsets_c[None, :] < C)
        offsets_fc2 = offsets_h[:, None] * stride_fc2_h + offsets_c[None, :] * stride_fc2_c

        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)
        fc2 = tl.load(fc2_ptr + offsets_fc2, mask=mask_fc2, other=0.0)
        accumulator = tl.dot(x, fc2, accumulator, out_dtype=tl.float32)

        offsets_h += BLOCK_SIZE_H
    
    mask_out = (offsets_bt[:, None] < BT) & (offsets_c[None, :] < C)
    offsets_out = offsets_bt[:, None] * stride_out_bt + offsets_c[None, :] * stride_out_c
    tl.atomic_add(out_ptr + offsets_out, accumulator, mask=mask_out)
    
def mlp(residual, x, fc1, fc2, activation=None):
    if activation == "ReLU":
        ACTIVATION = 1
    else:
        ACTIVATION = 0

    B, T, C = x.shape
    x = x.view(B * T, C)
    assert x.is_contiguous()
    assert fc1.is_contiguous()
    assert fc2.is_contiguous()
    H, C = fc1.shape
    C, H = fc2.shape
    out1 = torch.zeros(size=(B * T, H), device="cuda", dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE_BT"]), triton.cdiv(H, meta["BLOCK_SIZE_H"], ))
    _mlp_kernel1[grid](
        x, x.stride(0), x.stride(1), # B, T, C
        fc1, fc1.stride(0), fc1.stride(1), # H, C
        out1, out1.stride(0), out1.stride(1), # B, T, C
        B * T, C, H,
        ACTIVATION
    )
    out2 = residual.view(B * T, C).clone().contiguous()
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE_BT"]), triton.cdiv(C, meta["BLOCK_SIZE_C"]))
    _mlp_kernel2[grid](
        out1, out1.stride(0), out1.stride(1),
        fc2, fc2.stride(0), fc2.stride(1),
        out2, out2.stride(0), out2.stride(1),
        B * T, C, H,
    )
    return out2.view(B, T, C).contiguous()