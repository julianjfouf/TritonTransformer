import torch

import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": BC, "BLOCK_SIZE_H": BH}, num_warps=w) for BC in [32, 64, 128] for BH in [32, 64, 128, 256] for w in [4, 8]
]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_BT": 32, "BLOCK_SIZE_C": 128, "BLOCK_SIZE_H": 64}, num_warps=8)],
    # configs=configs,
    key=["BT", "C", "H"]
)
@triton.jit
def _linear_residual_kernel(
    x_ptr, stride_x_bt, stride_x_h,
    fc_ptr, stride_fc_c, stride_fc_h,
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
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mask_fc = (offsets_h[:, None] < H) & (offsets_c[None, :] < C)
        offsets_fc = offsets_h[:, None] * stride_fc_h + offsets_c[None, :] * stride_fc_c
        fc = tl.load(fc_ptr + offsets_fc, mask=mask_fc, other=0.0)

        accumulator = tl.dot(x, fc, accumulator)

        offsets_h += BLOCK_SIZE_H
    
    mask_out = (offsets_bt[:, None] < BT) & (offsets_c[None, :] < C)
    offsets_out = offsets_bt[:, None] * stride_out_bt + offsets_c[None, :] * stride_out_c
    tl.atomic_add(out_ptr + offsets_out, accumulator, mask=mask_out)

def linear_residual(residual: torch.Tensor, x: torch.Tensor, o: torch.Tensor):
    B, T, H = x.shape
    x = x.view(B * T, H)
    assert x.is_contiguous()
    assert o.is_contiguous()
    C, H = o.shape
    out = residual.view(B * T, C).clone().contiguous()
    grid = lambda meta: (triton.cdiv(B * T, meta["BLOCK_SIZE_BT"]), triton.cdiv(C, meta["BLOCK_SIZE_C"]))
    _linear_residual_kernel[grid](
        x, x.stride(0), x.stride(1),
        o, o.stride(0), o.stride(1),
        out, out.stride(0), out.stride(1),
        B * T, C, H,
    )
    return out.view(B, T, C).contiguous()