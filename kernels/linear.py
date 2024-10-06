# needs some work for numerical accuracy
# i believe linear is now done

import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_T": 128, "BLOCK_SIZE_IN": 64, "BLOCK_SIZE_OUT": 64}, num_warps=4)],
    key=["T", "IN", "OUT"]
)
@triton.jit
def _linear_kernel(
    x_ptr, stride_x_b, stride_x_t, stride_x_in,
    fc_ptr, stride_fc_in, stride_fc_out,
    out_ptr, stride_out_b, stride_out_t, stride_out_out,
    B, T, IN, OUT,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_IN: tl.constexpr, BLOCK_SIZE_OUT: tl.constexpr
):
    batch = tl.program_id(0)
    group_row = tl.program_id(1)
    group_col = tl.program_id(2)

    x_ptr += batch * stride_x_b
    out_ptr += batch * stride_out_b

    offsets_t = group_row * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offsets_out = group_col * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offsets_in = tl.arange(0, BLOCK_SIZE_IN)

    accumulator = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_OUT), dtype=tl.float32)
    for _ in range(0, tl.cdiv(IN, BLOCK_SIZE_IN)):
        mask_x = (offsets_t[:, None] < T) & (offsets_in[None, :] < IN)
        offsets_x = offsets_t[:, None] * stride_x_t + offsets_in[None, :] * stride_x_in
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mask_fc = (offsets_in[:, None] < IN) & (offsets_out[None, :] < OUT)
        offsets_fc = offsets_in[:, None] * stride_fc_in + offsets_out[None, :] * stride_fc_out
        fc = tl.load(fc_ptr + offsets_fc, mask=mask_fc, other=0.0)

        accumulator = tl.dot(x, fc, accumulator, out_dtype=tl.float32)

        offsets_in += BLOCK_SIZE_IN

    mask_out = (offsets_t[:, None] < T) & (offsets_out[None, :] < OUT)
    offsets_out = offsets_t[:, None] * stride_out_t + offsets_out[None, :] * stride_out_out
    tl.store(out_ptr + offsets_out, accumulator, mask=mask_out)

def linear(x, fc):
    assert x.is_contiguous()
    B, T, IN = x.shape
    weight = fc.weight
    OUT, IN = weight.shape
    # bias = fc.bias

    # if bias:
    #     out = fc_bias.clone().unsqueeze(0).repeat(B, 1, 1)

    out = torch.zeros(size=(B, T, OUT), device="cuda", dtype=torch.float16)
    grid = lambda meta: (B, triton.cdiv(T, meta["BLOCK_SIZE_T"]), triton.cdiv(OUT, meta["BLOCK_SIZE_OUT"]))
    _linear_kernel[grid](
        x, x.stride(0), x.stride(1), x.stride(2),
        weight, weight.stride(1), weight.stride(0),
        out, out.stride(0), out.stride(1), out.stride(2),
        B, T, IN, OUT,
    )
    return out