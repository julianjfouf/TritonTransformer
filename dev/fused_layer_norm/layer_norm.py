import torch

import triton
import triton.language as tl

configs1 = [
    triton.Config({"BLOCK_SIZE_C": BC}, num_warps=w) for BC in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] for w in [2, 4, 8]
]

@triton.autotune(
    configs=configs1,
    # configs=[triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=4)],
    key=["BT", "C"]
)
@triton.jit
def _layer_norm_kernel1(
    x_ptr, stride_x_bt, stride_x_c,
    out_ptr, stride_out_bt, stride_out_c,
    gamma_ptr, stride_gamma_c,
    beta_ptr, stride_beta_c,
    BT, C,
    BLOCK_SIZE_C: tl.constexpr
):
    row = tl.program_id(0)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    # calculate mean
    mean_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float16)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (row < BT) & (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mean_accumulator += x
        offsets_c += BLOCK_SIZE_C
    
    mean = tl.sum(mean_accumulator) / C
    mean = mean.to(tl.float16)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    var_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float16)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (row < BT) & (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        shifted_x = x - mean

        var_accumulator += shifted_x * shifted_x
        offsets_c += BLOCK_SIZE_C
    
    var = tl.sum(var_accumulator) / C
    var = var.to(tl.float16)
    rstd = tl.rsqrt(var + 1e-5)
    rstd = rstd.to(tl.float16)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (row < BT) & (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mask_gamma = (offsets_c < C)
        offsets_gamma = offsets_c * stride_gamma_c
        gamma = tl.load(gamma_ptr + offsets_gamma, mask=mask_gamma, other=0.0)
        
        mask_beta = (offsets_c < C)
        offsets_beta = offsets_c * stride_beta_c
        beta = tl.load(beta_ptr + offsets_beta, mask=mask_beta, other=0.0)

        mask_out = (row < BT) & (offsets_c < C)
        offsets_out = row * stride_out_bt + offsets_c * stride_out_c
        tl.store(out_ptr + offsets_out, ((x - mean) * rstd) * gamma + beta, mask=mask_out)

        offsets_c += BLOCK_SIZE_C
    
def layer_norm1(x, gamma, beta):
    B, T, C = x.shape
    x = x.view(B * T, C)
    assert x.is_contiguous()
    out = torch.zeros_like(x, device="cuda", dtype=torch.float16)
    grid = lambda meta: (B * T, )
    _layer_norm_kernel1[grid](
        x, x.stride(0), x.stride(1),
        out, out.stride(0), out.stride(1),
        gamma, gamma.stride(0),
        beta, beta.stride(0),
        B * T, C,
    )
    return out.view(B, T, C)


configs2 = [
    triton.Config({"BLOCK_SIZE_C": BC}, num_warps=w) for BC in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] for w in [2, 4, 8]
]

@triton.autotune(
    configs=configs2,
    # configs=[triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=4)],
    key=["BT", "C"]
)
@triton.jit
def _layer_norm_kernel2(
    x_ptr, stride_x_bt, stride_x_c,
    out_ptr, stride_out_bt, stride_out_c,
    gamma_ptr, stride_gamma_c,
    beta_ptr, stride_beta_c,
    BT, C,
    BLOCK_SIZE_C: tl.constexpr
):
    row = tl.program_id(0)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    # calculate mean
    mean_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float32)
    squared_mean_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float32)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (row < BT) & (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mean_accumulator += x
        squared_mean_accumulator += x * x
        offsets_c += BLOCK_SIZE_C
    
    mean = tl.sum(mean_accumulator) / C
    mean_squared = mean * mean
    squared_mean = tl.sum(squared_mean_accumulator) / C

    var = squared_mean - mean_squared
    rstd = tl.rsqrt(var + 1e-5)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (row < BT) & (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)

        mask_gamma = (offsets_c < C)
        offsets_gamma = offsets_c * stride_gamma_c
        gamma = tl.load(gamma_ptr + offsets_gamma, mask=mask_gamma, other=0.0)
        
        mask_beta = (offsets_c < C)
        offsets_beta = offsets_c * stride_beta_c
        beta = tl.load(beta_ptr + offsets_beta, mask=mask_beta, other=0.0)

        mask_out = (row < BT) & (offsets_c < C)
        offsets_out = row * stride_out_bt + offsets_c * stride_out_c
        tl.store(out_ptr + offsets_out, ((x - mean) * rstd) * gamma + beta, mask=mask_out)

        offsets_c += BLOCK_SIZE_C
    
def layer_norm2(x, gamma, beta):
    B, T, C = x.shape
    x = x.view(B * T, C)
    assert x.is_contiguous()
    out = torch.zeros_like(x, device="cuda", dtype=torch.float16)
    grid = lambda meta: (B * T, )
    _layer_norm_kernel2[grid](
        x, x.stride(0), x.stride(1),
        out, out.stride(0), out.stride(1),
        gamma, gamma.stride(0),
        beta, beta.stride(0),
        B * T, C,
    )
    return out.view(B, T, C)


configs3 = [
    triton.Config({"BLOCK_SIZE_C": BC}, num_warps=w) for BC in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] for w in [2, 4, 8]
]

@triton.autotune(
    configs=configs3,
    # configs=[triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=4)],
    key=["BT", "C"]
)
@triton.jit
def _layer_norm_kernel3(
    x_ptr, stride_x_bt, stride_x_c,
    out_ptr, stride_out_bt, stride_out_c,
    gamma_ptr, stride_gamma_c,
    beta_ptr, stride_beta_c,
    BT, C,
    BLOCK_SIZE_C: tl.constexpr
):
    row = tl.program_id(0)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    # calculate mean
    mean_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float32)
    squared_mean_accumulator = tl.zeros((BLOCK_SIZE_C, ), dtype=tl.float32)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0).to(tl.float32)

        mean_accumulator += x
        squared_mean_accumulator += x * x
        
        offsets_c += BLOCK_SIZE_C
    
    mean = tl.sum(mean_accumulator) / C
    mean_squared = mean * mean
    squared_mean = tl.sum(squared_mean_accumulator) / C

    var = squared_mean - mean_squared
    rstd = tl.rsqrt(var + 1e-5)

    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):
        mask_x = (offsets_c < C)
        offsets_x = row * stride_x_bt + offsets_c * stride_x_c
        x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0).to(tl.float32)

        mask_gamma = (offsets_c < C)
        offsets_gamma = offsets_c * stride_gamma_c
        gamma = tl.load(gamma_ptr + offsets_gamma, mask=mask_gamma, other=0.0).to(tl.float32)
        
        mask_beta = (offsets_c < C)
        offsets_beta = offsets_c * stride_beta_c
        beta = tl.load(beta_ptr + offsets_beta, mask=mask_beta, other=0.0).to(tl.float32)

        mask_out = (offsets_c < C)
        offsets_out = row * stride_out_bt + offsets_c * stride_out_c
        tl.store(out_ptr + offsets_out, beta + (x - mean) * rstd * gamma, mask=mask_out)

        offsets_c += BLOCK_SIZE_C
    
def layer_norm3(x, gamma, beta):
    B, T, C = x.shape
    x = x.view(B * T, C)
    assert x.is_contiguous()

    out = torch.zeros(size=(B * T, C), device="cuda", dtype=torch.float16)

    grid = lambda meta: (B * T, )
    _layer_norm_kernel3[grid](
        x, x.stride(0), x.stride(1),
        out, out.stride(0), out.stride(1),
        gamma, gamma.stride(0),
        beta, beta.stride(0),
        B * T, C,
    )

    return out.view(B, T, C)