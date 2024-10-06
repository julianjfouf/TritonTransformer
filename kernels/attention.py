import torch

import triton
import triton.language as tl

configs1 = [
    triton.Config({"BLOCK_SIZE_T1": BT1, "BLOCK_SIZE_T2": BT2, "BLOCK_SIZE_C": BC}, num_warps=w) for BT1 in [32, 64, 128] for BT2 in [32, 64, 128] for BC in [32, 64, 128] for w in [2, 4, 8]
]
@triton.autotune(
    # configs=configs1,

    # best for T=1024 and C=256, and varying B and H
    # configs=[triton.Config({"BLOCK_SIZE_T1": 64, "BLOCK_SIZE_T2": 128, "BLOCK_SIZE_C": 32}, num_warps=2)],

    # best for T=1024 and C=768, and varying B and H
    configs=[triton.Config({"BLOCK_SIZE_T1": 128, "BLOCK_SIZE_T2": 128, "BLOCK_SIZE_C": 64}, num_warps=4)], 

    key=["T", "C"]
)
@triton.jit
def _causal_attention_kernel1(
    q_ptr, stride_q_bh, stride_q_t1, stride_q_c,
    k_ptr, stride_k_bh, stride_k_t2, stride_k_c,
    out_ptr, stride_out_bh, stride_out_t1, stride_out_t2,
    group_rows_and_cols_ptr,
    BH, T, C,
    BLOCK_SIZE_T1: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_T2: tl.constexpr,
):
    # go to specific head of specific batch operating on
    batch_head = tl.program_id(0)
    q_ptr += batch_head * stride_q_bh
    k_ptr += batch_head * stride_k_bh
    out_ptr += batch_head * stride_out_bh

    kernel = tl.program_id(1)

    # get the tile we are on
    group_row = tl.load(group_rows_and_cols_ptr + kernel * 2 + 0)
    group_col = tl.load(group_rows_and_cols_ptr + kernel * 2 + 1)

    offsets_t1 = group_row * BLOCK_SIZE_T1 + tl.arange(0, BLOCK_SIZE_T1)
    offsets_t2 = group_col * BLOCK_SIZE_T2 + tl.arange(0, BLOCK_SIZE_T2)
    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    accumulator = tl.zeros((BLOCK_SIZE_T1, BLOCK_SIZE_T2), dtype=tl.float32)
    for _ in range(0, tl.cdiv(C, BLOCK_SIZE_C)):

        mask_q = (offsets_t1[:, None] < T) & (offsets_c[None, :] < C)
        offsets_q = offsets_t1[:, None] * stride_q_t1 + offsets_c[None, :] * stride_q_c
        q = tl.load(q_ptr + offsets_q, mask=mask_q, other=0.0)

        mask_k = (offsets_c[:, None] < C) & (offsets_t2[None, :] < T)
        offsets_k = offsets_c[:, None] * stride_k_c + offsets_t2[None, :] * stride_k_t2
        k = tl.load(k_ptr + offsets_k, mask=mask_k, other=0.0)

        accumulator = tl.dot(q, k, accumulator, out_dtype=tl.float32)
        
        offsets_c += BLOCK_SIZE_C

    mask_out = (offsets_t1[:, None] < T) & (offsets_t2[None, :] < T) & (offsets_t1[:, None] >= offsets_t2[None, :])
    offsets_out = offsets_t1[:, None] * stride_out_t1 + offsets_t2[None, :] * stride_out_t2
    tl.store(out_ptr + offsets_out, accumulator * tl.rsqrt(C.to(tl.float32)), mask=mask_out)

configsd2 = [
    triton.Config({"BLOCK_SIZE_T1": BT1, "BLOCK_SIZE_T2": BT2, "BLOCK_SIZE_C": BC}, num_warps=w) for BT1 in [32, 64, 128] for BT2 in [32, 64, 128] for BC in [32, 64, 128] for w in [2, 4, 8]
]
@triton.autotune(
    # configs=configsd2,

    # best for fixed T and C, and varying B and H
    configs=[triton.Config({"BLOCK_SIZE_T1": 64, "BLOCK_SIZE_T2": 64, "BLOCK_SIZE_C": 128}, num_warps=8)], 
    
    key=["T", "C"]
)
@triton.jit
def _causal_attention_kerneld2(
    qk_ptr, stride_qk_bh, stride_qk_t1, stride_qk_t2,
    v_ptr, stride_v_bh, stride_v_t2, stride_v_c,
    out_ptr, stride_out_bh, stride_out_t1, stride_out_c,
    BH, T, C,
    BLOCK_SIZE_T1: tl.constexpr, BLOCK_SIZE_T2: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    batch_head = tl.program_id(0)
    qk_ptr += batch_head * stride_qk_bh
    v_ptr += batch_head * stride_v_bh
    out_ptr += batch_head * stride_out_bh

    group_row = tl.program_id(1)
    group_col = tl.program_id(2)

    offsets_t1 = group_row * BLOCK_SIZE_T1 + tl.arange(0, BLOCK_SIZE_T1)
    offsets_c = group_col * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offsets_t2 = tl.arange(0, BLOCK_SIZE_T2)

    global_maxes = tl.zeros((BLOCK_SIZE_T1, ), dtype=tl.float32)
    accumulator_softmax = tl.zeros((BLOCK_SIZE_T1, BLOCK_SIZE_T2, ), dtype=tl.float32)
    accumulator_matmul = tl.zeros((BLOCK_SIZE_T1, BLOCK_SIZE_C, ), dtype=tl.float32)
    for _ in range(0, tl.cdiv(group_row * BLOCK_SIZE_T1 + BLOCK_SIZE_T1, BLOCK_SIZE_T2)):
        mask_qk = (offsets_t1[:, None] < T) & (offsets_t2[None, :] < T)
        offsets_qk = offsets_t1[:, None] * stride_qk_t1 + offsets_t2[None, :] * stride_qk_t2
        qk = tl.load(qk_ptr + offsets_qk, mask=mask_qk, other=float("-inf")).to(tl.float32)

        mask_v = (offsets_t2[:, None] < T) & (offsets_c[None, :] < C)
        offsets_v = offsets_t2[:, None] * stride_v_t2 + offsets_c[None, :] * stride_v_c
        v = tl.load(v_ptr + offsets_v, mask=mask_v, other=0.0)

        current_maxes = tl.max(qk, axis=1)
        new_maxes = tl.maximum(global_maxes, current_maxes)
        attn = tl.exp(qk - new_maxes[:, None])
        accumulator_softmax = accumulator_softmax * tl.exp(global_maxes - new_maxes)[:, None] + attn

        accumulator_matmul = accumulator_matmul * tl.exp(global_maxes - new_maxes)[:, None] + tl.dot(attn.to(tl.float16), v, out_dtype=tl.float32)

        global_maxes = new_maxes
        offsets_t2 += BLOCK_SIZE_T2

    mask_out = (offsets_t1[:, None] < T) & (offsets_c[None, :] < C)
    offsets_out = offsets_t1[:, None] * stride_out_t1 + offsets_c[None, :] * stride_out_c
    tl.store(out_ptr + offsets_out, accumulator_matmul / tl.sum(accumulator_softmax, axis=1)[:, None], mask=mask_out)

group_rows_and_cols = []
for i in range(0, 256):
    for j in range(0, i + 1):
        group_rows_and_cols.append([i, j])
group_rows_and_cols = torch.tensor(group_rows_and_cols, device="cuda", dtype=torch.int16)

def calculate_num_kernels(T, BLOCK_SIZE_T1, BLOCK_SIZE_T2):
    n = triton.cdiv(T, BLOCK_SIZE_T1)
    num_kernels = triton.cdiv(BLOCK_SIZE_T1, BLOCK_SIZE_T2) * (n * (n + 1) // 2)

    return num_kernels

def attention(q, k, v):
    B, H, T, C = q.shape
    q = q.contiguous().view(B * H, T, C)
    k = k.contiguous().view(B * H, T, C)
    v = v.contiguous().view(B * H, T, C)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    qk = torch.full(size=(B * H, T, T), fill_value=float("-inf"), device="cuda", dtype=torch.float16)
    grid = lambda meta: (B * H, calculate_num_kernels(T, meta["BLOCK_SIZE_T1"], meta["BLOCK_SIZE_T2"]))
    _causal_attention_kernel1[grid](
        q, q.stride(0), q.stride(1), q.stride(2),
        k, k.stride(0), k.stride(1), k.stride(2),
        qk, qk.stride(0), qk.stride(1), qk.stride(2),
        group_rows_and_cols,
        B * H, T, C,
    )

    out = torch.zeros(size=(B * H, T, C), device="cuda", dtype=torch.float16)
    grid = lambda meta: (B * H, triton.cdiv(T, meta["BLOCK_SIZE_T1"]), triton.cdiv(C, meta["BLOCK_SIZE_C"]))
    _causal_attention_kerneld2[grid](
        qk, qk.stride(0), qk.stride(1), qk.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        out, out.stride(0), out.stride(1), out.stride(2),
        B * H, T, C,
    )

    return out.view(B, H, T, C)