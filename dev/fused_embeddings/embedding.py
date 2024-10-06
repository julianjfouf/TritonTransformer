import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE_B": 2, "BLOCK_SIZE_T": BT, "BLOCK_SIZE_C": 128}, num_warps=8) for BT in [16, 32, 64]
]

@triton.autotune(
    configs=configs,
    key=["B", "T", "C"]
)
@triton.jit
def _embedding_kernel(
    ids_ptr, stride_ids_b, stride_ids_t,
    tok_emb_ptr, stride_tok_emb_vocab_size, stride_tok_emb_c,
    pos_emb_ptr, stride_pos_emb_t, stride_pos_emb_c,
    out_ptr, stride_out_b, stride_out_t, stride_out_c,
    B, T, C,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    batch_group = tl.program_id(0)
    time_group = tl.program_id(1)
    channels_group = tl.program_id(2)

    offsets_b = batch_group * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offsets_t = time_group * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offsets_c = channels_group * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    mask_ids = (offsets_b[:, None] < B) & (offsets_t[None, :] < T)
    offsets_ids = offsets_b[:, None] * stride_ids_b + offsets_t[None, :] * stride_ids_t
    ids = tl.load(ids_ptr + offsets_ids, mask=mask_ids, other=0) # B, T

    mask_tok_emb = (offsets_b[:, None, None] < B) & (offsets_t[None, :, None] < T) & (offsets_c[None, None, :] < C)
    offsets_tok_emb = ids[:, :, None] * stride_tok_emb_vocab_size + offsets_c[None, None, :] * stride_tok_emb_c
    tok_emb = tl.load(tok_emb_ptr + offsets_tok_emb, mask=mask_tok_emb, other=0.0)

    mask_pos_emb = (offsets_b[:, None, None] < B) & (offsets_t[None, :, None] < T) & (offsets_c[None, None, :] < C)
    offsets_pos_emb = tl.zeros((BLOCK_SIZE_B, 1, 1), dtype=tl.int32) + offsets_t[None, :, None] * stride_pos_emb_t + offsets_c[None, None, :] * stride_pos_emb_c
    pos_emb = tl.load(pos_emb_ptr + offsets_pos_emb, mask=mask_pos_emb, other=0.0)

    mask_out = (offsets_b[:, None, None] < B) & (offsets_t[None, :, None] < T) & (offsets_c[None, None, :] < C)
    offsets_out = offsets_b[:, None, None] * stride_out_b + offsets_t[None, :, None] * stride_out_t + offsets_c[None, None, :] * stride_out_c
    tl.store(out_ptr + offsets_out, tok_emb + pos_emb, mask=mask_out)

def embedding(ids, tok_embeddings, pos_embeddings, dtype=torch.float32):
    assert ids.is_contiguous()
    assert tok_embeddings.is_contiguous()
    assert pos_embeddings.is_contiguous()

    B, T = ids.shape
    vocab_size, C = tok_embeddings.shape
    T, C = pos_embeddings.shape

    out = torch.empty(size=(B, T, C), device="cuda", dtype=dtype)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_SIZE_B"]), triton.cdiv(T, meta["BLOCK_SIZE_T"]), triton.cdiv(C, meta["BLOCK_SIZE_C"]), )
    _embedding_kernel[grid](
        ids, ids.stride(0), ids.stride(1),
        tok_embeddings, tok_embeddings.stride(0), tok_embeddings.stride(1),
        pos_embeddings, pos_embeddings.stride(0), pos_embeddings.stride(1),
        out, out.stride(0), out.stride(1), out.stride(2),
        B, T, C,
    )
    return out