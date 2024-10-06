import torch
import torch.nn.functional as F

from attention import causal_attention

def test_attention():
    B = 4 # 4
    H = 12 # 12
    T = 1024
    C = 768
    q = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)

    output_triton = causal_attention(q, k, v)
    output_torch = F.scaled_dot_product_attention(q, k, v, is_causal=True) # F.softmax((q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5).masked_fill(mask[:,:,:T,:T] == 0, float('-inf')), dim=-1) @ v


    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("All ok!")
    else:
        print("Something is wrong!")
        print(output_torch)
        print(output_triton)
        
    diffs = torch.abs(output_triton - output_torch)
    print("avg diff:", torch.mean(diffs))
    print("max diff:", torch.max(diffs))
    print("min diff:", torch.min(diffs))

test_attention()