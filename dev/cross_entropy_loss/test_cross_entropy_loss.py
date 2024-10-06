import torch
import torch.nn.functional as F

from cross_entropy_loss import cross_entropy_loss

def test_cross_entropy_loss():
    n_classes = 50257
    batch_size = 1
    context_length = 1024
    reduction = "mean"
    predictions = torch.randn(size=(batch_size * context_length, n_classes,), device="cuda", dtype=torch.float16) * 10.0
    index = torch.randint(0, n_classes, size=(batch_size * context_length,), device="cuda")

    output_triton = cross_entropy_loss(predictions, index, reduction=reduction)
    output_torch = F.cross_entropy(predictions, index, reduction=reduction)

    predictions = predictions.to(torch.float32)
    fp32_output = F.cross_entropy(predictions, index, reduction=reduction)
    print("fp32 output_torch:", fp32_output)

    if torch.allclose(output_triton, output_torch):
        print("output_triton:", output_triton)
        print("output_torch:", output_torch)
        print("All ok!")
    else:
        print("Something is wrong!")
        print("triton:", output_triton)
        print("torch:", output_torch)

test_cross_entropy_loss()