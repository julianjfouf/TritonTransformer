import torch
import torch.nn as nn

import triton

from torchtransformer import TorchTransformer
from tritontransformer import TritonTransformer

import numpy as np
np.int = np.int32

context_length = 1024
vocab_size = 50257
n_hidden = 768
n_heads = 12
head_size = n_hidden // n_heads
n_layers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_torch = TorchTransformer(vocab_size, context_length, n_layers, n_hidden, n_heads, head_size).to(device)
model_torch.requires_grad_ = False
model_triton = TritonTransformer(vocab_size, context_length, n_layers, n_hidden, n_heads, head_size).to(device)
model_triton.requires_grad_ = False

for key in model_torch.state_dict():
    model_triton.state_dict()[key].copy_(model_torch.state_dict()[key])

compiled_model_torch = torch.compile(model_torch)
# compiled_model_triton = torch.compile(model_triton)

configs = []

for measure in ["ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["batch_size"],
        # x_vals=[128 * i for i in range(1, 9)], 
        x_vals=[2, 4, 8],
        x_log=False, 
        # x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],# "triton_compile"],
        line_names=["Triton", "Torch", "TorchCompile"],# "TritonCompile"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],# ("purple", "-")],
        ylabel=measure,
        plot_name="transformer-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, batch_size, provider):

    sample_input = torch.randint(0, vocab_size, size=(batch_size, context_length), device=device)
    sample_targets = torch.randint(0, vocab_size, size=(batch_size, context_length), device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: model_triton(sample_input, sample_targets), quantiles=quantiles)
    
    # if provider == "triton_compile":
        # ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_model_triton(sample_input, sample_targets), quantiles=quantiles)

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: model_torch(sample_input, sample_targets), quantiles=quantiles)
    
    if provider == "torch_compile":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_model_torch(sample_input, sample_targets), quantiles=quantiles)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")