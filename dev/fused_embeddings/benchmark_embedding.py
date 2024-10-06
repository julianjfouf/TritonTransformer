import torch
import torch.nn as nn

import triton

from embedding import embedding

configs = []

for measure in ["gbps", "tflops", "ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["vocab_size"],
        # x_vals=[128 * i for i in range(1, 9)], 
        x_vals=[2**i for i in range(4, 11)],
        # x_log=False, 
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel=measure,
        plot_name="embedding-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, vocab_size, provider):
    B = 4
    T = 2048
    C = 2048
    ids = torch.randint(0, vocab_size, size=(B, T), device="cuda")

    tok_embedding = nn.Embedding(vocab_size, C, device="cuda")
    pos_embedding = nn.Embedding(T, C, device="cuda")
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: embedding(ids, tok_embedding.weight, pos_embedding.weight), quantiles=quantiles)

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tok_embedding(ids) + pos_embedding(torch.arange(0, T, device="cuda")), quantiles=quantiles)
    
    if measure == "gbps":
        gbps = lambda ms: ((ids.numel() * ids.element_size() + tok_embedding.weight.numel() * tok_embedding.weight.element_size() + pos_embedding.weight.numel() * pos_embedding.weight.element_size()) * 1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda ms: (B * T * C * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")