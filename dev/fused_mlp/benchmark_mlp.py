import torch
import torch.nn as nn

import triton

from mlp import mlp1, mlp2

configs = []

for measure in ["gbps", "tflops", "ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["C"],
        # x_vals=[128 * i for i in range(1, 9)], 
        x_vals=[2**i for i in range(4, 12)],
        # x_log=False, 
        x_log=True,
        line_arg="provider",
        line_vals=["kernel1", "kernel2", "torch"],
        line_names=["kernel1", "kernel2", "Torch"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel=measure,
        plot_name="mlp-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, C, provider):
    B = 1
    T = 16
    x = torch.randn(size=(B, T, C), device="cuda", dtype=torch.float16)
    fc1 = nn.Linear(C, 4 * C, bias=False, device="cuda", dtype=torch.float16)
    relu = nn.ReLU()
    fc2 = nn.Linear(4 * C, C, bias=False, device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "kernel1":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: mlp1(x, x, fc1.weight, fc2.weight, activation="ReLU"), quantiles=quantiles)

    if provider == "kernel2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: mlp2(x, x, fc1.weight, fc2.weight, activation="ReLU"), quantiles=quantiles)
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + fc2(relu(fc1(x))), quantiles=quantiles)
    
    if measure == "gbps":
        gbps = lambda ms: ((x.numel() * x.element_size() + fc1.weight.numel() * fc1.weight.element_size()) * 1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda ms: (2 * 2 * B * T * C * C * 4 * C * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")