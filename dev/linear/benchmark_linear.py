import torch
import torch.nn as nn

import triton

from linear import linear

configs = []

for measure in ["gbps", "tflops", "ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["OUT"],
        # x_vals=[128 * i for i in range(1, 9)], 
        x_vals=[2**i for i in range(9, 18)],
        # x_log=False, 
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel=measure,
        plot_name="linear-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, OUT, provider):
    B = 4
    T = 1024
    IN = 768
    x = torch.randn(size=(B, T, IN), device="cuda", dtype=torch.float16)
    fc = nn.Linear(IN, OUT, bias=False, device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: linear(x, fc), quantiles=quantiles)

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fc(x), quantiles=quantiles)
    
    if measure == "gbps":
        gbps = lambda ms: ((x.numel() * x.element_size() + fc.weight.numel() * fc.weight.element_size()) * 1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda ms: (2 * 2 * B * T * IN * IN * OUT * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")