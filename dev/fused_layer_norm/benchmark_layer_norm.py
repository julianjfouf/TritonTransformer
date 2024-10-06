import torch
import torch.nn as nn

import triton

from layer_norm import layer_norm1, layer_norm2, layer_norm3

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
        line_vals=["kernel3", "torch"],
        line_names=["kernel3", "Torch"],
        styles=[("purple", "-"), ("green", "-")],
        ylabel=measure,
        plot_name="layer_norm-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, C, provider):
    B = 4
    T = 1024
    ln = nn.LayerNorm(C, device="cuda", dtype=torch.float16)
    x = torch.randn(size=(B, T, C), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    # if provider == "kernel1":
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer_norm1(x, ln.weight, ln.bias), quantiles=quantiles)
    
    # if provider == "kernel2":
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer_norm2(x, ln.weight, ln.bias), quantiles=quantiles)
    
    if provider == "kernel3":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer_norm3(x, ln.weight, ln.bias), quantiles=quantiles)
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ln(x), quantiles=quantiles)
    
    if measure == "gbps":
        gbps = lambda ms: (x.numel() * x.element_size() * 1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda ms: (2 * 2 * B * T * C * C * 4 * C * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")