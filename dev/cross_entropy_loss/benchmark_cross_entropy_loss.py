import torch
import torch.nn.functional as F

import triton

from cross_entropy_loss import cross_entropy_loss

configs = []

for measure in ["gbps", "tflops", "ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["n_classes"],
        x_vals=[2**i for i in range(8, 18)], # og: 8, 18
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel=measure,
        plot_name="cross-entropy-loss-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, n_classes, provider):
    n_rows = 1024
    reduction = "sum"
    predictions = torch.randn(size=(n_rows, n_classes, ), device="cuda", dtype=torch.float16)
    index = torch.randint(0, n_classes, size=(n_rows,), device="cuda")
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cross_entropy_loss(predictions, index, reduction=reduction), quantiles=quantiles)
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.cross_entropy(predictions, index, reduction=reduction), quantiles=quantiles)
    
    if measure == "gbps":
        gbps = lambda x: (predictions.numel() * predictions.element_size() * 1e-9) / (x * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda x: (predictions.numel() * 1e-12) / (x * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")