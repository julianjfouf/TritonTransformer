import torch
import torch.nn.functional as F

import triton

from attention import attention1, attention2

configs = []

for measure in ["gbps", "tflops", "ms"]:
    configs.append(
        triton.testing.Benchmark(
        x_names=["B"],
        x_vals=[2**i for i in range(1, 4)],
        x_log=True,
        line_arg="provider",
        line_vals=["kernel1", "kernel2", "torch", "flashattn", "trueflashattn"],
        line_names=["kernel1", "kernel2", "Torch", "flashattn", "trueflashattn"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("yellow", "-"), ("orange", "-")],
        ylabel=measure,
        plot_name="attention-" + measure,
        args={"measure": measure},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(measure, B, provider):
    H = 12
    T = 1024
    C = 256 # 768
    q = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    k = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    v = torch.randn(size=(B, H, T, C), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    mask = torch.tril(torch.ones(size=(T, T), device="cuda")).view(1, 1, T, T)

    if provider == "kernel1":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention1(q, k, v), quantiles=quantiles)

    if provider == "kernel2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention2(q, k, v), quantiles=quantiles)
    
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.softmax((q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5).masked_fill(mask[:,:,:T,:T] == 0, float('-inf')), dim=-1) @ v, quantiles=quantiles)

    if provider == "flashattn":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), quantiles=quantiles)

    if measure == "gbps":
        gbps = lambda ms: ((q.numel() * q.element_size() + k.numel() * k.element_size() + v.numel() * v.element_size()) * 1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(min_ms), gbps(max_ms)
    
    if measure == "tflops":
        tflops = lambda ms: (2 * 2 * B * H * T * T * C * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)
    
    return ms, min_ms, max_ms

benchmark.run(save_path="benchmark_results")