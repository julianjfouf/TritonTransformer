import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE_N": BN}, num_warps=w) for BN in [512, 1024, 2048] for w in [2, 4, 8]
]

@triton.autotune(
    # configs=configs,
    configs=[triton.Config({"BLOCK_SIZE_N": 2048}, num_warps=4)],
    key=["n_classes"]
)
@triton.jit
def _cross_entropy_loss_kernel(
    predictions_ptr, stride_predictions_bt, stride_predictions_n,
    index_ptr, stride_index_bt,
    output_ptr,
    n_rows, n_classes,
    REDUCTION,
    BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)

    index = tl.load(index_ptr + row * stride_index_bt)

    offsets_predictions_target = row * stride_predictions_bt + index * stride_predictions_n
    target = tl.load(predictions_ptr + offsets_predictions_target)
    
    offsets_predictions_n = tl.arange(0, BLOCK_SIZE_N)

    global_max = float("-inf")
    accumulator = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    for _ in range(0, tl.cdiv(n_classes, BLOCK_SIZE_N)):
        mask_predictions = (offsets_predictions_n < n_classes)
        offsets_predictions = row * stride_predictions_bt + offsets_predictions_n * stride_predictions_n
        block = tl.load(predictions_ptr + offsets_predictions, mask=mask_predictions, other=float("-inf")).to(tl.float32) # have to convert to float32 for tl.exp

        local_max = tl.max(block)
        new_max = max(global_max, local_max)
        accumulator = tl.exp(-new_max + global_max) * accumulator + tl.exp(block - new_max)
        global_max = new_max

        offsets_predictions_n += BLOCK_SIZE_N
        
    out = -(target - global_max) + tl.log(tl.sum(accumulator))
    
    if REDUCTION == 1:
        out = out / n_rows

    tl.atomic_add(output_ptr, out)

def cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor, reduction="mean"):

    assert predictions.is_contiguous()
    assert targets.is_contiguous()

    REDUCTION = 0
    if reduction == "mean":
        REDUCTION = 1

    n_rows, n_classes = predictions.shape
    n_rows, = targets.shape

    out = torch.tensor(0.0, device="cuda", dtype=torch.float32) # accumulate in float32

    grid = lambda meta: (n_rows, )
    _cross_entropy_loss_kernel[grid](
        predictions, predictions.stride(0), predictions.stride(1),
        targets, targets.stride(0),
        out,
        n_rows, n_classes,
        REDUCTION,
    )

    return out.to(torch.float16) # return float16