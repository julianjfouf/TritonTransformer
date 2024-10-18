This is still largely a work in progress, but I have published it for documentation purposes. Kernels still in progress: LayerNorm (flaws: numerical accuracy + speed), Attention (flaws: speed).

The Attention kernel is kinda weird though because when I benchmark it in isolation it does extremely well, but when benchmarked as part of the transformer, it does not do very good (possibly due to having to make the q, k, v matrices contiguous).

The LayerNorm kernel may just not be possible to be numerically accurate in float16 (?).

Will pick this repo back up when time permits.
