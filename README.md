# convolution-c-simd-intrinsics-asm
Convolution in c + SIMD Intrinsics + SIMD ASM by ChatGPT Experiment

# get going
`gcc main.c -o gaussian_filter -O3 -march=armv8-a+simd -lm`

`clang-19 -O3 -o gaussian_filter main.c -lm -lpthread -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -march=armv8-a+simd`

## notes

Using #pragmas in the pure c version does not seem to speed up the code, so its probably optimised very well through the -O3 option.

Base Version vs SIMD ASM seems to be slightly over 2x faster with double (64bit) float var size on a M1 Pro MBP with 128 Bit SIMD NEON
(>2 hints to other optimisations beside SIMD, like loop unrolling, ...)
Example results: 0.0199s vs. 0.0089s