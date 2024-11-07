# convolution-c-simd-intrinsics-asm
Convolution in c + SIMD Intrinsics + SIMD ASM by ChatGPT Experiment

# get going
`gcc main.c -o gaussian_filter -O3 -march=armv8-a+simd -lm`

`clang-19 -O3 -o gaussian_filter main.c -lm -lpthread -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -march=armv8-a+simd`

## notes

### First runs

Using #pragmas in the pure c version does not seem to speed up the code, so its probably optimised very well through the -O3 option.

Base Version vs SIMD ASM seems to be slightly over 2x faster with double (64bit) float var size on a M1 Pro MBP with 128 Bit SIMD NEON
(>2 hints to other optimisations beside SIMD, like loop unrolling, ...)
Execution time for Basic C version: 0.0199 seconds
Execution time for NEON Assembly version: 0.00887 seconds

### new pragmas

Execution time for Basic C version: 0.107987 seconds
Execution time for NEON Intrinsics version: 0.015030 seconds
Execution time for NEON Assembly version: 0.008848 seconds

    // Disable specific optimizations for Version 1
    #pragma GCC push_options
    #pragma GCC optimize ("no-unroll-loops")
    #pragma GCC optimize ("no-tree-vectorize")
    #pragma GCC optimize ("no-inline")
    // Disable all optimizations for Version 1
    #pragma clang optimize off

    ...

    // Re-enable optimizations after this function
    #pragma clang optimize on
    // Restore global optimization settings
    #pragma GCC pop_options

### conclusion

Execution time for Basic C version without Optimisation: 0.107987 seconds
Execution time for Basic C version: 0.0199 seconds
Execution time for NEON Intrinsics version: 0.015030 seconds
Execution time for NEON Assembly version: 0.008848 seconds

The base version execution time is over 10 times slower than the best optimised version.
Automatic optimisation of the basic version through -O3 is powerful an gains > 5x speedup already.
Intrinsics increase the speed another 25%
NEON assembly yields amazing results but usually would take lots of manual work and knowledge. ChatGPT was surprisingly good at creating code that compiles without error. Especially intrinsics and even asm!

### the catch

`Outputs do not match; there may be an issue with one of the implementations.`

The performance results MAY be influenced by potential defective code/behaviour, as the results seem to differ compared to the basic version.
I have not checked those results so far.
