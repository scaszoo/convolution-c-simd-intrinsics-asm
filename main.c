#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <math.h>

#define BLOCK_SIZE 64

#define N 512
#define M 512
#define KERNEL_SIZE 11

// Function prototypes for each version
void gaussian_filter_basic(double *input, double *output, int n, int m, double kernel[KERNEL_SIZE][KERNEL_SIZE]);
void gaussian_filter_neon_intrinsics(double *input, double *output, int n, int m, double kernel[KERNEL_SIZE][KERNEL_SIZE]);
void gaussian_filter_neon_asm(double *input, double *output, int n, int m, double kernel[KERNEL_SIZE][KERNEL_SIZE]);

// Helper function to initialize an example Gaussian kernel
void initialize_gaussian_kernel(double kernel[KERNEL_SIZE][KERNEL_SIZE]) {
    double sigma = 1.0;
    double sum = 0.0;
    int offset = KERNEL_SIZE / 2;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            double x = i - offset;
            double y = j - offset;
            kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    // Normalize kernel
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel[i][j] /= sum;
        }
    }
}

// Helper function to initialize an input matrix with random values
void initialize_matrix(double *matrix, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = (double)rand() / RAND_MAX;
        }
    }
}

// Disable specific optimizations for Version 1
#pragma GCC push_options
#pragma GCC optimize ("no-unroll-loops")
#pragma GCC optimize ("no-tree-vectorize")
#pragma GCC optimize ("no-inline")
// Disable all optimizations for Version 1
#pragma clang optimize off
void gaussian_filter_basic(double *input, double *output, int n, int m, double kernel[11][11]) {
    int pad = 5; // Padding for a 11x11 kernel is 5 pixels
    for (int i = pad; i < n - pad; i++) {
        for (int j = pad; j < m - pad; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < 11; ki++) {
                for (int kj = 0; kj < 11; kj++) {
                    int ni = i + ki - pad;
                    int nj = j + kj - pad;
                    sum += input[ni * m + nj] * kernel[ki][kj];
                }
            }
            output[i * m + j] = sum;
        }
    }
}
// Re-enable optimizations after this function
#pragma clang optimize on
// Restore global optimization settings
#pragma GCC pop_options

void gaussian_filter_neon_intrinsics(double *input, double *output, int n, int m, double kernel[11][11]) {
    int pad = 5; // Padding for a 11x11 kernel

    for (int i = pad; i < n - pad; i++) {
        for (int j = pad; j < m - pad; j++) {
            float64x2_t sum_vec = vdupq_n_f64(0.0); // Initialize sum vector to zero

            for (int ki = 0; ki < 11; ki++) {
                for (int kj = 0; kj < 11; kj += 2) { // Process 2 elements at a time for SIMD
                    int ni = i + ki - pad;
                    int nj = j + kj - pad;

                    // Load two adjacent elements from input and kernel
                    float64x2_t input_vec = vld1q_f64(&input[ni * m + nj]);
                    float64x2_t kernel_vec = vld1q_f64(&kernel[ki][kj]);

                    // Multiply input and kernel vectors, then accumulate to sum_vec
                    sum_vec = vfmaq_f64(sum_vec, input_vec, kernel_vec);
                }
            }

            // Sum up the results in sum_vec horizontally and store in output
            output[i * m + j] = vaddvq_f64(sum_vec); // Horizontal sum
        }
    }
}

// void gaussian_filter_neon_asm(double *input, double *output, int n, int m, double kernel[11][11]) {
//     int pad = 5;

//     for (int i = pad; i < n - pad; i++) {
//         for (int j = pad; j < m - pad; j++) {
//             double sum = 0.0;

//             asm volatile (
//                 "mov x0, %[sum]            \n"  // Register for the sum
//                 "mov x1, %[input]          \n"  // Input matrix pointer
//                 "mov x2, %[kernel]         \n"  // Kernel pointer
//                 "mov x3, %[m]              \n"  // Width (stride)

//                 // Outer loop over kernel rows
//                 "mov w4, #0                \n"
//                 "1:                        \n"  // Outer loop label

//                 // Inner loop over kernel columns, processing two elements at a time
//                 "mov w5, #0                \n"
//                 "2:                        \n"  // Inner loop label

//                 // Calculate input and kernel positions
//                 "add x6, x1, x4, LSL #3    \n" // x6 = &input[i * m + kj]
//                 "add x7, x2, x4, LSL #3    \n" // x7 = &kernel[ki][kj]

//                 // Load two double elements from input and kernel into NEON registers
//                 "ld1 {v0.2d}, [x6]         \n" // Load input into v0
//                 "ld1 {v1.2d}, [x7]         \n" // Load kernel into v1

//                 // Multiply and accumulate
//                 "fmla v2.2d, v0.2d, v1.2d  \n" // Accumulate result into v2

//                 // Increment kernel column index and loop
//                 "add w5, w5, #2            \n"
//                 "cmp w5, #11               \n"
//                 "b.lt 2b                   \n" // Repeat inner loop

//                 // Increment kernel row index and loop
//                 "add w4, w4, #1            \n"
//                 "cmp w4, #11               \n"
//                 "b.lt 1b                   \n" // Repeat outer loop

//                 // Sum across vector and store result
//                 "faddp v2.2d, v2.2d, v2.2d \n" // Pairwise add
//                 "str d2, [%[output], x3]   \n" // Store the result in output

//                 : [sum] "+r" (sum)
//                 : [input] "r" (input), [kernel] "r" (kernel), [m] "r" (m), [output] "r" (output)
//                 : "memory", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "v0", "v1", "v2"
//             );

//             output[i * m + j] = sum;
//         }
//     }
// }

void gaussian_filter_neon_asm(double *input, double *output, int n, int m, double kernel[11][11]) {
    int pad = 5; // Padding for a 11x11 kernel

    // Process the image in BLOCK_SIZE x BLOCK_SIZE blocks
    for (int bi = pad; bi < n - pad; bi += BLOCK_SIZE) {
        for (int bj = pad; bj < m - pad; bj += BLOCK_SIZE) {

            // Process each block
            for (int i = bi; i < bi + BLOCK_SIZE && i < n - pad; i++) {
                for (int j = bj; j < bj + BLOCK_SIZE && j < m - pad; j++) {
                    double sum = 0.0;

                    // Prefetch next row of input and kernel for improved memory access
                    __builtin_prefetch(&input[(i + 1) * m + j], 0, 1);

                    asm volatile (
                        "mov x0, %[sum]             \n" // Accumulator for sum
                        "mov x1, %[input]           \n" // Input matrix base pointer
                        "mov x2, %[kernel]          \n" // Kernel base pointer
                        "mov x3, %[m]               \n" // Image stride

                        // Initialize accumulators for partial sums
                        "dup v8.2d, x0              \n" // Accumulator for sum

                        // Outer loop over kernel rows
                        "mov w4, #0                 \n" // ki = 0
                        "1:                          \n" // Outer loop label

                        // Inner loop with unrolling by 2, processing 4 elements at a time
                        "mov w5, #0                 \n" // kj = 0
                        "2:                          \n" // Inner loop label

                        // Calculate input and kernel positions
                        "add x6, x1, x4, LSL #3     \n" // &input[i * m + kj]
                        "add x7, x2, x4, LSL #3     \n" // &kernel[ki][kj]

                        // Prefetch for cache optimization
                        "prfm pldl1keep, [x6, #64]  \n" // Prefetch input data 64 bytes ahead

                        // Load four double elements from input and kernel
                        "ld1 {v0.2d, v1.2d}, [x6]   \n" // Load 4 doubles from input
                        "ld1 {v2.2d, v3.2d}, [x7]   \n" // Load 4 doubles from kernel

                        // Multiply-accumulate using FMA instructions
                        "fmla v8.2d, v0.2d, v2.2d   \n" // Multiply-accumulate
                        "fmla v8.2d, v1.2d, v3.2d   \n" // Multiply-accumulate

                        // Increment kj by 4 for next SIMD load
                        "add w5, w5, #4             \n"
                        "cmp w5, #11                \n"
                        "b.lt 2b                    \n" // Repeat inner loop

                        // Increment ki by 1 for next kernel row
                        "add w4, w4, #1             \n"
                        "cmp w4, #11                \n"
                        "b.lt 1b                    \n" // Repeat outer loop

                        // Horizontal addition of accumulator v8
                        "faddp v8.2d, v8.2d, v8.2d  \n" // Pairwise addition
                        "str d8, [%[output], x3]    \n" // Store the result in output

                        : [sum] "+r" (sum)
                        : [input] "r" (input), [kernel] "r" (kernel), [m] "r" (m), [output] "r" (output)
                        : "memory", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v8"
                    );

                    output[i * m + j] = sum;
                }
            }
        }
    }
}

// Function to measure the execution time of a function
double measure_execution_time(void (*func)(double *, double *, int, int, double[KERNEL_SIZE][KERNEL_SIZE]), 
                              double *input, double *output, int n, int m, double kernel[KERNEL_SIZE][KERNEL_SIZE]) {
    clock_t start = clock();
    func(input, output, n, m, kernel);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    // Allocate memory for input and output matrices
    double *input = (double *)malloc(N * M * sizeof(double));
    double *output_basic = (double *)malloc(N * M * sizeof(double));
    double *output_intrinsics = (double *)malloc(N * M * sizeof(double));
    double *output_asm = (double *)malloc(N * M * sizeof(double));

    // Initialize input matrix and Gaussian kernel
    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    initialize_matrix(input, N, M);
    initialize_gaussian_kernel(kernel);

    // Run and time Version 1 (Basic C version)
    double time_basic = measure_execution_time(gaussian_filter_basic, input, output_basic, N, M, kernel);
    printf("Execution time for Basic C version: %.6f seconds\n", time_basic);

    // Run and time Version 2 (NEON Intrinsics version)
    double time_intrinsics = measure_execution_time(gaussian_filter_neon_intrinsics, input, output_intrinsics, N, M, kernel);
    printf("Execution time for NEON Intrinsics version: %.6f seconds\n", time_intrinsics);

    // Run and time Version 3 (NEON Assembly version)
    double time_asm = measure_execution_time(gaussian_filter_neon_asm, input, output_asm, N, M, kernel);
    printf("Execution time for NEON Assembly version: %.6f seconds\n", time_asm);

    // Validate that outputs are similar (optional, for correctness check)
    int correct = 1;
    for (int i = 0; i < N * M; i++) {
        if (fabs(output_basic[i] - output_intrinsics[i]) > 1e-6 || fabs(output_basic[i] - output_asm[i]) > 1e-6) {
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("All outputs match within acceptable tolerance.\n");
    } else {
        printf("Outputs do not match; there may be an issue with one of the implementations.\n");
    }

    // Free allocated memory
    free(input);
    free(output_basic);
    free(output_intrinsics);
    free(output_asm);

    return 0;
}
