import os
import random
import statistics
import subprocess
import tempfile
import time
from typing import Dict, List


def compute_speedups(
    matmul_func: str, matrix_sizes: List[int] = [512], num_trials: int = 5, debug=False
) -> Dict[int, float]:
    # C program template
    c_template = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <string.h>
    #include <math.h>

    // Student implementation
    <<MATMUL_IMPL>>

    // Reference implementation
    void matmul_reference(float* A, float* B, float* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    int main(int argc, char* argv[]) {
        if (argc != 2) {
            printf("Usage: ./matmul <matrix_size>\\n");
            return 1;
        }
        
        const int N = atoi(argv[1]);
        const int SIZE = N * N;
        
        // Allocate matrices
        float *A = (float*)malloc(SIZE * sizeof(float));
        float *B = (float*)malloc(SIZE * sizeof(float));
        float *C1 = (float*)malloc(SIZE * sizeof(float));
        float *C2 = (float*)malloc(SIZE * sizeof(float));
        
        if (!A || !B || !C1 || !C2) {
            printf("Memory allocation failed\\n");
            return 1;
        }
        
        // Initialize with random values
        srand(42);
        for (int i = 0; i < SIZE; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }
        
        // Time reference implementation
        clock_t start = clock();
        matmul_reference(A, B, C1, N);
        clock_t ref_time = clock() - start;
        
        // Time student implementation
        start = clock();
        matmul(A, B, C2, N);
        clock_t student_time = clock() - start;
        
        // Check correctness
        for (int i = 0; i < SIZE; i++) {
            if (fabs(C1[i] - C2[i]) > 1e-4) {
                printf("MISMATCH\\n");
                free(A);
                free(B);
                free(C1);
                free(C2);
                return 1;
            }
        }
        
        // Print timing ratio
        printf("%f\\n", (double)ref_time / student_time);
        
        free(A);
        free(B);
        free(C1);
        free(C2);
        return 0;
    }
    """

    results = {}

    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and write C file
            c_file = os.path.join(tmpdir, "matmul.c")
            with open(c_file, "w") as f:
                f.write(c_template.replace("<<MATMUL_IMPL>>", matmul_func))

            # Compile with optimizations
            compile_result = subprocess.run(
                [
                    "gcc",
                    "-O3",
                    "-march=native",
                    c_file,
                    "-o",
                    os.path.join(tmpdir, "matmul"),
                ],
                capture_output=True,
                text=True,
            )

            if compile_result.returncode != 0:
                if debug:
                    print(f"Compilation failed: {compile_result.stderr}")
                return {size: 0.0 for size in matrix_sizes}

            # Test each matrix size
            for size in matrix_sizes:
                speedups = []

                # Run multiple trials
                for trial in range(num_trials):
                    run_result = subprocess.run(
                        [os.path.join(tmpdir, "matmul"), str(size)],
                        capture_output=True,
                        text=True,
                    )

                    if run_result.returncode != 0 or "MISMATCH" in run_result.stdout:
                        if debug:
                            print(f"Runtime error or incorrect results for size {size}")
                        results[size] = 0.0
                        break

                    try:
                        speedup = float(run_result.stdout.strip())
                        speedups.append(speedup)
                    except ValueError:
                        if debug:
                            print(f"Failed to parse speedup value for size {size}")
                        results[size] = 0.0
                        break

                if speedups:  # If we have valid results
                    # Calculate average speedup for this size
                    results[size] = statistics.mean(speedups)
                    if debug:
                        print(
                            f"Matrix size {size}x{size}: Average speedup {results[size]:.2f}x "
                            f"(min: {min(speedups):.2f}x, max: {max(speedups):.2f}x)"
                        )

    except Exception as e:
        if debug:
            print(f"Unexpected error: {e}")
        return {size: 0.0 for size in matrix_sizes}

    return results


# Example usage
ref_impl = """
```c
void matmul(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```
"""

better_impl = """
```c
void matmul(float* A, float* B, float* C, int N) {
    const int BLOCK_SIZE = 32;  // Tune this based on your cache size
    
    // Zero the output matrix
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }
    
    // Block multiplication
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                // Mini-matrix multiplication
                for (int i = i0; i < (i0 + BLOCK_SIZE < N ? i0 + BLOCK_SIZE : N); i++) {
                    for (int j = j0; j < (j0 + BLOCK_SIZE < N ? j0 + BLOCK_SIZE : N); j++) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < (k0 + BLOCK_SIZE < N ? k0 + BLOCK_SIZE : N); k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}
```
"""

o3_impl = """
void matmul(float* A, float* B, float* C, int N) {
    // Transpose B for better cache locality.
    float *BT = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j * N + i] = B[i * N + j];
        }
    }
    
    // Zero initialize C.
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }
    
    // Blocked matrix multiplication.
    int BS = 32; // Block size (tune this for your architecture)
    for (int i = 0; i < N; i += BS) {
        for (int j = 0; j < N; j += BS) {
            for (int k = 0; k < N; k += BS) {
                int i_max = (i + BS < N) ? i + BS : N;
                int j_max = (j + BS < N) ? j + BS : N;
                int k_max = (k + BS < N) ? k + BS : N;
                for (int ii = i; ii < i_max; ii++) {
                    for (int jj = j; jj < j_max; jj++) {
                        float sum = C[ii * N + jj];
                        for (int kk = k; kk < k_max; kk++) {
                            sum += A[ii * N + kk] * BT[jj * N + kk];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
    
    free(BT);
}
"""


def extract_answer(output: str):
    solution = output.split("```c")[-1].split("```")[0]
    return solution


# reward function to be used in veRL
def compute_score(data_source, solution_str, ground_truth):
    answer = extract_answer(solution_str)
    speedups = compute_speedups(answer)
    score = statistics.mean(speedups.values())
    return score


if __name__ == "__main__":
    score = compute_score("", o3_impl, "")
    print(score)
