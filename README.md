# NumC

Various optimizations:
1. SIMD data-level parallelism: this was used in matrix addition, subtraction, multiplication, and powering. This allowed speedups of 4x.
2. OpenMP API tread-level parallelism: these methods were also used in matrix addition, subtraction, mutliplication, and powering. Also 
3. Loop Unrolling: By having each iteration of the loop complete more code, the CPU did not have to branch as many times and could take advantage of its pipelining. Therefore, the number of times the instructions processed were flushed due to a branch instruction was reduced 4-fold, and a furhter speedup was achieved.
4. Repeated Squares Algorithm: Drastically reduces runtime complexity of the matrix powering method. Say we were trying to compute the power of a nxn matrix to the mth power. The initial idea was iteratively calling matrix multiplication in order to compute high powers of large matrices (sizes over 500x500) but this became way too slow as matrix multiplication was already cubic polynomial time, and had to loop m number of times. The repeated squares algorithm reduced the number of iterations in the loop down to logarithmic time. Allowing a total runtime of O(n^3 * log(m)). This allowed matrix powering to achieve a 700x speedup in comparison to the intial method on my local machine.
