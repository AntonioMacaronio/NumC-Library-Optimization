#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in ROW MAJOR ORDER.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int numRows = mat->rows;
    int numCols = mat->cols;
    return *((mat->data) + row * numCols + col);
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in ROW MAJOR ORDER.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int numRows = mat->rows;
    int numCols = mat->cols;
    *((mat->data) + row * numCols + col) = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) {
        return -1;
    } 

    matrix* address = calloc(1, sizeof(matrix));
    if (address == NULL) {
        return -2;
    }
    double* data = calloc(1, rows * cols * sizeof(double));
    if (data == NULL) {
        return -2;
    }
    address->data = data;
    address->rows = rows;
    address->cols = cols;
    address->parent = NULL;
    address->ref_cnt = 1;
    *mat = address;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (!mat) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
            return;
        }    
    } else {
            deallocate_matrix(mat->parent);
            free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if ((rows <= 0) || (cols <= 0)) {
        return -1;
    }
    matrix* new = calloc(1, sizeof(matrix));
    if (new == NULL) {
        return -2;
    }
    new->data = from->data + offset;
    new->rows = rows;
    new->cols = cols;
    new->parent = from;
    from->ref_cnt += 1;
    *mat = new;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in ROW MAJOR ORDER.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int numRows = mat->rows;
    int numCols = mat->cols;
    
    double* matData = mat->data;
    __m256d valueVector = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (int i = 0; i < numRows * numCols/4 * 4; i = i + 4) {
        _mm256_storeu_pd(mat->data, valueVector);
    }
    for (int i = numRows * numCols/4 * 4; i < numRows * numCols; i++) {
        mat->data[i] = val;
    }
    return;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int numRows = mat->rows;
    int numCols = mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < numRows * numCols/4 * 4; i = i + 4) {
        // x vector
        __m256d xVector = _mm256_loadu_pd(mat->data + i);
        // -x vector
        __m256d neg_oneVector = _mm256_set1_pd(-1);
        __m256d neg_xVector = _mm256_mul_pd(xVector, neg_oneVector);

        // the absolute value is the max(-x, x)
        __m256d absoluteVector = _mm256_max_pd(xVector, neg_xVector);
        // store that into the address
        _mm256_storeu_pd(result->data + i, absoluteVector);
    } 
    for (int i = numRows * numCols/4 * 4; i < numRows * numCols; i++) {
        result->data[i] = fabs(mat->data[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int numRows = mat->rows;
    int numCols = mat->cols;
    for (int i = 0; i < numRows * numCols; i++) {
        result->data[i] = -1.0 * (mat->data[i]);
    } 
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int numRows = mat1->rows;
    int numCols = mat1->cols;
    #pragma omp parallel for
    for (int i = 0; i < numRows * numCols/4 * 4; i = i + 4) {
        __m256d mat1Vector = _mm256_loadu_pd(mat1->data + i);
        __m256d mat2Vector = _mm256_loadu_pd(mat2->data + i);
        __m256d sumVector = _mm256_add_pd(mat1Vector, mat2Vector);
        _mm256_storeu_pd(result->data + i, sumVector);
    } 

    for (int i = numRows * numCols / 4 * 4; i < numRows * numCols; i++) {
        result->data[i] = (mat1->data[i]) + (mat2->data[i]);
    } 
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int numRows = mat1->rows;
    int numCols = mat1->cols;
    for (int i = 0; i < numRows * numCols; i++) {
        result->data[i] = (mat1->data[i]) - (mat2->data[i]);
    } 
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    int m1numRows = mat1->rows;
    int m1numCols = mat1->cols;
    int m2numRows = mat2->rows;
    int m2numCols = mat2->cols;

    matrix *m2transposed = malloc(sizeof(matrix));
    m2transposed->rows = m2numCols;
    m2transposed->cols = m2numRows;
    m2transposed->data = malloc(m2numRows * m2numCols * sizeof(double));
    // #pragma omp parallel for
    for (int row = 0; row < m2numRows; row++) {
        for (int col = 0; col < m2numCols; col++) {
            m2transposed->data[row + col * m2numRows] = mat2->data[col + row * m2numCols];
        }
    }
    #pragma omp parallel for collapse(2)
    for (int row1 = 0; row1 < m1numRows; row1++) {
        for (int row2 = 0; row2 < m2transposed->rows; row2++) {
            double sum = 0;
            __m256d sumVector = _mm256_set_pd(0, 0, 0, 0); 
            for (int i = 0; i < m1numCols / 16 * 16; i = i + 16) {
                __m256d m1row = _mm256_loadu_pd(mat1->data + (row1 * m1numCols) + i);
                __m256d m2_transposed_row = _mm256_loadu_pd(m2transposed->data + (row2 * m2transposed->cols) + i);
                sumVector = _mm256_fmadd_pd(m1row, m2_transposed_row, sumVector);

                m1row = _mm256_loadu_pd(mat1->data + (row1 * m1numCols) + i + 4);
                m2_transposed_row = _mm256_loadu_pd(m2transposed->data + (row2 * m2transposed->cols) + i + 4);
                sumVector = _mm256_fmadd_pd(m1row, m2_transposed_row, sumVector); 

                m1row = _mm256_loadu_pd(mat1->data + (row1 * m1numCols) + i + 8);
                m2_transposed_row = _mm256_loadu_pd(m2transposed->data + (row2 * m2transposed->cols) + i + 8);
                sumVector = _mm256_fmadd_pd(m1row, m2_transposed_row, sumVector); 

                m1row = _mm256_loadu_pd(mat1->data + (row1 * m1numCols) + i + 12);
                m2_transposed_row = _mm256_loadu_pd(m2transposed->data + (row2 * m2transposed->cols) + i + 12);
                sumVector = _mm256_fmadd_pd(m1row, m2_transposed_row, sumVector);  
            }
            double temp_arr[4];
            _mm256_storeu_pd(temp_arr, sumVector);
            sum += temp_arr[0] + temp_arr[1] + temp_arr[2] + temp_arr[3];

            // tail case:
            for (int i = m1numCols / 16 * 16; i < m1numCols; i++) {
                sum += mat1->data[(row1 * m1numCols) + i] * m2transposed->data[(row2 * m1numCols) + i];
            }
            result->data[row1 * m2numCols + row2] = sum;
        }
    }
    free(m2transposed->data);
    free(m2transposed);
    return 0;
    // for (int row = 0; row < m1numRows; row++) {
    //     for (int col = 0; col < m2numCols; col++) {
    //         int sum = 0;
    //         for (int i = 0; i < m1numCols; i++) {
    //             sum += mat1->data[(row * m1numCols) + i] * mat2->data[(i * m2numCols) + col];
    //         }
    //         result->data[row * m2numCols + col] = sum;
    //     }
    // }
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in ROW MAJOR ORDER.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int size = mat->rows;
    if (pow == 0) {
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                if (row == col) {
                    result->data[row * size + col] = 1;
                } else {
                    result->data[row * size + col] = 0;
                }
            }
        }
        return 0;
    }
    // result = mat;
    for (int i = 0; i < size * size; i++) {
        result->data[i] = mat->data[i];
    }
    if (pow == 1) {
        return 0;
    }
    // create identity matrix which is y
    matrix *y = calloc(1, sizeof(matrix));
    y->data = malloc(size * size * sizeof(double));
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            if (row == col) {
                y->data[row * size + col] = 1;
            } else {
                y->data[row * size + col] = 0;
            }
        }        
    } 
    matrix *tempX = calloc(1, sizeof(matrix));
    tempX->rows = mat->rows;     // mat->rows = size
    tempX->cols = mat->cols;     // mat->cols = size
    tempX->data = malloc(size * size * sizeof(double));

    matrix *tempY = calloc(1, sizeof(matrix));
    tempY->rows = mat->rows;     // mat->rows = size
    tempY->cols = mat->cols;     // mat->cols = size
    tempY->data = malloc(size * size * sizeof(double));
    while(pow > 1) {
        // make tempX equal to mat
        #pragma omp parallel for
        for (int i = 0; i < size * size; i++) {
            tempX->data[i] = result->data[i];
        } 
        if (pow % 2 == 0) {
            mul_matrix(result, tempX, tempX);
            pow = pow / 2;
        } 
        else {
            // make tempY = Y
            #pragma omp parallel for
            for (int i = 0; i < size * size; i++) {
                tempY->data[i] = y->data[i];
            }
            mul_matrix(y, tempX, tempY);
            mul_matrix(result, tempX, tempX);
            pow = (pow - 1) / 2;
        }
    }
    for (int i = 0; i < size * size; i++) {
        tempX->data[i] = result->data[i];
    }
    for (int i = 0; i < size * size; i++) {
        tempY->data[i] = y->data[i];
    }
    mul_matrix(result, tempX, tempY);
    free(tempX->data);
    free(tempY->data);
    free(y->data);
    free(tempX);
    free(tempY);
    free(y);
    return 0;
}
