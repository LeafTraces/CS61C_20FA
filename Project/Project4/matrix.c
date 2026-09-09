#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

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

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0){
        PyErr_SetString(PyExc_ValueError, "rows and cols must be positive.");
        return -1;
    }

    matrix *m = malloc(sizeof(matrix));
    if (m == NULL){
        PyErr_SetString(PyExc_RuntimeError, "matrix allocation failed.");
        return -1;
    }

    m->data = malloc((size_t) rows * sizeof(double*));
    if (m->data == NULL){
        free(m);
        PyErr_SetString(PyExc_RuntimeError, "matrix data allocation failed.");
        return -1;
    }

    double *element = calloc((size_t) rows * (size_t) cols, sizeof(double));
    if (element == NULL){
        free(m->data);
        free(m);
        PyErr_SetString(PyExc_RuntimeError, "matrix elements allocation failed.");
        return -1;
    }

    for (int i = 0; i < rows; i++){
        m->data[i] = element + (size_t) i * cols;
    }

    m->rows = rows;
    m->cols = cols;
    m->is_1d = (rows==1 || cols==1);
    m->ref_cnt = 1;
    m->parent = NULL;

    *mat = m;

    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {

    // Check dimensions
    if (rows <= 0 || cols <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "rows and cols must be positive");
        return -1;
    }

    // Check slice bounds
    if (from == NULL ||
        row_offset < 0 ||
        col_offset < 0 ||
        rows > from->rows ||
        cols > from->cols ||
        row_offset > from->rows - rows ||
        col_offset > from->cols - cols) {

        PyErr_SetString(PyExc_IndexError,
                        "slice out of bounds");
        return -1;
    }

    matrix *m = malloc(sizeof(matrix));
    if (m == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to allocate matrix");
        return -1;
    }

    m->data = malloc((size_t) rows * sizeof(double *));
    if (m->data == NULL) {
        free(m);

        PyErr_SetString(PyExc_RuntimeError,
                        "failed to allocate matrix data");
        return -1;
    }

    for (int i = 0; i < rows; i++){
        m->data[i] = from->data[row_offset + i] + col_offset;
    }

    m->rows = rows;
    m->cols = cols;
    m->is_1d = (rows == 1 || cols == 1);
    m->ref_cnt = 1;
    m->parent = from;

    from->ref_cnt++;
    *mat = m;

    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    if (mat == NULL){
        return;
    }

    mat->ref_cnt--;

    if (mat->ref_cnt > 0){
        return;
    }

    if (mat->parent == NULL){
        free(mat->data[0]);
        free(mat->data);
        free(mat);
        return;
    }

    matrix *parent = mat->parent;
    free(mat->data);
    free(mat);
    deallocate_matrix(parent);
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    int rows = mat->rows;
    int cols = mat->cols;

    size_t k = (size_t) rows * (size_t) cols;

    double *element = mat->data[0];
    for (size_t i = 0; i < k; i++){
        element[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    size_t k = (size_t) result->rows * (size_t) result->cols;
    double *result_e = result->data[0];
    double *mat1_e = mat1->data[0];
    double *mat2_e = mat2->data[0];

    for (size_t i = 0; i < k; i++){
        result_e[i] = mat1_e[i] + mat2_e[i];
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    size_t k = (size_t) result->rows * (size_t) result->cols;
    double *result_e = result->data[0];
    double *mat1_e = mat1->data[0];
    double *mat2_e = mat2->data[0];

    for (size_t i = 0; i < k; i++){
        result_e[i] = mat1_e[i] - mat2_e[i];
    }
    
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    for (int i = 0; i < result->rows; i++){
        for (int j = 0; j < result->cols; j++){
            double sum = 0;
            for (int p = 0; p < mat1->cols; p++){
                sum += mat1->data[i][p] * mat2->data[p][j];
            }
            result->data[i][j] = sum;
        }
    }
    
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int n = mat->rows;

    if (mat->rows != mat->cols){
        PyErr_SetString(PyExc_ValueError,
                        "rows and cols must be equal");
        return -1;
    
    }
    
    if (pow == 0){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                result->data[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        return 0;
    }

    // acc = I
    matrix *acc = NULL;
    allocate_matrix(&acc, n, n);
    for (int i = 0; i < n; i++){
        acc->data[i][i] = 1.0;
    }

    matrix *temp = NULL;
    allocate_matrix(&temp, n, n);
    for (int p = 0; p < pow; p++){
        mul_matrix(temp, acc, mat);
        
        matrix *swap = acc;
        acc = temp;
        temp = swap;
    }

    memcpy(result->data[0], acc->data[0], (size_t) n * n * sizeof(double));

    deallocate_matrix(acc);
    deallocate_matrix(temp);

    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    size_t k = (size_t) result->rows * (size_t) result->cols;
    double *result_e = result->data[0];
    double *mat_e = mat->data[0];

    for (size_t i = 0; i < k; i++){
        result_e[i] = -mat_e[i];
    }
    
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    size_t k = (size_t) result->rows * (size_t) result->cols;
    double *result_e = result->data[0];
    double *mat_e = mat->data[0];

    for (size_t i = 0; i < k; i++){
        result_e[i] = mat_e[i] < 0 ? -mat_e[i] : mat_e[i];
    }
    
    return 0;
}

