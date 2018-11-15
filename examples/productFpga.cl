enum {
    BATCH_SIZE = 16,
    VECTOR_SIZE = 16
};

static void productBatch(
    __global const float16 *lhs,
    __global const float16 *rhsT,
    __global float *result,
    uint rows,
    uint cols,
    uint resultCols,
    size_t batchI,
    size_t batchJ);

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {
    for(size_t i = 0; i < rows; i += BATCH_SIZE) {
        for(size_t j = 0; j < resultCols; j += BATCH_SIZE) {
            productBatch(lhs, rhsT, result, rows, cols, resultCols, i, j);
        }
    }
}

static void productBatch(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        size_t batchI,
        size_t batchJ) {
    for(size_t i = 0; i < BATCH_SIZE; ++i) {
        for(size_t j = 0; j < BATCH_SIZE; ++j) {
            const size_t globalI = batchI + i;
            const size_t globalJ = batchJ + j;
            float value = 0.0f;
            for(size_t k = 0; k < cols; k += VECTOR_SIZE) {
                value += dot(lhs[(globalI * cols + k) / VECTOR_SIZE], rhsT[(globalJ * cols + k) / VECTOR_SIZE]);
            }
            result[globalI * resultCols + globalJ] = value;
        }
    }
}

