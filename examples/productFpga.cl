enum {
    BATCH_ROWS = 128,
    BATCH_COLS = 128,
    BATCH_K = 16,
};

static void productBatch(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ,
        size_t offsetK);

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
void product(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {

    for(size_t i = 0; i < rows; i += BATCH_ROWS) {
        for(size_t j = 0; j < resultCols; j += BATCH_COLS) {
            for(size_t k = 0; k < cols; k += BATCH_K) {
                productBatch(lhs, rhsT, result, rows, cols, resultCols, i, j, k);
            }
        }
    }
}

void productBatch(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ,
        size_t k) {
    for(size_t i = 0; i < BATCH_ROWS; ++i) {
        const size_t globalI = i + offsetI;
        for(size_t j = 0; j < BATCH_COLS; ++j) {
            const size_t globalJ = j + offsetJ;
            float value = 0.0f;
	    value += dot(lhs[(globalI * cols + k) / 16], rhsT[(globalJ * cols + k) / 16]);
	    result[globalI * resultCols + globalJ] += value;
        }
    }
}

