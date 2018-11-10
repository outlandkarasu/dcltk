static enum {
    BATCH_ROWS = 128,
    BATCH_COLS = 128,
};

static void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ);

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
void product(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {

    for(size_t i = 0; i < rows; i += BATCH_ROWS) {
        for(size_t j = 0; j < resultCols; j += BATCH_COLS) {
            productBatch(lhs, rhs, result, rows, cols, resultCols, i, j);
        }
    }
}

void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ) {
    for(size_t i = 0; i < BATCH_ROWS; ++i) {
        const size_t globalI = i + offsetI;
        for(size_t j = 0; j < BATCH_COLS; ++j) {
            const size_t globalJ = j + offsetJ;
            float value = 0.0f;
	    for(size_t k = 0; k < cols; ++k) {
	        value += lhs[globalI * rows + k] * rhs[k * resultCols + globalJ];
	    }
	    result[globalI * resultCols + globalJ] = value;
        }
    }
}

