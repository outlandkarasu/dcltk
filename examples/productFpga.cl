static enum {
    BATCH_ROWS = 128,
    BATCH_COLS = 128,
    BATCH_K = 16,
};

static void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        float batchResults[][BATCH_COLS],
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ,
        size_t offsetK);

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
            float batchResults[BATCH_ROWS][BATCH_COLS];
            for(size_t bi = 0; bi < BATCH_ROWS; ++bi) {
                for(size_t bj = 0; bj < BATCH_COLS; ++bj) {
                    batchResults[bi][bj] = 0.0f;
                }
            }

	    for(size_t k = 0; k < cols; k += BATCH_K) {
                productBatch(lhs, rhs, batchResults, rows, cols, resultCols, i, j, k);
            }

            for(size_t bi = 0; bi < BATCH_ROWS; ++bi) {
                const size_t globalI = i + bi;
                for(size_t bj = 0; bj < BATCH_COLS; ++bj) {
                    const size_t globalJ = j + bj;
                    result[globalI * resultCols + globalJ] = batchResults[bi][bj];
                }
            }
        }
    }
}

void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        float batchResults[][BATCH_COLS],
        uint rows,
        uint cols,
        uint resultCols,
        size_t offsetI,
        size_t offsetJ,
        size_t offsetK) {
    for(size_t i = 0; i < BATCH_ROWS; ++i) {
        const size_t globalI = i + offsetI;
        for(size_t j = 0; j < BATCH_COLS; ++j) {
            const size_t globalJ = j + offsetJ;
	    for(size_t k = 0; k < BATCH_K; ++k) {
	        batchResults[globalI][globalJ] += lhs[globalI * cols + (k + offsetK)] * rhs[(k + offsetK) * resultCols + globalJ];
	    }
        }
    }
}

