enum {
    BATCH_ROWS = 128,
    BATCH_COLS = 128,
    BATCH_SIZE_K = 32,
};

static void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        float result[][BATCH_COLS],
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
            float values[BATCH_ROWS][BATCH_COLS];
            for(size_t pi = 0; pi < BATCH_ROWS; ++pi) {
                for(size_t pj = 0; pj < BATCH_COLS; ++pj) {
                    values[pi][pj] = 0.0f;
                }
            }

            for(size_t k = 0; k < cols; k += BATCH_SIZE_K) {
                productBatch(lhs, rhs, values, rows, cols, resultCols, i, j, k);
            }

            for(size_t pi = 0; pi < BATCH_ROWS; ++pi) {
                for(size_t pj = 0; pj < BATCH_COLS; ++pj) {
	            result[(i + pi) * resultCols + j + pj] = values[pi][pj];
                }
            }
        }
    }
}

void productBatch(
        __global const float *lhs,
        __global const float *rhs,
        float values[][BATCH_COLS],
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
            float value = 0.0f;
	    for(size_t k = 0; k < BATCH_SIZE_K; ++k) {
                const size_t globalK = k + offsetK;
	        value += lhs[globalI * cols + globalK] * rhs[globalK * resultCols + globalJ];
	    }
	    values[i][j] = value;
        }
    }
}

