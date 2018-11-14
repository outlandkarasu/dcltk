enum {
    BATCH_ROWS = 128,
    BATCH_COLS = 128,
    BATCH_K = 16,
};

static void productBatch(
        __global const float *lhs,
        __global const float *rhs,
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
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {

    for(size_t i = 0; i < rows; i += BATCH_ROWS) {
        for(size_t j = 0; j < resultCols; j += BATCH_COLS) {
            for(size_t k = 0; k < cols; k += BATCH_K) {
                productBatch(lhs, rhs, result, rows, cols, resultCols, i, j, k);
            }
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
        size_t offsetJ,
        size_t offsetK) {
    for(size_t i = 0; i < BATCH_ROWS; ++i) {
        bool loadedCols = 0;
        const size_t globalI = i + offsetI;
        float rowBuffer[BATCH_K];
        float colsBuffer[BATCH_COLS][BATCH_K];

        // load a row values.
        for(size_t k = 0; k < BATCH_K; ++k) {
            rowBuffer[k] = lhs[globalI * cols + (offsetK + k)];
        }

        for(size_t j = 0; j < BATCH_COLS; ++j) {
            const size_t globalJ = j + offsetJ;
            float value = 0.0f;

            // load a col values.
            if(!loadedCols) {
	        for(size_t k = 0; k < BATCH_K; ++k) {
                    const size_t globalK = offsetK + k;
                    colsBuffer[j][k] = rhs[globalK * resultCols + globalJ];
	        }
            }

	    for(size_t k = 0; k < BATCH_K; ++k) {
	        value += rowBuffer[k] * colsBuffer[j][k];
	    }

	    result[globalI * resultCols + globalJ] += value;
        }
        loadedCols = 1;
    }
}

