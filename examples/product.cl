__kernel void product(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {
    enum {
        BATCH_ROWS = %d,
        BATCH_COLS = %d,
        BATCH_SIZE_K = %d,
        PRIVATE_ROWS = %d,
        PRIVATE_COLS = %d,
        LOCAL_WORK_COUNT_ROW = BATCH_ROWS / PRIVATE_ROWS,
        LOCAL_WORK_COUNT_COL = BATCH_COLS / PRIVATE_COLS,
        LOCAL_SIZE = LOCAL_WORK_COUNT_ROW * LOCAL_WORK_COUNT_COL,
        LOCAL_COPY_COUNT_LHS = (BATCH_ROWS * BATCH_SIZE_K) / LOCAL_SIZE,
        LOCAL_COPY_COUNT_RHS = (BATCH_SIZE_K * BATCH_COLS) / LOCAL_SIZE
    };

    __local float localLhs[BATCH_SIZE_K][BATCH_ROWS + 2];
    __local float localRhs[BATCH_SIZE_K][BATCH_COLS];

    const size_t localJ = get_local_id(0);
    const size_t localI = get_local_id(1);

    const size_t groupJ = get_group_id(0) * BATCH_COLS;
    const size_t groupI = get_group_id(1) * BATCH_ROWS;

    const size_t localId = get_local_id(0) + get_local_size(0) * get_local_id(1);

    float values[PRIVATE_ROWS][PRIVATE_COLS];
    for(size_t i = 0; i < PRIVATE_ROWS; ++i) {
        for(size_t j = 0; j < PRIVATE_COLS; ++j) {
            values[i][j] = 0.0f;
        }
    }

    for(size_t k = 0; k < cols; k += BATCH_SIZE_K) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for(size_t i = 0; i < LOCAL_COPY_COUNT_LHS; ++i) {
            const size_t id = (i * LOCAL_SIZE) + localId;
            const size_t row = id / BATCH_SIZE_K;
            const size_t col = id %% BATCH_SIZE_K;
            localLhs[col][row] = lhs[(groupI + row) * cols + (k + col)];
        }
        for(size_t i = 0; i < LOCAL_COPY_COUNT_RHS; ++i) {
            const size_t id = (i * LOCAL_SIZE) + localId;
            const size_t row = id / BATCH_COLS;
            const size_t col = id %% BATCH_COLS;
            localRhs[row][col] = rhs[(k + row) * resultCols + (groupJ + col)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t lk = 0; lk < BATCH_SIZE_K; ++lk) {
            float cols[PRIVATE_COLS];
            for(size_t j = 0; j < PRIVATE_COLS; ++j) {
                cols[j] = localRhs[lk][localJ + j * LOCAL_WORK_COUNT_COL];
            }
            for(size_t i = 0; i < PRIVATE_ROWS; ++i) {
                const float row = localLhs[lk][localI + i * LOCAL_WORK_COUNT_ROW];
                for(size_t j = 0; j < PRIVATE_COLS; ++j) {
                    values[i][j] += row * cols[j];
                }
            }
        }
    }
    for(size_t i = 0; i < PRIVATE_ROWS; ++i) {
        const size_t rowOffset = (groupI + localI + i * LOCAL_WORK_COUNT_ROW) * resultCols;
        for(size_t j = 0; j < PRIVATE_COLS; ++j) {
            result[rowOffset + (groupJ + localJ + j * LOCAL_WORK_COUNT_COL)] = values[i][j];
        }
    }
}
