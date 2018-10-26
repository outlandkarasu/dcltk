__kernel void product(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        __local float *localLhs,
        __local float *localRhs) {
    enum {
        BATCH_SIZE_K = %d
    };
    const size_t localJ = get_local_id(0);
    const size_t localCols = get_local_size(0);
    const size_t localI = get_local_id(1);
    const size_t localRows = get_local_size(1);
    const size_t globalJ = get_global_id(0);
    const size_t globalI = get_global_id(1);

    const size_t groupJ = get_group_id(0) * localCols;
    const size_t groupI = get_group_id(1) * localRows;
    const size_t localSize = get_local_size(0) * get_local_size(1);
    const size_t localId = get_local_id(0) + get_local_size(0) * get_local_id(1);
    const size_t localCopyCountLhs = localRows * BATCH_SIZE_K / localSize;
    const size_t localCopyCountRhs = BATCH_SIZE_K * localCols / localSize;

    float value = 0.0f;

    for(size_t k = 0; k < cols; k += BATCH_SIZE_K) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for(size_t i = 0; i < localCopyCountLhs; ++i) {
            const size_t id = (i * localSize) + localId;
            const size_t row = id / BATCH_SIZE_K;
            const size_t col = id %% BATCH_SIZE_K;
            localLhs[row * BATCH_SIZE_K + col] = lhs[(groupI + row) * cols + (k + col)];
        }
        for(size_t i = 0; i < localCopyCountRhs; ++i) {
            const size_t id = (i * localSize) + localId;
            const size_t row = id / localCols;
            const size_t col = id %% localCols;
            localRhs[row * localCols + col] = rhs[(k + row) * resultCols + (groupJ + col)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t lk = 0; lk < BATCH_SIZE_K; ++lk) {
            value += localLhs[localI * BATCH_SIZE_K + lk] * localRhs[lk * localCols + localJ];
        }
    }
    result[globalI * resultCols + globalJ] = value;
}
