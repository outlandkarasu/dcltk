enum {
    BATCH_SIZE = 2,
    VECTOR_SIZE = 2,
};

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float *lhs,
        __global const float *rhsT,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols) {
    const size_t batchJ = get_global_id(0) * BATCH_SIZE;
    const size_t batchI = get_global_id(1) * BATCH_SIZE;
    for(size_t i = 0; i < BATCH_SIZE; ++i) {
        const size_t globalI = i + batchI;
        for(size_t j = 0; j < BATCH_SIZE; ++j) {
            const size_t globalJ = j + batchJ;
            float value = 0.0f;
            for(size_t k = 0; k < cols; ++k) {
                value += lhs[globalI * cols + k] * rhsT[globalJ * cols + k];
            }
            result[globalI * resultCols + globalJ] = value;
        }
    }
}

