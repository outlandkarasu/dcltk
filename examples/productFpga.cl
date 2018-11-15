enum {
    BATCH_ROWS = 16,
    BATCH_COLS = 16,
    BATCH_K = 16,
    VECTOR_SIZE = 16
};

__kernel
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint cols) {
    const size_t j = get_global_id(0);
    const size_t i = get_global_id(1);
    const size_t resultCols = get_global_size(1);

    float value = 0.0f;
    for(size_t k = 0; k < cols; k += VECTOR_SIZE) {
        value += dot(lhs[(i * cols + k) / VECTOR_SIZE], rhsT[(j * cols + k) / VECTOR_SIZE]);
    }
    result[i * resultCols + j] = value;
}

