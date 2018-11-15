enum {
    BATCH_SIZE = 16,
    VECTOR_SIZE = 16
};

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float16 *lhs,
        __global const float16 *rhsT,
        __global float *result,
        uint cols) {
    const size_t j = get_global_id(0);
    const size_t i = get_global_id(1);
    const size_t resultCols = get_global_size(1);
    const size_t localJ = get_local_id(0);
    const size_t localI = get_local_id(1);

    __local float16 localLhs[BATCH_SIZE][BATCH_SIZE] __attribute__((xcl_array_partition(cyclic, 16, 2)));
    __local float16 localRhsT[BATCH_SIZE][BATCH_SIZE] __attribute__((xcl_array_partition(cyclic, 16, 2)));

    float value = 0.0f;
    for(size_t k = 0; k < cols; k += VECTOR_SIZE * BATCH_SIZE) {

        barrier(CLK_LOCAL_MEM_FENCE);
        localLhs[localI][localJ] = lhs[(i * cols + k) / VECTOR_SIZE + localJ];
        localRhsT[localJ][localI] = rhsT[(j * cols + k) / VECTOR_SIZE + localI];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t lk = 0; lk < BATCH_SIZE; ++lk) {
            value += dot(localLhs[localI][lk], localRhsT[localJ][lk]);
        }
    }
    result[i * resultCols + j] = value;
}

