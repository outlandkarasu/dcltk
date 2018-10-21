__kernel void product(
        __global const float *lhs,
        __global const float *rhs,
        __global float *result,
        uint rows,
        uint cols,
        uint resultCols,
        __local float *localLhs,
        __local float *localRhs) {
    const size_t localJ = get_local_id(0);
    const size_t localCols = get_local_size(0);
    const size_t localI = get_local_id(1);
    const size_t localRows = get_local_size(1);
    const size_t globalJ = get_global_id(0);
    const size_t globalI = get_global_id(1);

    float value = 0.0f;

    for(size_t k = 0; k < cols; k += localCols) {
        barrier(CLK_LOCAL_MEM_FENCE);
        localLhs[localI * localCols + localJ] = lhs[globalI * cols + (k + localJ)];
        localRhs[localI * localCols + localJ] = rhs[(k + localI) * resultCols + globalJ];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(size_t lk = 0; lk < localCols; ++lk) {
            value += localLhs[localI * localCols + lk] * localRhs[lk * localCols + localJ];
        }
    }
    result[globalI * resultCols + globalJ] = value;
}
