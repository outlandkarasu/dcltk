__kernel void product(
    __global const float *lhs,
    __global const float *rhs,
    __global float *result,
    uint rows,
    uint cols,
    uint resultCols,
    __local float *localRow,
    __local float *localCol) {
    const size_t groupI = get_global_id(0);
    const size_t groupRows = get_global_size(0);
    const size_t groupJ = get_global_id(1);
    const size_t groupCols = get_global_size(1);

    const size_t localI = get_local_id(0);
    const size_t localRows = get_local_size(0);
    const size_t localJ = get_local_id(1);
    const size_t localCols = get_local_size(1);

    for(size_t i = 0; i < rows; i += groupRows) {
        for(size_t j = 0; j < resultCols; j += groupCols) {
            float value = 0.0f;

            for(size_t k = 0; k < cols; k += localCols) {

                barrier(CLK_LOCAL_MEM_FENCE);
                if((i + groupI) < rows && (k + localJ) < cols) {
                    localRow[localI * localCols + localJ] = lhs[(i + groupI) * cols + (k + localJ)];
                }
                if((j + groupJ) < resultCols && (k + localI) < cols) {
                    localCol[localI * localCols + localJ] = rhs[(k + localI) * resultCols + (j + groupJ)];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if((i + groupI) < rows && (j + groupJ) < resultCols) {
                    for(size_t lk = 0; lk < localCols && (k + lk) < cols; ++lk) {
                        value += localRow[localI * localCols + lk] * localCol[lk * localCols + localJ];
                    }
                }
            }
            if((i + groupI) < rows && (j + groupJ) < resultCols) {
            	result[(i + groupI) * resultCols + (j + groupJ)] = value;
            }
        }
    }
}
