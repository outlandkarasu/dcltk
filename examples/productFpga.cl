#define GROUP_SIZE 64

static void loadToLocal(
    __global const float * restrict lhs,
    __global const float * restrict rhsT,
    __local float localRows[GROUP_SIZE][GROUP_SIZE],
    __local float localCols[GROUP_SIZE][GROUP_SIZE],
    size_t cols,
    size_t k);

static void calculate(
    __local const float localRows[GROUP_SIZE][GROUP_SIZE],
    __local const float localCols[GROUP_SIZE][GROUP_SIZE],
    float * restrict value);

__kernel
__attribute__((reqd_work_group_size(GROUP_SIZE, GROUP_SIZE, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float * restrict lhs,
        __global const float * restrict rhsT,
        __global float * restrict result,
        uint rows,
        uint cols,
        uint resultCols) {
    __local float localRows[GROUP_SIZE][GROUP_SIZE] __attribute__((xcl_array_partition(cyclic, GROUP_SIZE, 2)));
    __local float localCols[GROUP_SIZE][GROUP_SIZE] __attribute__((xcl_array_partition(cyclic, GROUP_SIZE, 2)));
    const size_t j = get_global_id(0);
    const size_t i = get_global_id(1);
    float value = 0.0f;

    for(size_t k = 0; k < cols; k += GROUP_SIZE) {
        loadToLocal(lhs, rhsT, localRows, localCols, cols, k);
        calculate(localRows, localCols, &value);
    }

    result[i * resultCols + j] = value;
}

static void loadToLocal(
        __global const float * restrict lhs,
        __global const float * restrict rhsT,
        __local float localRows[GROUP_SIZE][GROUP_SIZE],
        __local float localCols[GROUP_SIZE][GROUP_SIZE],
        size_t cols,
        size_t k) {
    barrier(CLK_LOCAL_MEM_FENCE);
    __attribute__((xcl_pipeline_workitems)) {
        const size_t localJ = get_local_id(0);
        const size_t localI = get_local_id(1);
        const size_t groupJ = get_group_id(0) * GROUP_SIZE;
        const size_t groupI = get_group_id(1) * GROUP_SIZE;
        localRows[localI][localJ] = lhs[(groupI + localI) * cols + k + localJ];
        localCols[localI][localJ] = rhsT[(groupJ + localI) * cols + k + localJ];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

static void calculate(
        __local const float localRows[GROUP_SIZE][GROUP_SIZE],
        __local const float localCols[GROUP_SIZE][GROUP_SIZE],
        float * restrict value) {
    __attribute__((xcl_pipeline_workitems)) {
        const size_t localJ = get_local_id(0);
        const size_t localI = get_local_id(1);

        __attribute__((opencl_unroll_hint(GROUP_SIZE)))
        for(size_t localK = 0; localK < GROUP_SIZE; ++localK) {
            *value += localRows[localI][localK] * localCols[localJ][localK];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

