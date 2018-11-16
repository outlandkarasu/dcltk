#define GROUP_SIZE 2
#define BATCH_SIZE 2
#define VECTOR_SIZE 2

typedef float2 VectorType;
#define LOAD_VECTOR vload2

static void loadToLocal(
    __global const float * restrict lhs,
    __global const float * restrict rhsT,
    __local VectorType localRows[GROUP_SIZE][GROUP_SIZE],
    __local VectorType localCols[GROUP_SIZE][GROUP_SIZE],
    size_t cols,
    size_t groupI,
    size_t groupJ,
    size_t k,
    size_t localI,
    size_t localJ);

static void calculate(
    __local const VectorType localRows[GROUP_SIZE][GROUP_SIZE],
    __local const VectorType localCols[GROUP_SIZE][GROUP_SIZE],
    size_t localI,
    size_t localJ,
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
    __local VectorType localRows[GROUP_SIZE][GROUP_SIZE];
    __local VectorType localCols[GROUP_SIZE][GROUP_SIZE];
    const size_t groupJ = get_group_id(0) * GROUP_SIZE;
    const size_t groupI = get_group_id(1) * GROUP_SIZE;
    const size_t localJ = get_local_id(0);
    const size_t localI = get_local_id(1);
    const size_t j = get_global_id(0);
    const size_t i = get_global_id(1);
    float value = 0.0f;

    for(size_t k = 0; k < cols; k += (VECTOR_SIZE * GROUP_SIZE)) {
        barrier(CLK_LOCAL_MEM_FENCE);
        loadToLocal(lhs, rhsT, localRows, localCols, cols, groupI, groupJ, k, localI, localJ);
        barrier(CLK_LOCAL_MEM_FENCE);

        calculate(localRows, localCols, localI, localJ, &value);
    }

    result[i * resultCols + j] = value;
}

static void loadToLocal(
        __global const float * restrict lhs,
        __global const float * restrict rhsT,
        __local VectorType localRows[GROUP_SIZE][GROUP_SIZE],
        __local VectorType localCols[GROUP_SIZE][GROUP_SIZE],
        size_t cols,
        size_t groupI,
        size_t groupJ,
        size_t k,
        size_t localI,
        size_t localJ) {
    localRows[localI][localJ] = LOAD_VECTOR(((groupI + localI) * cols + k) / VECTOR_SIZE + localJ, lhs);
    localCols[localI][localJ] = LOAD_VECTOR(((groupJ + localI) * cols + k) / VECTOR_SIZE + localJ, rhsT);
}

static void calculate(
        __local const VectorType localRows[GROUP_SIZE][GROUP_SIZE],
        __local const VectorType localCols[GROUP_SIZE][GROUP_SIZE],
        size_t localI,
        size_t localJ,
        float * restrict value) {
    for(size_t localK = 0; localK < GROUP_SIZE; ++localK) {
        *value += dot(localRows[localI][localK], localCols[localJ][localK]);
    }
}

