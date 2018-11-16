enum {
    BATCH_SIZE = 2,
    VECTOR_SIZE = 2,
};

typedef float2 VectorType;
#define LOAD_VECTOR vload2

__kernel
__attribute__((reqd_work_group_size(2, 2, 1)))
__attribute__((xcl_zero_global_work_offset))
void product(
        __global const float * restrict lhs,
        __global const float * restrict rhsT,
        __global float * restrict result,
        uint rows,
        uint cols,
        uint resultCols) {
    const size_t batchJ = get_global_id(0) * BATCH_SIZE;
    const size_t batchI = get_global_id(1) * BATCH_SIZE;

    float values[BATCH_SIZE][BATCH_SIZE];
    for(size_t i = 0; i < BATCH_SIZE; ++i) {
        for(size_t j = 0; j < BATCH_SIZE; ++j) {
            values[i][j] = 0.0f;
        }
    }

    for(size_t k = 0; k < cols; k += VECTOR_SIZE) {
        VectorType privateCols[BATCH_SIZE];
        for(size_t j = 0; j < BATCH_SIZE; ++j) {
            privateCols[j] = LOAD_VECTOR(((j + batchJ) * cols + k) / VECTOR_SIZE, rhsT);
        }
        for(size_t i = 0; i < BATCH_SIZE; ++i) {
            const VectorType privateRow = LOAD_VECTOR(((i + batchI) * cols + k) / VECTOR_SIZE, lhs);
            for(size_t j = 0; j < BATCH_SIZE; ++j) {
                values[i][j] += dot(privateRow, privateCols[j]);
            }
        }
    }

    for(size_t i = 0; i < BATCH_SIZE; ++i) {
        for(size_t j = 0; j < BATCH_SIZE; ++j) {
            result[(i + batchI) * resultCols + (j + batchJ)] = values[i][j];
        }
    }
}

