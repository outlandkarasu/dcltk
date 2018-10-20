import std.algorithm : min, max;
import std.datetime.stopwatch : benchmark;
import std.math : approxEqual, sqrt, floor, ceil;
import std.parallelism : parallel;
import std.random : uniform01;
import std.range : iota;
import std.string : format;
import std.stdio : writefln;

import cl = dcltk;
import derelict.opencl.cl : cl_event, cl_command_queue, cl_kernel;

private T roundUp(T)(T value, T unit) {
    return (value % unit) == 0 ? value : value + (unit - (value % unit));
}
private T roundDown(T)(T value, T unit) {
    return value - (value % unit);
}

/// matrix product by CPU.
private void productCpu(
        const(float)[] lhs,
        const(float)[] rhs,
        float[] result,
        uint rows,
        uint cols,
        uint resultCols)
in {
    assert(lhs.length == rows * cols);
    assert(rhs.length == cols * resultCols);
    assert(result.length == rows * resultCols);
} body {
    foreach(i; parallel(iota(0, rows))) {
        for(size_t j = 0; j < resultCols; ++j) {
            float value = 0.0f;
            for(size_t k = 0; k < cols; ++k) {
                value += lhs[i * cols + k] * rhs[k * resultCols + j];
            }
            result[i * resultCols + j] = value;
        }
    }
}

///
unittest {
    immutable(float)[] lhs = [1, 2, 3, 4];
    immutable(float)[] rhs = [5, 6, 7, 8];
    auto result = new float[2 * 2];

    productCpu(lhs, rhs, result, 2, 2, 2);

    assert(approxEqual(1 * 5 + 2 * 7, result[0 * 2 + 0]));
    assert(approxEqual(1 * 6 + 2 * 8, result[0 * 2 + 1]));
    assert(approxEqual(3 * 5 + 4 * 7, result[1 * 2 + 0]));
    assert(approxEqual(3 * 6 + 4 * 8, result[1 * 2 + 1]));

    auto resultScalar = new float[1];
    productCpu(lhs, rhs, resultScalar, 1, 4, 1);
    assert(approxEqual(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8, resultScalar[0]));
}

void main() {
    // matrix size.
    enum {
        ROWS = 2048,
        COLS = 2048,
        RESULT_COLS = 2048,
        BATCH_SIZE = 128,
        PRIVATE_SIZE = 8,
        WORK_GROUP_SIZE = BATCH_SIZE / PRIVATE_SIZE,
        BATCH_SIZE_K = 16
    }

    // initialize operand matrixes.
    auto lhs = new float[ROWS * COLS];
    foreach(ref e; lhs) {
        e = uniform01!float;
    }
    auto rhs = new float[COLS * RESULT_COLS];
    foreach(ref e; rhs) {
        e = uniform01!float;
    }
    auto cpuResult = new float[ROWS * RESULT_COLS];
    auto gpuResult = new float[ROWS * RESULT_COLS];

    auto platformId = cl.loadOpenCl();
    writefln("loaded: %s %s %s(%s) %s [%s]",
        cl.getPlatformProfile(platformId),
        cl.getPlatformName(platformId),
        cl.getPlatformVersion(platformId),
        cl.getPlatformCLVersion(platformId),
        cl.getPlatformVendor(platformId),
        cl.getPlatformExtensions(platformId));

    auto context = cl.createDefaultContext(platformId);
    scope(exit) cl.releaseContext(context);
    auto deviceIds = cl.getContextDeviceIds(context);
    if(deviceIds.length <= 0) {
        throw new cl.OpenClException("device not found.");
    }

    auto device = deviceIds[0];
    writefln("device: %s %s gmem: %d, lmem: %d, cu: %d, w: %d, dim: %d %s",
        cl.getDeviceName(device),
        cl.getDeviceVersion(device),
        cl.getDeviceGlobalMemorySize(device),
        cl.getDeviceLocalMemorySize(device),
        cl.getDeviceMaxComputeUnits(device),
        cl.getDeviceMaxWorkGroupSize(device),
        cl.getDeviceMaxWorkItemDimensions(device),
        cl.getDeviceMaxWorkItemSizes(device));
    if(!cl.isGpuDevice(device)) {
        writefln("WARNING: device is not GPU!");
    }

    auto commandQueue = cl.createCommandQueue(context, device);
    scope(exit) cl.releaseCommandQueue(commandQueue);

    auto program = cl.createProgramFromSource(context, `
        __kernel void product(
                __global const float *lhs,
                __global const float *rhs,
                __global float *result,
                int rows,
                int cols,
                int resultCols) {
            enum {
                WORK_GROUP_SIZE = %d,
                BATCH_SIZE = %d,
                PRIVATE_SIZE = %d,
                BATCH_SIZE_K = %d,
                PRIVATE_ROWS = PRIVATE_SIZE,
                PRIVATE_COLS = PRIVATE_SIZE,
                LOCAL_ROWS = BATCH_SIZE,
                LOCAL_COLS = BATCH_SIZE,
                LOCAL_SIZE = WORK_GROUP_SIZE * WORK_GROUP_SIZE,
                LOCAL_COPY_COUNT = LOCAL_ROWS * BATCH_SIZE_K / LOCAL_SIZE
            };

            __local float localLhs[BATCH_SIZE][BATCH_SIZE_K];
            __local float localRhs[BATCH_SIZE][BATCH_SIZE_K + 2];
            float value[PRIVATE_ROWS][PRIVATE_COLS];

            const int localI = get_local_id(1);
            const int localJ = get_local_id(0);

            const int localId = localI * WORK_GROUP_SIZE + localJ;
            const int groupI = get_group_id(1) * LOCAL_ROWS;
            const int groupJ = get_group_id(0) * LOCAL_COLS;

            // initialize private memory.
            for(int pi = 0; pi < PRIVATE_ROWS; ++pi) {
                for(int pj = 0; pj < PRIVATE_COLS; ++pj) {
                    value[pi][pj] = 0.0f;
                }
            }

            for(int k = 0; k < cols; k += BATCH_SIZE_K) {
                barrier(CLK_LOCAL_MEM_FENCE);
                for(int offset = 0; offset < LOCAL_COPY_COUNT; ++offset) {
                    const int id = (offset * LOCAL_SIZE) + localId;
                    const int copyLhsI = id / BATCH_SIZE_K;
                    const int copyLhsJ = id %% BATCH_SIZE_K;
                    const int copyRhsI = id / LOCAL_ROWS;
                    const int copyRhsJ = id %% LOCAL_ROWS;
                    localLhs[copyLhsI][copyLhsJ] = lhs[(groupI + copyLhsI) * cols + (k + copyLhsJ)];
                    localRhs[copyRhsJ][copyRhsI] = rhs[(k + copyRhsI) * resultCols + (groupJ + copyRhsJ)];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for(int lk = 0; lk < BATCH_SIZE_K; ++lk) {
                    float privateCols[PRIVATE_COLS];
                    for(int pj = 0; pj < PRIVATE_COLS; ++pj) {
                        privateCols[pj] = localRhs[pj * WORK_GROUP_SIZE + localJ][lk];
                    }
                    for(int pi = 0; pi < PRIVATE_ROWS; ++pi) {
                        const float privateRow = localLhs[pi * WORK_GROUP_SIZE + localI][lk];
                        for(int pj = 0; pj < PRIVATE_COLS; ++pj) {
                            value[pi][pj] = mad(privateRow, privateCols[pj], value[pi][pj]);
                        }
                    }
                }
            }
            for(int pi = 0; pi < PRIVATE_ROWS; ++pi) {
                const int globalRowOffset = (groupI + (pi * WORK_GROUP_SIZE) + localI) * resultCols;
                for(int pj = 0; pj < PRIVATE_COLS; ++pj) {
                    result[globalRowOffset + (groupJ + (pj * WORK_GROUP_SIZE) + localJ)] = value[pi][pj];
                }
            }
        }
    `.format(WORK_GROUP_SIZE, BATCH_SIZE, PRIVATE_SIZE, BATCH_SIZE_K));
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "product");
    scope(exit) cl.releaseKernel(kernel);

    immutable(size_t)[] globalWorkSizes = [
        roundUp(RESULT_COLS, BATCH_SIZE) / PRIVATE_SIZE,
        roundUp(ROWS, BATCH_SIZE) / PRIVATE_SIZE
    ];
    immutable(size_t)[] localWorkSizes = [WORK_GROUP_SIZE, WORK_GROUP_SIZE];
    writefln("workSizes: %s, %s", localWorkSizes, globalWorkSizes);

    // calculate padded matrix size.
    immutable bufferCols = cast(int) roundUp(COLS, BATCH_SIZE);
    immutable bufferRows = cast(int) roundUp(ROWS, BATCH_SIZE);
    immutable bufferResultCols = cast(int) roundUp(RESULT_COLS, BATCH_SIZE);
    writefln("bc: %s, br: %s, brc: %s", bufferCols, bufferRows, bufferResultCols);

    immutable lhsSize = bufferCols * bufferRows;
    immutable rhsSize = bufferResultCols * bufferCols;
    immutable resultSize = bufferResultCols * bufferRows;

    // create buffers.
    auto lhsBuffer = cl.createReadBuffer(context, lhsSize * float.sizeof);
    scope(exit) cl.releaseBuffer(lhsBuffer);
    auto rhsBuffer = cl.createReadBuffer(context, rhsSize * float.sizeof);
    scope(exit) cl.releaseBuffer(rhsBuffer);
    auto resultBuffer = cl.createWriteBuffer(context, resultSize * float.sizeof);
    scope(exit) cl.releaseBuffer(resultBuffer);

    // copy parameter matrixes.
    cl.enqueueFillBuffer(
        commandQueue, lhsBuffer, [0.0f], 0, lhsSize);
    cl.enqueueWriteBuffer(
        commandQueue,
        lhsBuffer,
        cl.Position(0, 0),
        cl.Region(COLS, ROWS),
        bufferCols,
        lhs);
    cl.enqueueFillBuffer(
        commandQueue, rhsBuffer, [0.0f], 0, rhsSize);
    cl.enqueueWriteBuffer(
        commandQueue,
        rhsBuffer,
        cl.Position(0, 0),
        cl.Region(RESULT_COLS, COLS),
        bufferResultCols,
        rhs);

    // set kernel arguments.
    cl.setKernelArg(kernel, 0, lhsBuffer);
    cl.setKernelArg(kernel, 1, rhsBuffer);
    cl.setKernelArg(kernel, 2, resultBuffer);
    cl.setKernelArg(kernel, 3, bufferRows);
    cl.setKernelArg(kernel, 4, bufferCols);
    cl.setKernelArg(kernel, 5, bufferResultCols);

    void productGpu() {
        cl_event event;
        cl.enqueueKernel(
            commandQueue,
            kernel,
            globalWorkSizes,
            localWorkSizes,
            event);
        cl.waitAndReleaseEvents(event);
    }

    // benchmark CPU and GPU.
    version(DcltkWithCpuTest) {
        immutable cpuMsecs = benchmark!(() => productCpu(
            lhs, rhs, cpuResult, ROWS, COLS, RESULT_COLS))(1)[0].total!"msecs";
    } else {
        immutable cpuMsecs = 0;
    }

    immutable gpuMsecs = benchmark!(() => productGpu())(4)[0].total!"msecs" / 4;
    writefln("cpu: %d msecs, gpu: %d msecs", cpuMsecs, gpuMsecs);

    // writefln("%s", cpuResult);
    // writefln("%s", gpuResult);

    // check result values.
    version(DcltkWithCpuTest) {
        cl_event event;
        cl.enqueueReadBuffer(
            commandQueue,
            resultBuffer,
            cl.Position(0, 0),
            cl.Region(RESULT_COLS, ROWS),
            bufferResultCols,
            gpuResult,
            event);
        cl.waitAndReleaseEvents(event);
        foreach(i, e; cpuResult) {
            assert(approxEqual(e, gpuResult[i]));
        }
    }
}
