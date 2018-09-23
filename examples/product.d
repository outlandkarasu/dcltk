import std.datetime.stopwatch : benchmark;
import std.math : approxEqual, sqrt, floor, ceil;
import std.parallelism : parallel;
import std.random : uniform01;
import std.range : iota;
import std.string : format;
import std.stdio : writefln;

import cl = dcltk;
import derelict.opencl.cl : cl_event;

/// matrix product by CPU.
void productCpu(
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
        ROWS = 1000,
        COLS = 2000,
        RESULT_COLS = 3000,
        PRIVATE_ROWS = 2,
        PRIVATE_COLS = 2,
        WORK_GROUP_SIZE = 32
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
                uint rows,
                uint cols,
                uint resultCols,
                __local float *localRow,
                __local float *localCol) {
            enum {
                PRIVATE_ROWS = %d,
                PRIVATE_COLS = %d,
                WORK_GROUP_SIZE = %d,
            };

            const size_t groupI = get_global_id(0) * PRIVATE_ROWS;
            const size_t groupRows = get_global_size(0) * PRIVATE_ROWS;
            const size_t groupJ = get_global_id(1) * PRIVATE_COLS;
            const size_t groupCols = get_global_size(1) * PRIVATE_COLS;

            const size_t localI = get_local_id(0) * PRIVATE_ROWS;
            const size_t localRows = get_local_size(0) * PRIVATE_ROWS;
            const size_t localJ = get_local_id(1) * PRIVATE_COLS;
            const size_t localCols = get_local_size(1) * PRIVATE_COLS;
            const size_t localCols4 = localCols / 4; // for vectorization

            for(size_t i = 0; i < rows; i += groupRows) {
                const size_t globalRow = i + groupI;
                for(size_t j = 0; j < resultCols; j += groupCols) {
                    const size_t globalCol = j + groupJ;
                    float value[PRIVATE_ROWS][PRIVATE_COLS];

                    // initialize private memory.
                    for(size_t pi = 0; pi < PRIVATE_ROWS; ++pi) {
                        for(size_t pj = 0; pj < PRIVATE_COLS; ++pj) {
                            value[pi][pj] = 0.0f;
                        }
                    }

                    for(size_t k = 0; k < cols; k += localCols) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        for(size_t pi = 0; pi < PRIVATE_ROWS; ++pi) {
                            for(size_t pj = 0; pj < PRIVATE_COLS; ++pj) {
                                localRow[(localI + pi) * localCols + (localJ + pj)] = lhs[(globalRow + pi) * cols + (k + localJ + pj)];
                                localCol[(localJ + pj) * localRows + (localI + pi)] = rhs[(k + localI + pi) * resultCols + (globalCol + pj)];
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);

                        for(size_t pi = 0; pi < PRIVATE_ROWS; ++pi) {
                            float privateRow[WORK_GROUP_SIZE * PRIVATE_COLS];
                            for(size_t lk = 0; lk < localCols; ++lk) {
                                privateRow[lk] = localRow[(localI + pi) * localCols + lk];
                            }
                            for(size_t pj = 0; pj < PRIVATE_COLS; ++pj) {
                                float privateCol[WORK_GROUP_SIZE * PRIVATE_COLS];
	                            for(size_t lk = 0; lk < localCols; ++lk) {
	                                privateCol[lk] = localCol[(localJ + pj) * localRows + lk];
	                            }
	                            for(size_t lk = 0; lk < localCols4; ++lk) {
                                    const float4 r = vload4(lk, privateRow);
                                    const float4 c = vload4(lk, privateCol);
                                    value[pi][pj] += dot(r, c);
	                            }
	                            for(size_t lk = (localCols4 * 4); lk < localCols; ++lk) {
                                    value[pi][pj] = mad(privateRow[lk], privateCol[lk], value[pi][pj]);
	                            }
                            }
                        }
                    }
                    for(size_t pi = 0; pi < PRIVATE_ROWS; ++pi) {
                        const size_t globalRowOffset = (globalRow + pi) * resultCols + globalCol;
                        for(size_t pj = 0; pj < PRIVATE_COLS; ++pj) {
                            result[globalRowOffset + pj] = value[pi][pj];
                        }
                    }
                }
            }
        }
    `.format(PRIVATE_ROWS, PRIVATE_COLS, WORK_GROUP_SIZE));
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "product");
    scope(exit) cl.releaseKernel(kernel);

    const workSizes = cl.calculateWorkSizes(commandQueue, kernel);
    writefln("workSizes: %s", workSizes);

    // calculate padded matrix size.
    immutable groupSize =
        (workSizes.localWorkSizes[0] * PRIVATE_COLS)
        * (workSizes.localWorkSizes[1] * PRIVATE_ROWS);
    immutable workWidth = workSizes.globalWorkSizes[1] * PRIVATE_COLS;
    immutable workHeight = workSizes.globalWorkSizes[0] * PRIVATE_ROWS;
    immutable bufferCols = cast(uint)((COLS + workWidth - 1) / workWidth * workWidth);
    immutable bufferRows = cast(uint)((ROWS + workHeight - 1) / workHeight * workHeight);
    immutable bufferResultCols = cast(uint)((RESULT_COLS + workWidth - 1) / workWidth * workWidth);
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
    cl.allocateLocalMemory(kernel, 6, groupSize * float.sizeof);
    cl.allocateLocalMemory(kernel, 7, groupSize * float.sizeof);

    void productGpu() {
        cl_event event;
        cl.enqueueKernel(
            commandQueue,
            kernel,
            workSizes.globalWorkSizes,
            workSizes.localWorkSizes);
        cl.enqueueReadBuffer(
            commandQueue,
            resultBuffer,
            cl.Position(0, 0),
            cl.Region(RESULT_COLS, ROWS),
            bufferResultCols,
            gpuResult,
            event);
        cl.flushCommandQueue(commandQueue);
        cl.waitAndReleaseEvents(event);
        cl.finishCommandQueue(commandQueue);
    }

    // benchmark CPU and GPU.
    immutable cpuMsecs = benchmark!(() => productCpu(
                lhs, rhs, cpuResult, ROWS, COLS, RESULT_COLS))(1)[0].total!"msecs";
    immutable gpuMsecs = benchmark!(() => productGpu())(1)[0].total!"msecs";
    writefln("cpu: %d msecs, gpu: %d msecs", cpuMsecs, gpuMsecs);

    // writefln("%s", cpuResult);
    // writefln("%s", gpuResult);

    // check result values.
    foreach(i, e; cpuResult) {
        assert(approxEqual(e, gpuResult[i]));
    }
}

