import std.datetime.stopwatch : benchmark;
import std.math : approxEqual, sqrt, floor, ceil;
import std.random : uniform01;
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
    for(size_t i = 0; i < rows; ++i) {
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
        RESULT_COLS = 3000
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
	                            //value += localRow[localI * localCols + lk] * rhs[(k + lk) * resultCols + (j + groupJ)];
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
    `);
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "product");
    scope(exit) cl.releaseKernel(kernel);

    auto lhsBuffer = cl.createReadBuffer(context, lhs);
    scope(exit) cl.releaseBuffer(lhsBuffer);
    auto rhsBuffer = cl.createReadBuffer(context, rhs);
    scope(exit) cl.releaseBuffer(rhsBuffer);
    auto resultBuffer = cl.createWriteBuffer(context, gpuResult.length * float.sizeof);
    scope(exit) cl.releaseBuffer(resultBuffer);

    // set kernel arguments.
    cl.setKernelArg(kernel, 0, lhsBuffer);
    cl.setKernelArg(kernel, 1, rhsBuffer);
    cl.setKernelArg(kernel, 2, resultBuffer);
    cl.setKernelArg(kernel, 3, ROWS);
    cl.setKernelArg(kernel, 4, COLS);
    cl.setKernelArg(kernel, 5, RESULT_COLS);
    cl.allocateLocalMemory(kernel, 6, 1024 * float.sizeof);
    cl.allocateLocalMemory(kernel, 7, 1024 * float.sizeof);

    writefln("kernel w: %s, pw: %s",
        cl.getKernelWorkGroupSize(kernel, device),
        cl.getKernelPreferredWorkGroupSizeMultiple(kernel, device));

    void productGpu() {
        cl_event event;
        cl.enqueueKernel(commandQueue, kernel, [32 * 4, 32 * 4], [32, 32], event);
        cl.waitAndReleaseEvents(event);
    }

    // benchmark CPU and GPU.
    immutable gpuMsecs = benchmark!(() => productGpu())(4)[0].total!"msecs" / 4;
    immutable gpuFlops = (cast(real) ROWS) * RESULT_COLS * (COLS * 2.0) / ((cast(real) gpuMsecs) / 1000.0);
    
    version(DcltkWithCpuTest) {
        cl_event event;
        cl.enqueueReadBuffer(commandQueue, resultBuffer, 0, gpuResult, event);
        cl.waitAndReleaseEvents(event);

        immutable cpuMsecs = benchmark!(() => productCpu(
                    lhs, rhs, cpuResult, ROWS, COLS, RESULT_COLS))(1)[0].total!"msecs";
        writefln("cpu: %d msecs, gpu: %d msecs (%.1f GFLOPS, faster %.1f times)",
            cpuMsecs, gpuMsecs, gpuFlops / (10.0^^9), cast(real) cpuMsecs / cast(real) gpuMsecs);

        // writefln("%s", cpuResult);
        // writefln("%s", gpuResult);

        // check result values.
        foreach(i, e; cpuResult) {
            assert(approxEqual(e, gpuResult[i]));
        }
    } else {
        writefln("gpu: %d msecs (%.1f GFLOPS)", gpuMsecs, gpuFlops / (10.0^^9));
    }
}
