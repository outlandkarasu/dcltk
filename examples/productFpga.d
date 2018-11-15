import std.datetime.stopwatch : benchmark;
import std.file : read;
import std.math : approxEqual, sqrt, floor, ceil;
import std.parallelism : parallel;
import std.random : uniform01;
import std.range : iota;
import std.stdio : writefln;
import std.string : format;
import std.traits : isIntegral;

import cl = dcltk;
import derelict.opencl.cl : cl_event, CL_DEVICE_TYPE_ACCELERATOR;

/// round up for unit.
private T roundUp(T)(T value, T unit) @safe pure nothrow @nogc if(isIntegral!T) {
    return (value + unit - 1) / unit * unit;
}

private T[] transpose(T)(const(T)[] values, size_t rows, size_t cols) {
    auto result = new T[values.length];
    foreach(i; 0 .. rows) {
        foreach(j; 0 .. cols) {
            result[j * rows + i] = values[i * cols + j];
        }
    }
    return result;
}

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
        ROWS = 256,
        COLS = 256,
        RESULT_COLS = 256,
        BATCH_SIZE = 16,
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

    auto context = cl.createContextFromType(platformId, CL_DEVICE_TYPE_ACCELERATOR);
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

    auto program = cl.createProgramWithBinary(
        context, device, cast(ubyte[]) read("./examples/productFpga.xclbin"));
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "product");
    scope(exit) cl.releaseKernel(kernel);

    // calculate padded matrix size.
    immutable bufferCols = cast(uint) roundUp(COLS, BATCH_SIZE);
    assert(bufferCols == COLS);
    immutable bufferRows = cast(uint) roundUp(ROWS, BATCH_SIZE);
    assert(bufferRows == ROWS);
    immutable bufferResultCols = cast(uint) roundUp(RESULT_COLS, BATCH_SIZE);
    assert(bufferResultCols == RESULT_COLS);
    writefln("bc: %s, br: %s, brc: %s", bufferCols, bufferRows, bufferResultCols);

    immutable lhsSize = bufferCols * bufferRows;
    immutable rhsSize = bufferResultCols * bufferCols;
    immutable resultSize = bufferResultCols * bufferRows;

    // create buffers.
    auto lhsBuffer = cl.createHostWriteOnlyBuffer(context, lhs);
    scope(exit) cl.releaseBuffer(lhsBuffer);
    auto rhsT = transpose(rhs, bufferCols, bufferResultCols);
    auto rhsBuffer = cl.createHostWriteOnlyBuffer(context, rhsT);
    scope(exit) cl.releaseBuffer(rhsBuffer);
    auto resultBuffer = cl.createHostReadOnlyBuffer(context, resultSize * float.sizeof);
    scope(exit) cl.releaseBuffer(resultBuffer);

    // set kernel arguments.
    cl.setKernelArg(kernel, 0, lhsBuffer);
    cl.setKernelArg(kernel, 1, rhsBuffer);
    cl.setKernelArg(kernel, 2, resultBuffer);
    cl.setKernelArg(kernel, 3, bufferRows);
    cl.setKernelArg(kernel, 4, bufferCols);
    cl.setKernelArg(kernel, 5, bufferResultCols);

    immutable(size_t)[] globalWorkSizes = [1, 1];
    immutable(size_t)[] localWorkSizes = [1, 1];
    writefln("workSizes: %s, %s", localWorkSizes, globalWorkSizes);

    void productGpu() {
        cl_event event;
        cl.enqueueKernel(commandQueue, kernel, globalWorkSizes, localWorkSizes, event);
        cl.waitAndReleaseEvents(event);
    }

    // benchmark CPU and GPU.
    cl.finishCommandQueue(commandQueue);
    enum REPEAT = 1;
    immutable gpuMsecs = benchmark!(() => productGpu())(REPEAT)[0].total!"msecs" / REPEAT;
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
            assert(approxEqual(e, gpuResult[i]), "[%d] %s != %s".format(i, e, gpuResult[i]));
        }
    } else {
        writefln("gpu: %d msecs (%.1f GFLOPS)", gpuMsecs, gpuFlops / (10.0^^9));
    }
}
