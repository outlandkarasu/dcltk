module dcltk.command_queue;

import derelict.opencl.cl;

import dcltk.error : enforceCl;
import dcltk.device :
    getDeviceMaxWorkItemSizes,
    getDeviceMaxComputeUnits;
import dcltk.kernel :
    getKernelPreferredWorkGroupSizeMultiple,
    getKernelWorkGroupSize;

import std.algorithm : min, max;
import std.exception : assumeUnique;
import std.math : sqrt, ceil, floor;

/**
 *  create in-order command queue.
 *
 *  Params:
 *      context = context.
 *      deviceId = device ID.
 *  Returns:
 *      new in-order command queue.
 */
cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId) {
    return createCommandQueue(context, deviceId, 0);
}

/**
 *  create out-of-order command queue.
 *
 *  Params:
 *      context = context.
 *      deviceId = device ID.
 *  Returns:
 *      new in-order command queue.
 */
cl_command_queue createOutOfOrderCommandQueue(
        cl_context context,
        cl_device_id deviceId) {
    return createCommandQueue(context, deviceId, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
}

/**
 *  create in-order command queue.
 *
 *  Params:
 *      context = context.
 *      deviceId = device ID.
 *      properties = command queue properties
 *  Returns:
 *      new in-order command queue.
 */
cl_command_queue createCommandQueue(
        cl_context context,
        cl_device_id deviceId,
        cl_command_queue_properties properties) {
    cl_int errorCode;
    auto commandQueue = clCreateCommandQueue(
            context, deviceId, properties, &errorCode);
    enforceCl(errorCode);
    return commandQueue;
}

/// release command queue.
void releaseCommandQueue(cl_command_queue commandQueue) {
    enforceCl(clReleaseCommandQueue(commandQueue));
}

/**
 *  get command queue information.
 *
 *  Params:
 *      commandQueue = command queue.
 *      paramName = information name.
 *  Returns:
 *      information value.
 */
T getCommandQueueInfo(T)(cl_command_queue commandQueue, cl_command_queue_info paramName) {
    T result;
    enforceCl(clGetCommandQueueInfo(commandQueue, paramName, T.sizeof, &result, null));
    return result;
}

/// get command queue device id.
cl_device_id getCommandQueueDeviceId(cl_command_queue commandQueue) {
    return getCommandQueueInfo!(cl_device_id)(commandQueue, CL_QUEUE_DEVICE);
}

/**
 *  enqueue kernel.
 *
 *  Params:
 *      commandQueue = command queue.
 *      globalWorkSizes = global work item count for dimensions.
 *      localWorkSizes = work group sizes for dimensions.
 *      event = event pointer.
 */
void enqueueKernel(
        cl_command_queue commandQueue,
        cl_kernel kernel,
        const(size_t)[] globalWorkSizes,
        const(size_t)[] localWorkSizes,
        cl_event* event)
in {
    assert(globalWorkSizes.length == localWorkSizes.length);
} body {
    enforceCl(clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                cast(cl_uint) globalWorkSizes.length,
                null,
                globalWorkSizes.ptr,
                localWorkSizes.ptr,
                0,
                null,
                event));
}

/**
 *  enqueue kernel.
 *
 *  Params:
 *      commandQueue = command queue.
 *      globalWorkSizes = global work item count for dimensions.
 *      localWorkSizes = work group sizes for dimensions.
 */
void enqueueKernel(
        cl_command_queue commandQueue,
        cl_kernel kernel,
        const(size_t)[] globalWorkSizes,
        const(size_t)[] localWorkSizes) {
    enqueueKernel(commandQueue, kernel, globalWorkSizes, localWorkSizes, null);
}

/**
 *  enqueue kernel.
 *
 *  Params:
 *      commandQueue = command queue.
 *      globalWorkSizes = global work item count for dimensions.
 *      localWorkSizes = work group sizes for dimensions.
 *      event = event reference.
 */
void enqueueKernel(
        cl_command_queue commandQueue,
        cl_kernel kernel,
        const(size_t)[] globalWorkSizes,
        const(size_t)[] localWorkSizes,
        out cl_event event) {
    enqueueKernel(commandQueue, kernel, globalWorkSizes, localWorkSizes, &event);
}

/// flush command queue.
void flushCommandQueue(cl_command_queue commandQueue) {
    enforceCl(clFlush(commandQueue));
}

/// finish command queue.
void finishCommandQueue(cl_command_queue commandQueue) {
    enforceCl(clFinish(commandQueue));
}

/// work item sizes
struct CommandQueueWorkSizes {
    size_t[] globalWorkSizes;
    size_t[] localWorkSizes;
}

/**
 *  calculate work sizes.
 *
 *  Params:
 *      commandQueue = command queue.
 *      kernel = kernel.
 */
immutable(CommandQueueWorkSizes) calculateWorkSizes(
        cl_command_queue commandQueue,
        cl_kernel kernel) {
    auto deviceId = getCommandQueueDeviceId(commandQueue);
    immutable preferredSize = getKernelPreferredWorkGroupSizeMultiple(kernel, deviceId);
    immutable workGroupSize = max(getKernelWorkGroupSize(kernel, deviceId) / preferredSize * preferredSize, 1);
    immutable maxSizes = getDeviceMaxWorkItemSizes(deviceId);
    if(maxSizes[1] <= 1) {
        return immutable(CommandQueueWorkSizes)([maxSizes[0], 1], [workGroupSize, 1]);
    }

    immutable workGroupSizeSqrt = max(cast(size_t) floor(sqrt(cast(real) workGroupSize)), 1);
    auto groups = cast(size_t) ceil(sqrt(cast(real) getDeviceMaxComputeUnits(deviceId)));
    immutable groups0 = min(groups, max(maxSizes[0] / workGroupSizeSqrt, 1));
    immutable groups1 = min(groups, max(maxSizes[1] / workGroupSizeSqrt, 1));
    return immutable(CommandQueueWorkSizes)(
        [workGroupSizeSqrt * groups0, workGroupSizeSqrt * groups1],
        [workGroupSizeSqrt, workGroupSizeSqrt]);
}

/**
 *  enqueue kernel.
 *
 *  Params:
 *      commandQueue = command queue.
 *      kernel = kernel.
 *      workSizes = work item sizes.
 */
void enqueueKernel(cl_command_queue commandQueue, cl_kernel kernel) {
    immutable workSizes = calculateWorkSizes(commandQueue, kernel);
    enqueueKernel(
        commandQueue,
        kernel,
        workSizes.globalWorkSizes,
        workSizes.localWorkSizes);
}

/**
 *  enqueue kernel.
 *
 *  Params:
 *      commandQueue = command queue.
 *      kernel = kernel.
 *      event = event.
 */
void enqueueKernel(
        cl_command_queue commandQueue,
        cl_kernel kernel,
        out cl_event event) {
    immutable workSizes = calculateWorkSizes(commandQueue, kernel);
    enqueueKernel(
        commandQueue,
        kernel,
        workSizes.globalWorkSizes,
        workSizes.localWorkSizes,
        event);
}

