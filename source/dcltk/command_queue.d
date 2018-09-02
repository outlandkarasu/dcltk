module dcltk.command_queue;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

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

