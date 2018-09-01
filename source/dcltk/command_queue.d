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
