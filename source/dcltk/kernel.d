module dcltk.kernel;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

import std.string : toStringz;

/**
 *  create kernel.
 *
 *  Params:
 *      program = program.
 *      name = kernel name..
 *  Returns:
 *      kernel.
 */
cl_kernel createKernel(cl_program program, string name) {
    cl_int errorCode;
    auto result = clCreateKernel(program, toStringz(name), &errorCode);
    enforceCl(errorCode);
    return result;
}

/// release kernel.
void releaseKernel(cl_kernel kernel) {
    enforceCl(clReleaseKernel(kernel));
}

/**
 *  set a kernel argument.
 *
 *  Params:
 *      kernel = kernel.
 *      index = argument index. (0 to n - 1)
 *      value = value reference.
 */
void setKernelArg(T)(cl_kernel kernel, cl_uint index, T value) {
    enforceCl(clSetKernelArg(kernel, index, T.sizeof, &value));
}

/**
 *  set a kernel argument.
 *
 *  Params:
 *      kernel = kernel.
 *      index = argument index. (0 to n - 1)
 *      value = value array.
 */
void setKernelArg(T : const(T)[])(cl_kernel kernel, cl_uint index, const(T)[] value) {
    enforceCl(clSetKernelArg(kernel, index, T.sizeof * value.length, value.ptr));
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

