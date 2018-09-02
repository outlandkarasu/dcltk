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

