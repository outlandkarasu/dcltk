module dcltk.kernel;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

import std.string : toStringz;
import std.exception : assumeUnique;
import std.traits : isDynamicArray;

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
void setKernelArg(T)(cl_kernel kernel, cl_uint index, T value) if (!isDynamicArray!T) {
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
 *  allocate local memory for kernel parameter.
 *
 *  Params:
 *      kernel = kernel.
 *      index = argument index. (0 to n - 1)
 *      size = local memory size.
 */
void allocateLocalMemory(cl_kernel kernel, cl_uint index, size_t size) {
    enforceCl(clSetKernelArg(kernel, index, size, null));
}

/**
 *  get kernel work group info.
 *
 *  Params:
 *      kernel = kernel.
 *      device = device.
 *      name = parameter name.
 *  Returns:
 *      kernel work group info.
 */
T getKernelWorkGroupInfo(T)(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info name) {
    T result;
    enforceCl(clGetKernelWorkGroupInfo(kernel, device, name, result.sizeof, &result, null));
    return result;
}

/// get kernel work group size.
size_t getKernelWorkGroupSize(cl_kernel kernel, cl_device_id device) {
    return getKernelWorkGroupInfo!(size_t)(kernel, device, CL_KERNEL_WORK_GROUP_SIZE);
}

/// get kernel preferred work group size.
size_t getKernelPreferredWorkGroupSizeMultiple(cl_kernel kernel, cl_device_id device) {
    return getKernelWorkGroupInfo!(size_t)(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
}

