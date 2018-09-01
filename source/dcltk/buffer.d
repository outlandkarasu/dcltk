module dcltk.buffer;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

import std.stdio : writefln;

/**
 *  create buffer object.
 *
 *  Params:
 *      context = context.
 *      flags = memory flags.
 *      data = buffer data.
 *  Returns:
 *      buffer object.
 */
cl_mem createBuffer(cl_context context, cl_mem_flags flags, void[] data) {
    cl_int errorCode;
    auto buffer = clCreateBuffer(
            context, flags, data.length, data.ptr, &errorCode);
    enforceCl(errorCode);
    return buffer;
}

/**
 *  create read write buffer object.
 *
 *  Params:
 *      context = context.
 *      data = buffer data.
 *  Returns:
 *      buffer object.
 */
cl_mem createBuffer(cl_context context, const(void)[] data) {
    return createBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cast(void[]) data);
}

/**
 *  create write buffer object.
 *
 *  Params:
 *      context = context.
 *  Returns:
 *      write buffer object.
 */
cl_mem createWriteBuffer(cl_context context, size_t size) {
    cl_int errorCode;
    auto buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
            size,
            null,
            &errorCode);
    enforceCl(errorCode);
    return buffer;
}

/**
 *  create read buffer object.
 *
 *  Params:
 *      context = context.
 *      data = buffer data.
 *  Returns:
 *      write buffer object.
 */
cl_mem createReadBuffer(cl_context context, const(void)[] data) {
    return createBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
            cast(void[]) data);
}

/// release buffer object.
void releaseBuffer(cl_mem buffer) {
    enforceCl(clReleaseMemObject(buffer));
}
