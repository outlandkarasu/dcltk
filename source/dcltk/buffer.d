module dcltk.buffer;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

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
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            cast(void[]) data);
}

/// release buffer object.
void releaseBuffer(cl_mem buffer) {
    enforceCl(clReleaseMemObject(buffer));
}

/**
 *  enqueue read buffer.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      offset = read offset.
 *      dest = dest memory.
 *      event = enqueue event.
 */
void enqueueReadBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, void[] dest, out cl_event event) {
    enforceCl(clEnqueueReadBuffer(queue, buffer, false, offset, dest.length, dest.ptr, 0, null, &event));
}

/**
 *  enqueue read buffer.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      offset = read offset.
 *      dest = dest memory.
 */
void enqueueReadBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, void[] dest) {
    enforceCl(clEnqueueReadBuffer(queue, buffer, false, offset, dest.length, dest.ptr, 0, null, null));
}

/**
 *  enqueue write buffer.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      offset = write offset.
 *      source = source memory.
 *      event = enqueue event.
 */
void enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, const(void)[] source, out cl_event event) {
    enforceCl(clEnqueueWriteBuffer(queue, buffer, false, offset, source.length, source.ptr, 0, null, &event));
}

/**
 *  enqueue source buffer.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      offset = read offset.
 *      source = source memory.
 */
void enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, const(void)[] source) {
    enforceCl(clEnqueueWriteBuffer(queue, buffer, false, offset, source.length, source.ptr, 0, null, null));
}

