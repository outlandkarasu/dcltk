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
 *  create host read only buffer object.
 *
 *  Params:
 *      context = context.
 *  Returns:
 *      host read only buffer object.
 */
cl_mem createHostReadOnlyBuffer(cl_context context, size_t size) {
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
 *  create host write only and device read only buffer object.
 *
 *  Params:
 *      context = context.
 *      data = buffer data.
 *  Returns:
 *      host write only and device read only buffer object.
 */
cl_mem createHostWriteOnlyBuffer(cl_context context, const(void)[] data) {
    return createBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            cast(void[]) data);
}

/**
 *  create host write only and device read only buffer object.
 *
 *  Params:
 *      context = context.
 *      size = buffer size.
 *  Returns:
 *      host write only and device read only buffer object.
 */
cl_mem createHostWriteOnlyBuffer(cl_context context, size_t size) {
    cl_int errorCode;
    auto buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            size,
            null,
            &errorCode);
    enforceCl(errorCode);
    return buffer;
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

/// Position struct.
struct Position {
    size_t x;
    size_t y;
    size_t z;
}

/// Region struct.
struct Region {
    size_t w;
    size_t h;
    size_t d = 1;
}

private immutable(size_t)[] ZERO_ORIGIN = [0, 0, 0];

/**
 *  enqueue read buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = read from position.
 *      bufferRegion = read region.
 *      bufferRowSize = buffer row size.
 *      dest = dest memory.
 *      event = event.
 */
private void enqueueReadBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        T[] dest,
        cl_event* event)
in {
    assert(bufferRegion.w * bufferRegion.h * bufferRegion.d == dest.length);
} body {
    enforceCl(clEnqueueReadBufferRect(
        queue,
        buffer,
        false,
        [bufferPos.x * T.sizeof, bufferPos.y, bufferPos.z].ptr,
        ZERO_ORIGIN.ptr,
        [bufferRegion.w * T.sizeof, bufferRegion.h, bufferRegion.d].ptr,
        bufferRowSize * T.sizeof,
        0,
        0,
        0,
        dest.ptr,
        0,
        null,
        event));
}

/**
 *  enqueue read buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = read from position.
 *      bufferRegion = read region.
 *      bufferRowSize = buffer row size.
 *      dest = dest memory.
 *      event = event.
 */
void enqueueReadBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        T[] dest,
        out cl_event event) {
    enqueueReadBuffer(queue, buffer, bufferPos, bufferRegion, bufferRowSize, dest, &event);
}

/**
 *  enqueue read buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = read from position.
 *      bufferRegion = read region.
 *      bufferRowSize = buffer row size.
 *      dest = dest memory.
 */
void enqueueReadBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        T[] dest) {
    enqueueReadBuffer(queue, buffer, bufferPos, bufferRegion, bufferRowSize, dest, null);
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

/**
 *  enqueue write buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = write to position.
 *      bufferRegion = write region.
 *      bufferRowPitch = buffer row pitch.
 *      source = source memory.
 *      event = event.
 */
private void enqueueWriteBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        const(T)[] source,
        cl_event* event)
in {
    assert(bufferRegion.w * bufferRegion.h * bufferRegion.d == source.length);
} body {
    enforceCl(clEnqueueWriteBufferRect(
        queue,
        buffer,
        false,
        [bufferPos.x * T.sizeof, bufferPos.y, bufferPos.z].ptr,
        ZERO_ORIGIN.ptr,
        [bufferRegion.w * T.sizeof, bufferRegion.h, bufferRegion.d].ptr,
        bufferRowSize * T.sizeof,
        0,
        0,
        0,
        source.ptr,
        0,
        null,
        event));
}

/**
 *  enqueue write buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = write to position.
 *      bufferRegion = write region.
 *      bufferRowSize = buffer row size.
 *      source = source memory.
 *      event = event.
 */
void enqueueWriteBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        const(T)[] source,
        out cl_event event) {
    enqueueWriteBuffer(queue, buffer, bufferPos, bufferRegion, bufferRowSize, source, &event);
}

/**
 *  enqueue write buffer rect.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer object.
 *      bufferPos = write to position.
 *      bufferRegion = write region.
 *      bufferRowSize = buffer row size.
 *      source = source memory.
 */
void enqueueWriteBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        auto ref const(Position) bufferPos,
        auto ref const(Region) bufferRegion,
        size_t bufferRowSize,
        const(T)[] source) {
    enqueueWriteBuffer(queue, buffer, bufferPos, bufferRegion, bufferRowSize, source, null);
}

/**
 *  enqueue buffer filling task.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer.
 *      pattern = fill pattern.
 *      offset = filling offset.
 *      count = fill count.
 *      event = event.
 */
private void enqueueFillBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        const(T)[] pattern,
        size_t offset,
        size_t count,
        cl_event* event) {
    immutable patternBytes = pattern.length * T.sizeof;
    enforceCl(clEnqueueFillBuffer(
        queue,
        buffer,
        pattern.ptr,
        patternBytes,
        offset * T.sizeof,
        count * patternBytes,
        0,
        null,
        event));
}

/**
 *  enqueue buffer filling task.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer.
 *      pattern = fill pattern.
 *      offset = filling offset.
 *      count = fill count.
 *      event = event.
 */
void enqueueFillBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        const(T)[] pattern,
        size_t offset,
        size_t count,
        out cl_event event) {
    enqueueFillBuffer(queue, buffer, pattern, offset, count, &event);
}

/**
 *  enqueue buffer filling task.
 *
 *  Params:
 *      queue = command queue.
 *      buffer = buffer.
 *      pattern = fill pattern.
 *      offset = filling offset.
 *      count = fill count.
 */
void enqueueFillBuffer(T)(
        cl_command_queue queue,
        cl_mem buffer,
        const(T)[] pattern,
        size_t offset,
        size_t count) {
    enqueueFillBuffer(queue, buffer, pattern, offset, count, null);
}

/**
 *  enqueue copy buffer task.
 *
 *  Params:
 *      queue = command queue.
 *      source = source buffer.
 *      dest = dest buffer.
 *      sourceOffset = source buffer offset.
 *      destOffset = dest buffer offset.
 *      size = copy size.
 *      event = wait event.
 */
void enqueueCopyBuffer(
        cl_command_queue queue,
        cl_mem source,
        cl_mem dest,
        size_t sourceOffset,
        size_t destOffset,
        size_t size,
        cl_event *event) {
    enforceCl(clEnqueueCopyBuffer(
                queue, source, dest, sourceOffset, destOffset, size, 0, null, event));
}

/**
 *  enqueue copy buffer task.
 *
 *  Params:
 *      queue = command queue.
 *      source = source buffer.
 *      dest = dest buffer.
 *      sourceOffset = source buffer offset.
 *      destOffset = dest buffer offset.
 *      size = copy size.
 *      event = wait event.
 */
void enqueueCopyBuffer(
        cl_command_queue queue,
        cl_mem source,
        cl_mem dest,
        size_t sourceOffset,
        size_t destOffset,
        size_t size,
        out cl_event event) {
    enqueueCopyBuffer(queue, source, dest, sourceOffset, destOffset, size, &event);
}

/**
 *  enqueue copy buffer task.
 *
 *  Params:
 *      queue = command queue.
 *      source = source buffer.
 *      dest = dest buffer.
 *      sourceOffset = source buffer offset.
 *      destOffset = dest buffer offset.
 */
void enqueueCopyBuffer(
        cl_command_queue queue,
        cl_mem source,
        cl_mem dest,
        size_t sourceOffset,
        size_t destOffset,
        size_t size) {
    enqueueCopyBuffer(queue, source, dest, sourceOffset, destOffset, size, null);
}

