module dcltk.context;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

/**
 *  create context.
 *
 *  Params:
 *      platformId = platform ID.
 *      devices = context devices.
 *  Returns:
 *      new context.
 */
cl_context createContext(cl_platform_id platformId, cl_device_id[] devices...) {
    immutable(cl_context_properties)[] properties = [
        CL_CONTEXT_PLATFORM, cast(cl_context_properties) platformId,
        0
    ];
    cl_int errorCode;
    auto context = clCreateContext(
        properties.ptr,
        cast(cl_uint) devices.length,
        devices.ptr,
        null,
        null,
        &errorCode);
    enforceCl(errorCode);
    return context;
}

/**
 *  create context from type.
 *
 *  Params:
 *      platformId = platform ID.
 *      type = device type.
 *  Returns:
 *      new context.
 */
cl_context createContextFromType(cl_platform_id platformId, cl_device_type type) {
    immutable(cl_context_properties)[] properties = [
        CL_CONTEXT_PLATFORM, cast(cl_context_properties) platformId,
        0
    ];
    cl_int errorCode;
    auto context = clCreateContextFromType(
        properties.ptr,
        type,
        null,
        null,
        &errorCode);
    enforceCl(errorCode);
    return context;
}

/**
 *  create GPU context.
 *
 *  Params:
 *      platformId = platform ID.
 *  Returns:
 *      new GPU context.
 */
cl_context createGpuContext(cl_platform_id platformId) {
    return createContextFromType(platformId, CL_DEVICE_TYPE_GPU);
}

/**
 *  create default device type context.
 *
 *  Params:
 *      platformId = platform ID.
 *  Returns:
 *      new default device type context.
 */
cl_context createDefaultContext(cl_platform_id platformId) {
    return createContextFromType(platformId, CL_DEVICE_TYPE_DEFAULT);
}

/// release context.
void releaseContext(cl_context context) {
    enforceCl(clReleaseContext(context));
}

