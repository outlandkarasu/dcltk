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

/// release context.
void releaseContext(cl_context context) {
    enforceCl(clReleaseContext(context));
}
