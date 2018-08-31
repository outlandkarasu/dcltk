module dcltk.device;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

/**
 *  get devices.
 *
 *  Params:
 *      platformId = platform ID.
 *      type = device type mask.
 *  Returns:
 *      device IDs.
 */
cl_device_id[] getDeviceIds(cl_platform_id platformId, cl_device_type type) {
    cl_uint count = 0;
    enforceCl(clGetDeviceIDs(platformId, type, 0, null, &count));

    auto result = new cl_device_id[count];
    enforceCl(clGetDeviceIDs(platformId, type, count, result.ptr, &count));
    return result[0 .. count];
}

/// get all devices.
cl_device_id[] getAllDeviceIds(cl_platform_id platformId) {
    return getDeviceIds(platformId, CL_DEVICE_TYPE_ALL);
}

/// get GPU devices.
cl_device_id[] getGpuDeviceIds(cl_platform_id platformId) {
    return getDeviceIds(platformId, CL_DEVICE_TYPE_GPU);
}

/**
 *  get device info.
 *
 *  Params:
 *      T = information type.
 *      deviceId = device ID.
 *      name = information name.
 *  Returns:
 *      device information.
 */
T getDeviceInfo(T)(cl_device_id deviceId, cl_device_info name) {
    T result;
    enforceCl(clGetDeviceInfo(deviceId, name, T.sizeof, &result, null));
    return result;
}

/**
 *  get device info.
 *
 *  Params:
 *      T = information type.
 *      deviceId = device ID.
 *      name = information name.
 *  Returns:
 *      device information.
 */
T[] getDeviceInfo(T : T[])(cl_device_id deviceId, cl_device_info name) {
    size_t size = 0;
    enforceCl(clGetDeviceInfo(deviceId, name, 0, null, size));

    auto result = new T[(size + T.sizeof - 1) / T.sizeof];
    enforceCl(clGetDeviceInfo(deviceId, name, size, &result, &size));
    return result[0 .. size / T.sizeof];
}

