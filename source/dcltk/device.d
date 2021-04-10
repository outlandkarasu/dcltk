module dcltk.device;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

import std.exception : assumeUnique;
import std.string : fromStringz;

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

/// get CPU devices.
cl_device_id[] getCpuDeviceIds(cl_platform_id platformId) {
    return getDeviceIds(platformId, CL_DEVICE_TYPE_CPU);
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
    enforceCl(clGetDeviceInfo(deviceId, name, 0, null, &size));

    auto result = new T[(size + T.sizeof - 1) / T.sizeof];
    enforceCl(clGetDeviceInfo(deviceId, name, size, result.ptr, &size));
    return result[0 .. size / T.sizeof];
}

/// get device global memory size.
cl_ulong getDeviceGlobalMemorySize(cl_device_id deviceId) {
    return getDeviceInfo!(cl_ulong)(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE);
}
/// get device local memory size.
cl_ulong getDeviceLocalMemorySize(cl_device_id deviceId) {
    return getDeviceInfo!(cl_ulong)(deviceId, CL_DEVICE_LOCAL_MEM_SIZE);
}

/// get device max compute units.
cl_uint getDeviceMaxComputeUnits(cl_device_id deviceId) {
    return getDeviceInfo!(cl_uint)(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS);
}

/// get device max work group size.
size_t getDeviceMaxWorkGroupSize(cl_device_id deviceId) {
    return getDeviceInfo!(size_t)(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

/// get device max work item dimensions.
cl_uint getDeviceMaxWorkItemDimensions(cl_device_id deviceId) {
    return getDeviceInfo!(cl_uint)(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
}

/// get device max work item sizes.
immutable(size_t)[] getDeviceMaxWorkItemSizes(cl_device_id deviceId) {
    return assumeUnique(getDeviceInfo!(size_t[])(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES));
}

/// get device name.
string getDeviceName(cl_device_id deviceId) {
    const name = getDeviceInfo!(char[])(deviceId, CL_DEVICE_NAME);
    return assumeUnique(fromStringz(name.ptr));
}

/// get device version.
string getDeviceVersion(cl_device_id deviceId) {
    const deviceVersion = getDeviceInfo!(char[])(deviceId, CL_DEVICE_OPENCL_C_VERSION);
    return assumeUnique(fromStringz(deviceVersion.ptr));
}

/// get device type.
cl_device_type getDeviceType(cl_device_id deviceId) {
    return getDeviceInfo!(cl_device_type)(deviceId, CL_DEVICE_TYPE);
}

/// is device GPU.
bool isGpuDevice(cl_device_id deviceId) {
    return getDeviceType(deviceId) == CL_DEVICE_TYPE_GPU;
}

