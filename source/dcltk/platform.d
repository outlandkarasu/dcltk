module dcltk.platform;

import std.algorithm : splitter;
import std.array : appender;
import std.exception : assumeUnique;
import std.range : array;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

/**
 *  get available platform ids.
 *
 *  Returns:
 *      available platform ids.
 */
cl_platform_id[] getPlatformIds() {
    cl_uint count = 0;
    enforceCl(clGetPlatformIDs(0, null, &count));

    auto result = new cl_platform_id[count];
    enforceCl(clGetPlatformIDs(count, result.ptr, &count));
    return result[0 .. count];
}

/**
 *  get platform information.
 *
 *  Params:
 *      platformId = platform ID.
 *      name = information name.
 *  Returns:
 *      platform inforamtion.
 */
string getPlatformInfo(cl_platform_id platformId, cl_platform_info name) {
    size_t size = 0;
    enforceCl(clGetPlatformInfo(platformId, name, 0, null, &size));

    auto result = new char[size];
    enforceCl(clGetPlatformInfo(platformId, name, size, result.ptr, &size));
    return assumeUnique(result[0 .. size]);
}

/// get platform profile
string getPlatformProfile(cl_platform_id platformId) {
    return getPlatformInfo(platformId, CL_PLATFORM_PROFILE);
}

/// get platform version
string getPlatformVersion(cl_platform_id platformId) {
    return getPlatformInfo(platformId, CL_PLATFORM_VERSION);
}

/// get platform name
string getPlatformName(cl_platform_id platformId) {
    return getPlatformInfo(platformId, CL_PLATFORM_NAME);
}

/// get platform vendor
string getPlatformVendor(cl_platform_id platformId) {
    return getPlatformInfo(platformId, CL_PLATFORM_VENDOR);
}

/// get platform extensions
string getPlatformExtensions(cl_platform_id platformId) {
    return getPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS);
}

/// get platform version value.
CLVersion getPlatformCLVersion(cl_platform_id platformId) {
    auto tokens = getPlatformVersion(platformId).splitter(" ").array;
    if(tokens.length < 2) {
        return CLVersion.None;
    }
    switch(tokens[1]) {
    case "1.0": return CLVersion.CL10;
    case "1.1": return CLVersion.CL11;
    case "1.2": return CLVersion.CL12;
    case "2.0": return CLVersion.CL20;
    case "2.1": return CLVersion.CL21;
    case "2.2": return CLVersion.CL22;
    default:
        return CLVersion.None;
    }
}

