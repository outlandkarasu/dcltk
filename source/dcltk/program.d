module dcltk.program;

import derelict.opencl.cl;

import dcltk.error : enforceCl, OpenClException;

import std.algorithm : map, joiner;
import std.conv : to;
import std.exception : assumeUnique;
import std.string : toStringz;

/**
 *  create program from source.
 *
 *  Params:
 *      context = context.
 *      source = program source.
 *  Returns:
 *      program.
 */
cl_program createProgramFromSource(cl_context context, string source) {
    cl_int errorCode;
    size_t length = source.length;
    const(char)* sourcePointer = toStringz(source);
    auto program = clCreateProgramWithSource(
            context, 1, &sourcePointer, &length, &errorCode);
    enforceCl(errorCode);
    return program;
}

/// release program.
void releaseProgram(cl_program program) {
    enforceCl(clReleaseProgram(program));
}

/**
 *  build program.
 *
 *  Params:
 *      program = program.
 *      deviceIds = build target device IDs.
 *      options = compile options.
 */
void buildProgram(cl_program program, cl_device_id[] deviceIds, string options = "") {
    auto errorCode = clBuildProgram(
                program,
                cast(cl_uint) deviceIds.length,
                deviceIds.ptr,
                toStringz(options),
                null,
                null);
    if(errorCode == CL_BUILD_PROGRAM_FAILURE) {
        auto logs = deviceIds
            .map!(d => getProgramBuildLog(program, d))
            .joiner("\n")
            .to!string;
        throw new OpenClException(logs);
    }
    enforceCl(errorCode);
}

/**
 *  get program build log.
 *
 *  Params:
 *      program = program.
 *      device = device ID.
 */
string getProgramBuildLog(cl_program program, cl_device_id deviceId) {
    size_t size;
    enforceCl(clGetProgramBuildInfo(
                program, deviceId, CL_PROGRAM_BUILD_LOG, 0, null, &size));

    auto result = new char[size];
    enforceCl(clGetProgramBuildInfo(
                program, deviceId, CL_PROGRAM_BUILD_LOG, size, result.ptr, &size));
    return assumeUnique(result[0 .. size]);
}

