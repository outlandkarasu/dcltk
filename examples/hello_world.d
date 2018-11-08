import std.stdio : writefln;

import cl = dcltk;
import derelict.opencl.cl : cl_event;

void main() {
    auto platformId = cl.loadOpenCl();
    writefln("loaded: %s %s %s(%s) %s [%s]",
        cl.getPlatformProfile(platformId),
        cl.getPlatformName(platformId),
        cl.getPlatformVersion(platformId),
        cl.getPlatformCLVersion(platformId),
        cl.getPlatformVendor(platformId),
        cl.getPlatformExtensions(platformId));

    auto deviceIds = cl.getAllDeviceIds(platformId);
    writefln("devices:");
    foreach(i, d; deviceIds) {
        writefln("%d: %s %s gmem: %d, lmem: %d, cu: %d, w: %d, dim: %d %s",
                i,
                cl.getDeviceName(d),
                cl.getDeviceVersion(d),
                cl.getDeviceGlobalMemorySize(d),
                cl.getDeviceLocalMemorySize(d),
                cl.getDeviceMaxComputeUnits(d),
                cl.getDeviceMaxWorkGroupSize(d),
                cl.getDeviceMaxWorkItemDimensions(d),
                cl.getDeviceMaxWorkItemSizes(d));
    }
    auto context = cl.createDefaultContext(platformId);
    scope(exit) cl.releaseContext(context);

    auto device = deviceIds[0];
    auto commandQueue = cl.createCommandQueue(context, device);
    scope(exit) cl.releaseCommandQueue(commandQueue);

    auto program = cl.createProgramWithSource(context, `
        __kernel void helloWorld(void) {
            printf("Hello,World!\n");
        }
    `);
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "helloWorld");
    scope(exit) cl.releaseKernel(kernel);

    cl_event event;
    cl.enqueueKernel(commandQueue, kernel, [1], [1], event);
    cl.flushCommandQueue(commandQueue);

    cl.waitAndReleaseEvents(event);
    cl.finishCommandQueue(commandQueue);
}

