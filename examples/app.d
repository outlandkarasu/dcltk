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

    auto data = [100.0f];
    auto buffer = cl.createBuffer(context, data);
    scope(exit) cl.releaseBuffer(buffer);
    auto writeBuffer = cl.createWriteBuffer(context, 100);
    scope(exit) cl.releaseBuffer(writeBuffer);
    auto readBuffer = cl.createReadBuffer(context, data);
    scope(exit) cl.releaseBuffer(readBuffer);

    cl.enqueueWriteBuffer(commandQueue, buffer, 0, [200.0f]);

    cl_event event;
    cl.enqueueReadBuffer(commandQueue, buffer, 0, data, event);
    cl.waitAndReleaseEvents(event);
    assert(data[0] == 200.0f);

    cl.enqueueWriteBuffer(commandQueue, buffer, 0, [300.0f], event);
    cl.waitAndReleaseEvents(event);
    cl.enqueueReadBuffer(commandQueue, buffer, 0, data, event);
    cl.waitAndReleaseEvents(event);
    assert(data[0] == 300.0f);

    auto program = cl.createProgramFromSource(context, `
        __kernel void k(float value) {
            printf("value:%g\n", value);
        }
    `);
    scope(exit) cl.releaseProgram(program);
    cl.buildProgram(program, deviceIds);

    auto kernel = cl.createKernel(program, "k");
    scope(exit) cl.releaseKernel(kernel);

    cl.setKernelArg(kernel, 0, 100.0f);
    cl.enqueueKernel(commandQueue, kernel, [1], [1], event);
    cl.waitAndReleaseEvents(event);
    cl.flushCommandQueue(commandQueue);
    cl.finishCommandQueue(commandQueue);
}

