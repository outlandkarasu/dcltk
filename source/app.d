import std.stdio : writefln;

import cl = dcltk;

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
    auto context = cl.createContext(platformId, deviceIds);
    scope(exit) cl.releaseContext(context);
}

