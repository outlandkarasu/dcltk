import std.stdio : writefln;

import cl = dcltk;

void main() {
    cl.loadOpenCl();

    foreach(i, id; cl.getPlatformIds()) {
        writefln("%d: %s %s %s(%s) %s [%s]",
                i,
                cl.getPlatformProfile(id),
                cl.getPlatformName(id),
                cl.getPlatformVersion(id),
                cl.getPlatformCLVersion(id),
                cl.getPlatformVendor(id),
                cl.getPlatformExtensions(id));
    }
}

