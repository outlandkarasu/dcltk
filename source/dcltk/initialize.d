module dcltk.initialize;

import derelict.opencl.cl;

import dcltk.error : OpenClException;
import dcltk.platform : getPlatformIds, getPlatformCLVersion;

import std.algorithm : map, maxElement;

/**
 *  load highest OpenCL libraries.
 *
 *  Returns:
 *      loaded platform ID.
 */
cl_platform_id loadOpenCl() {
    DerelictCL.load();

    struct PlatformVersion {
        cl_platform_id id;
        CLVersion clVersion;
    }

    auto versions = getPlatformIds()
        .map!((id) => PlatformVersion(id, getPlatformCLVersion(id)));
    if(versions.empty) {
        throw new OpenClException("nothing platform.");
    }

    auto highestVersion = versions.maxElement!"a.clVersion";
    DerelictCL.reload(highestVersion.clVersion);
    DerelictCL.loadEXT(highestVersion.id);
    return highestVersion.id;
}

/**
 *  initialize OpenCL libraries.
 */
void initializeOpenCL() {
    DerelictCL.load();
}

/**
 *  reload OpenCL libraries.
 *
 *  Params:
 *      platformId = target platform ID.
 *      version = target version.
 */
void reloadOpenCL(cl_platform_id platformId, CLVersion version)
{
    DerelictCL.reload(version);
    DerelictCL.loadEXT(platformId);
}

