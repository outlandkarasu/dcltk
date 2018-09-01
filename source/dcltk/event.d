module dcltk.event;

import derelict.opencl.cl;

import dcltk.error : enforceCl;

/**
 *  wait for events.
 *
 *  Params:
 *      events = wait events.
 */
void waitForEvents(const(cl_event)[] events...) {
    enforceCl(clWaitForEvents(cast(cl_uint)events.length, events.ptr));
}

/// release event.
void releaseEvent(cl_event event) {
    enforceCl(clReleaseEvent(event));
}

/**
 *  wait and release events.
 *
 *  Params:
 *      events = wait and release target events..
 */
void waitAndReleaseEvents(cl_event[] events...) {
    waitForEvents(events);
    foreach(e; events) {
        releaseEvent(e);
    }
}
