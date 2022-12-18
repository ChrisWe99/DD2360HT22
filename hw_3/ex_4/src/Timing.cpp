#include "Timing.h"

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// tried to measure in milliseconds
double cpuMillisecond()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec*1.e3 + (double)tp.tv_usec*1.e-3);
}
