/*
 * opencv2/core/types.hpp - minimal stub for standalone build.
 * The code only uses this to check for ushort typedef.
 */
#ifndef __OPENCV_CORE_TYPES_HPP__
#define __OPENCV_CORE_TYPES_HPP__

// ushort is already defined by standard headers on Linux,
// but we define the guard so internal.h doesn't redefine it.
#ifndef _MSC_VER
#include <sys/types.h>  // provides u_short on Linux
typedef unsigned short ushort;
#endif

#endif // __OPENCV_CORE_TYPES_HPP__
