/*
 * pcl_macros.h - minimal PCL macros stub for standalone build.
 */
#ifndef PCL_MACROS_H_
#define PCL_MACROS_H_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef PCLAPI_EXPORTS
    #define PCL_EXPORTS __declspec(dllexport)
  #else
    #define PCL_EXPORTS __declspec(dllimport)
  #endif
#else
  #if __GNUC__ >= 4
    #define PCL_EXPORTS __attribute__((visibility("default")))
  #else
    #define PCL_EXPORTS
  #endif
#endif

#define PCL_DEPRECATED(func) func
#define PCL_WARN(...) fprintf(stderr, ##__VA_ARGS__)

#include <cmath>
#ifndef pcl_isnan
#  define pcl_isnan(x) std::isnan(x)
#endif
#ifndef pcl_isinf
#  define pcl_isinf(x) std::isinf(x)
#endif
#ifndef pcl_isfinite
#  define pcl_isfinite(x) std::isfinite(x)
#endif

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#endif // PCL_MACROS_H_
