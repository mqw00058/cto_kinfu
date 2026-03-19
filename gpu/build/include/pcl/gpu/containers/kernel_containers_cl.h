/*
 * kernel_containers_cl.h
 * Lightweight kernel-side view types for OpenCL buffers.
 * These are passed to kernel launcher functions and hold a cl_mem handle
 * plus dimensions. They do NOT own the memory.
 *
 * Stub header recreated for standalone build.
 */

#ifndef PCL_GPU_CONTAINERS_KERNEL_CONTAINERS_CL_H_
#define PCL_GPU_CONTAINERS_KERNEL_CONTAINERS_CL_H_

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Forward declare the container classes to avoid circular include
// (device_array_cl.h includes this file)
namespace pcl { namespace gpu {
  template<typename T> class CLDeviceArray;
  template<typename T> class CLDeviceArray2D;
  class CLDeviceImage2D;
} }
using pcl::gpu::CLDeviceArray;
using pcl::gpu::CLDeviceArray2D;
using pcl::gpu::CLDeviceImage2D;

/** \brief 2D pointer + step (cols). Non-owning view over a CLDeviceArray2D.
 *  Supports cross-type construction (e.g. CLDeviceArray2D<int> -> CLPtrStep<short2>)
 *  because the TSDF volume packs short2 pairs into int elements.
 */
template<typename T>
struct CLPtrStep
{
  cl_mem handle;  ///< Raw OpenCL buffer
  int    step;    ///< Row stride in elements

  CLPtrStep() : handle(NULL), step(0) {}

  /** \brief Construct from a same-type 2D device array. */
  CLPtrStep(CLDeviceArray2D<T>& arr)
    : handle(arr.handle() ? *arr.handle() : NULL)
    , step(arr.cols())
  {}

  CLPtrStep(const CLDeviceArray2D<T>& arr)
    : handle(const_cast<CLDeviceArray2D<T>&>(arr).handle()
             ? *const_cast<CLDeviceArray2D<T>&>(arr).handle() : NULL)
    , step(arr.cols())
  {}

  /** \brief Cross-type constructor: borrow the cl_mem from a differently-typed array.
   *  Used e.g. when CLDeviceArray2D<int> is used as CLPtrStep<short2> (type-punning). */
  template<typename U>
  CLPtrStep(CLDeviceArray2D<U>& arr)
    : handle(arr.handle() ? *arr.handle() : NULL)
    , step(arr.cols())
  {}

  template<typename U>
  CLPtrStep(const CLDeviceArray2D<U>& arr)
    : handle(const_cast<CLDeviceArray2D<U>&>(arr).handle()
             ? *const_cast<CLDeviceArray2D<U>&>(arr).handle() : NULL)
    , step(const_cast<CLDeviceArray2D<U>&>(arr).cols())
  {}
};

/** \brief 2D pointer + step + size (rows, cols). Non-owning view. */
template<typename T>
struct CLPtrStepSz : public CLPtrStep<T>
{
  int rows;
  int cols;

  CLPtrStepSz() : CLPtrStep<T>(), rows(0), cols(0) {}

  CLPtrStepSz(CLDeviceArray2D<T>& arr)
    : CLPtrStep<T>(arr)
    , rows(arr.rows())
    , cols(arr.cols())
  {}

  CLPtrStepSz(const CLDeviceArray2D<T>& arr)
    : CLPtrStep<T>(arr)
    , rows(arr.rows())
    , cols(arr.cols())
  {}

  /** \brief Cross-type constructor for type-punned 2D arrays. */
  template<typename U>
  CLPtrStepSz(CLDeviceArray2D<U>& arr)
    : CLPtrStep<T>(arr)
    , rows(arr.rows())
    , cols(arr.cols())
  {}

  template<typename U>
  CLPtrStepSz(const CLDeviceArray2D<U>& arr)
    : CLPtrStep<T>(arr)
    , rows(arr.rows())
    , cols(arr.cols())
  {}
};

/** \brief 1D pointer + size. Non-owning view over a CLDeviceArray. */
template<typename T>
struct CLPtrSz
{
  cl_mem handle;  ///< Raw OpenCL buffer
  int    size;    ///< Number of elements

  CLPtrSz() : handle(NULL), size(0) {}

  CLPtrSz(CLDeviceArray<T>& arr)
    : handle(arr.handle() ? *arr.handle() : NULL)
    , size((int)arr.size())
  {}

  CLPtrSz(const CLDeviceArray<T>& arr)
    : handle(const_cast<CLDeviceArray<T>&>(arr).handle()
             ? *const_cast<CLDeviceArray<T>&>(arr).handle() : NULL)
    , size((int)arr.size())
  {}
};

#endif // PCL_GPU_CONTAINERS_KERNEL_CONTAINERS_CL_H_
