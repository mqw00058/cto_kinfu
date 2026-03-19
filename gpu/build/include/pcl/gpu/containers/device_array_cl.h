/*
 * device_array_cl.h
 * OpenCL device array containers for PCL GPU kinfu OpenCL port.
 * Based on PCL GPU containers API, adapted for OpenCL by LG Electronics.
 * Stub header recreated for standalone build.
 */

#ifndef PCL_GPU_CONTAINERS_DEVICE_ARRAY_CL_H_
#define PCL_GPU_CONTAINERS_DEVICE_ARRAY_CL_H_

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <cstring>
#include <vector>

// Make PCL_EXPORTS available without requiring pcl/pcl_macros.h to be included first
#ifndef PCL_EXPORTS
#  if defined _WIN32 || defined __CYGWIN__
#    ifdef PCLAPI_EXPORTS
#      define PCL_EXPORTS __declspec(dllexport)
#    else
#      define PCL_EXPORTS __declspec(dllimport)
#    endif
#  else
#    if __GNUC__ >= 4
#      define PCL_EXPORTS __attribute__((visibility("default")))
#    else
#      define PCL_EXPORTS
#    endif
#  endif
#endif

namespace pcl
{
  namespace gpu
  {
    /** \brief 1D device array (OpenCL buffer). */
    template<typename T>
    class CLDeviceArray
    {
    public:
      CLDeviceArray() : context_(NULL), mem_(NULL), size_(0) {}

      /** \brief Construct from null literal (returns empty array). */
      CLDeviceArray(std::nullptr_t) : context_(NULL), mem_(NULL), size_(0) {}

      /** \brief Construct from integer 0 (null literal compatibility). */
      CLDeviceArray(int null_val) : context_(NULL), mem_(NULL), size_(0)
      { (void)null_val; /* must be 0 */ }

      explicit CLDeviceArray(cl_context ctx) : context_(ctx), mem_(NULL), size_(0) {}

      /** \brief Construct a non-owning view over an existing buffer with given size (element count). */
      CLDeviceArray(cl_context ctx, cl_mem* existing, std::size_t count)
        : context_(ctx), mem_(NULL), size_(count)
      {
        if (existing && *existing)
        {
          mem_ = *existing;
          clRetainMemObject(mem_);
        }
      }

      CLDeviceArray(const CLDeviceArray& other)
        : context_(other.context_), mem_(other.mem_), size_(other.size_)
      {
        if (mem_) clRetainMemObject(mem_);
      }

      ~CLDeviceArray()
      {
        release();
      }

      CLDeviceArray& operator=(const CLDeviceArray& other)
      {
        if (this != &other)
        {
          release();
          context_ = other.context_;
          mem_ = other.mem_;
          size_ = other.size_;
          if (mem_) clRetainMemObject(mem_);
        }
        return *this;
      }

      /** \brief Allocate GPU memory for count elements. */
      void create(std::size_t count)
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        size_ = count;
        if (count > 0 && context_)
        {
          cl_int err;
          mem_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, count * sizeof(T), NULL, &err);
        }
      }

      /** \brief Upload data from host to device. */
      void upload(cl_command_queue queue, const void* data, std::size_t count)
      {
        create(count);
        if (mem_ && data)
          clEnqueueWriteBuffer(queue, mem_, CL_TRUE, 0, count * sizeof(T), data, 0, NULL, NULL);
      }

      /** \brief Download data from device to host (first 'count' elements). */
      void download(cl_command_queue queue, void* data, std::size_t count) const
      {
        if (mem_ && data)
          clEnqueueReadBuffer(queue, mem_, CL_TRUE, 0, count * sizeof(T), data, 0, NULL, NULL);
      }

      /** \brief Download all data from device to host. */
      void download(cl_command_queue queue, void* data) const
      {
        if (mem_ && data)
          clEnqueueReadBuffer(queue, mem_, CL_TRUE, 0, size_ * sizeof(T), data, 0, NULL, NULL);
      }

      /** \brief Returns pointer to internal cl_mem. */
      cl_mem* handle() { return &mem_; }
      const cl_mem* handle() const { return &mem_; }

      /** \brief Returns number of elements. */
      std::size_t size() const { return size_; }

      /** \brief Returns size in bytes. */
      std::size_t sizeBytes() const { return size_ * sizeof(T); }

      /** \brief Returns true if empty. */
      bool empty() const { return size_ == 0 || mem_ == NULL; }

      void release()
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        size_ = 0;
      }

    private:
      cl_context context_;
      cl_mem mem_;
      std::size_t size_;
    };


    /** \brief 2D device array (OpenCL buffer, row-major). */
    template<typename T>
    class CLDeviceArray2D
    {
    public:
      CLDeviceArray2D() : context_(NULL), mem_(NULL), rows_(0), cols_(0) {}

      explicit CLDeviceArray2D(cl_context ctx) : context_(ctx), mem_(NULL), rows_(0), cols_(0) {}

      CLDeviceArray2D(const CLDeviceArray2D& other)
        : context_(other.context_), mem_(other.mem_), rows_(other.rows_), cols_(other.cols_)
      {
        if (mem_) clRetainMemObject(mem_);
      }

      ~CLDeviceArray2D()
      {
        release();
      }

      CLDeviceArray2D& operator=(const CLDeviceArray2D& other)
      {
        if (this != &other)
        {
          release();
          context_ = other.context_;
          mem_ = other.mem_;
          rows_ = other.rows_;
          cols_ = other.cols_;
          if (mem_) clRetainMemObject(mem_);
        }
        return *this;
      }

      /** \brief Allocate GPU memory for rows x cols elements. */
      void create(int rows, int cols)
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        rows_ = rows;
        cols_ = cols;
        if (rows > 0 && cols > 0 && context_)
        {
          cl_int err;
          mem_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                (std::size_t)rows * cols * sizeof(T), NULL, &err);
        }
      }

      /** \brief Upload row-major data from host. stride is step in bytes (0 = tight). */
      void upload(cl_command_queue queue, const void* data, int stride, int cols, int rows)
      {
        create(rows, cols);
        if (mem_ && data)
          clEnqueueWriteBuffer(queue, mem_, CL_TRUE, 0,
                               (std::size_t)rows * cols * sizeof(T), data, 0, NULL, NULL);
      }

      /** \brief Download row-major data to host. stride is step in bytes (0 = tight). */
      void download(cl_command_queue queue, void* data, int stride) const
      {
        if (mem_ && data)
          clEnqueueReadBuffer(queue, mem_, CL_TRUE, 0,
                              (std::size_t)rows_ * cols_ * sizeof(T), data, 0, NULL, NULL);
      }

      /** \brief Convenience overload: download into a std::vector<T>, filling 'cols' with cols(). */
      void download(cl_command_queue queue, std::vector<T>& vec, int& out_cols) const
      {
        vec.resize((std::size_t)rows_ * cols_);
        out_cols = cols_;
        if (mem_ && !vec.empty())
          clEnqueueReadBuffer(queue, mem_, CL_TRUE, 0,
                              vec.size() * sizeof(T), vec.data(), 0, NULL, NULL);
      }

      /** \brief Returns pointer to internal cl_mem. */
      cl_mem* handle() { return &mem_; }
      const cl_mem* handle() const { return &mem_; }

      int rows() const { return rows_; }
      int cols() const { return cols_; }

      bool empty() const { return rows_ == 0 || cols_ == 0 || mem_ == NULL; }

      void release()
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        rows_ = 0; cols_ = 0;
      }

    private:
      cl_context context_;
      cl_mem mem_;
      int rows_;
      int cols_;
    };


    /** \brief 2D OpenCL Image object (used for vertex/normal maps). */
    class CLDeviceImage2D
    {
    public:
      CLDeviceImage2D() : context_(NULL), mem_(NULL), rows_(0), cols_(0) {}

      explicit CLDeviceImage2D(cl_context ctx) : context_(ctx), mem_(NULL), rows_(0), cols_(0) {}

      CLDeviceImage2D(const CLDeviceImage2D& other)
        : context_(other.context_), mem_(other.mem_), rows_(other.rows_), cols_(other.cols_)
      {
        if (mem_) clRetainMemObject(mem_);
      }

      ~CLDeviceImage2D()
      {
        release();
      }

      CLDeviceImage2D& operator=(const CLDeviceImage2D& other)
      {
        if (this != &other)
        {
          release();
          context_ = other.context_;
          mem_ = other.mem_;
          rows_ = other.rows_;
          cols_ = other.cols_;
          if (mem_) clRetainMemObject(mem_);
        }
        return *this;
      }

      /** \brief Create a 2D OpenCL image. */
      void create(int width, int height, const cl_image_format* fmt)
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        cols_ = width;
        rows_ = height;
        if (width > 0 && height > 0 && context_ && fmt)
        {
          cl_int err;
#ifdef CL_VERSION_1_2
          cl_image_desc desc;
          memset(&desc, 0, sizeof(desc));
          desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
          desc.image_width  = (std::size_t)width;
          desc.image_height = (std::size_t)height;
          mem_ = clCreateImage(context_, CL_MEM_READ_WRITE, fmt, &desc, NULL, &err);
#else
          mem_ = clCreateImage2D(context_, CL_MEM_READ_WRITE, fmt,
                                 (std::size_t)width, (std::size_t)height,
                                 0, NULL, &err);
#endif
        }
      }

#ifdef CL_GL_INTEROP
      /** \brief Create a 2D OpenCL image from an OpenGL texture. */
      cl_int create(int width, int height, const cl_image_format* fmt, unsigned int gl_tex)
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        cols_ = width;
        rows_ = height;
        cl_int err = CL_SUCCESS;
        if (width > 0 && height > 0 && context_)
        {
          mem_ = clCreateFromGLTexture2D(context_, CL_MEM_READ_WRITE, 0x0DE1 /*GL_TEXTURE_2D*/, 0, gl_tex, &err);
        }
        return err;
      }
#endif

      /** \brief Upload pixel data to the image. */
      void upload(cl_command_queue queue, const void* data, int stride, int width, int height,
                  const cl_image_format* /*fmt*/)
      {
        if (mem_ && data)
        {
          std::size_t origin[3] = {0, 0, 0};
          std::size_t region[3] = {(std::size_t)width, (std::size_t)height, 1};
          clEnqueueWriteImage(queue, mem_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
        }
      }

      /** \brief Download pixel data from the image. */
      void download(cl_command_queue queue, void* data, int /*stride*/) const
      {
        if (mem_ && data)
        {
          std::size_t origin[3] = {0, 0, 0};
          std::size_t region[3] = {(std::size_t)cols_, (std::size_t)rows_, 1};
          clEnqueueReadImage(queue, mem_, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
        }
      }

      /** \brief Returns pointer to internal cl_mem. */
      cl_mem* handle() { return &mem_; }
      const cl_mem* handle() const { return &mem_; }

      int rows() const { return rows_; }
      int cols() const { return cols_; }
      // Aliases used by some source files
      int height() const { return rows_; }
      int width()  const { return cols_; }

      bool empty() const { return rows_ == 0 || cols_ == 0 || mem_ == NULL; }

      void release()
      {
        if (mem_) { clReleaseMemObject(mem_); mem_ = NULL; }
        rows_ = 0; cols_ = 0;
      }

    private:
      cl_context context_;
      cl_mem mem_;
      int rows_;
      int cols_;
    };

  } // namespace gpu
} // namespace pcl

// Pull container types into global scope for backward compatibility
using pcl::gpu::CLDeviceArray;
using pcl::gpu::CLDeviceArray2D;
using pcl::gpu::CLDeviceImage2D;

// Include kernel container view types (CLPtrStep, CLPtrStepSz, CLPtrSz)
// These are needed by internal.h which only includes device_array_cl.h
#include <pcl/gpu/containers/kernel_containers_cl.h>

#endif // PCL_GPU_CONTAINERS_DEVICE_ARRAY_CL_H_
