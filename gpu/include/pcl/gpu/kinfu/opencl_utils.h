/*
 *  opencl_utils.h
 *	Simple OpenCL configuration
 *
 *  AUTHOR : Jeongyoun Yi <jeongyoun.yi@lge.com>
 *	UPDATE : 2015-02-06
 *
 *  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
 */

#ifndef __OPENCL_UTILS_H
#define __OPENCL_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <pcl/gpu/kinfu/time.h>
#include "kinfu_config.h"
#ifdef __ANDROID__
#include <android/log.h>
#include <EGL/egl.h>
#include <CL/cl_gl.h>
#endif
//
//#ifdef __ANDROID__
//#include <android/log.h>
//#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "__OPENCL_UTILS_H", __VA_ARGS__) 
//#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , "__OPENCL_UTILS_H", __VA_ARGS__) 
//#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , "__OPENCL_UTILS_H", __VA_ARGS__) 
//#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , "__OPENCL_UTILS_H", __VA_ARGS__) 
//#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "__OPENCL_UTILS_H", __VA_ARGS__) 
//#else
//#define LOGV(...)
//#define LOGD(...) 
//#define LOGI(...) 
//#define LOGW(...) 
//#define LOGE(...) 
//#endif
//

// oh, apple. please.
#if defined( __APPLE__ )
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

//////////////////////
#define ALLOC_HOST_MEMORY
//////////////////////
#define CL_RELEASE_EVENT(e) { if(e) clReleaseEvent(e); e = 0; }
#define CL_RELEASE_MEM(e) { if(e) clReleaseMemObject(e); e = 0; }
#define CL_RELEASE_SAMPLER(e) { if(e) clReleaseSampler(e); e = 0; }

#if defined WIN32 || defined _WIN32 || defined WINCE || defined __MINGW32__
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// support platforms
#define HC_MAX_CL_DEVICES   (3)
#define HC_MAX_STRING_BUF   (1024)
//#ifdef _DEBUG
//# define dprintf(fmtstr, ...) printf(fmtstr, ##__VA_ARGS__)
//#else
//# define Debug(fmtstr, ...)
//#endif
#ifdef _DEBUG
#ifdef __ANDROID__
#define CHK_ERR(prefix, err)\
if (err != CL_SUCCESS)\
{\
__android_log_print(ANDROID_LOG_ERROR, "OpenCL", "CHK_ERR (%s, %d)\n", prefix, err); \
__android_log_print(ANDROID_LOG_ERROR, "OpenCL", "%s:%d\n", __FILE__, __LINE__); \
exit(EXIT_FAILURE); \
};
#else
void exit_break();
#define CHK_ERR(prefix, err)\
if (err != CL_SUCCESS)\
{\
fprintf(stderr, "CHK_ERR (%s, %d)\n", prefix, err); \
fprintf(stderr, "%s:%d\n", __FILE__, __LINE__); \
exit_break(); \
};
#endif
#else
#define CHK_ERR(prefix, err)
#endif

extern CL_Device g_select_device;

enum SUPPORTED_PLATFORM_VENDOR
{
    CL_PLATFORM_NVIDIA,
    CL_PLATFORM_AMD,
    CL_PLATFORM_APPLE,
    PLATFORM_VENDOR_SIZE
};

static const char* __support_platform_vendor[3] =
{
    "NVIDIA Corporation",
    "Advanced Micro Devices, Inc.",
    "Apple"
};

static const char* __preferred_device_name = "GeForce";

class cl_map {
public:
	cl_mem map;
	int cols;
	int rows;
};

class cl_map3D {
public:
	cl_mem map;
	int cols;
	int rows;
	int pages;
};

class opencl_utils
{
private:
	cl_platform_id		m_platform_id;
public:
	cl_device_id		m_device;
	cl_context			m_context;
	cl_command_queue	m_command_queue;
    cl_command_queue	m_command_queue_sub[3];
	cl_event			m_event_wait_list[3];
	bool				m_event_wait;	
	/* opencl m_device, m_context, m_command_queue, m_command_queue_sub */
	static opencl_utils* g_opencl_utils;
	DLL_EXPORT static opencl_utils* get();	
	DLL_EXPORT static opencl_utils* get(CL_Device d);	
	opencl_utils();
	virtual ~opencl_utils();private:
    inline cl_context getContext(){ return m_context; }
    inline cl_command_queue getCommandQueue(){ return m_command_queue; }
    inline cl_device_id getDeviceId(){ return m_device; }

    void openclInit_NVIDIA();
	void openclInit_INTEL();
	cl_context createContextFromType(cl_device_type device_type, cl_int* pErr, void* gl_context, void* gl_dc, int preferred_device_idx, int preferred_platform_idx, cl_platform_id* platform_id);
    cl_context createContextFromPlatform(cl_platform_id platform, cl_device_type device_type, cl_int* pErr, void* gl_context, void* gl_dc, int preferred_device_idx, int preferred_platform_idx);    
    cl_device_id getDevice(cl_context context);
    bool isPreferredDevice(cl_device_id device);
    bool isSupportedVendor(const char* platform_vendor_name);
    
    
    
public:
    // For debugging purpose
    void printPlatformInfo(cl_platform_id platform);
    void printDeviceInfo(cl_device_id device);
    
    // debugging
    inline static void checkErr(const char* func, cl_int err){ if(err != CL_SUCCESS) printf("[%s] %s\n", func, getErrorString(err)); /* else printf("[%s] SUCCESS\n", func); */ }
    static const char *getErrorString(cl_int error);

public:
	cl_program buildProgram(const char* source, size_t size, const char* option = NULL);
	cl_program buildProgramFromFile(const char* path, const char* option = NULL);
	cl_kernel compileKernelFromFile(cl_program program, const char* name, cl_int* err);
	char* readText(const char* path, long *cnt);

	double getElapsedTimeInMS(cl_event* event);
};

#endif // __OPENCL_UTILS_H