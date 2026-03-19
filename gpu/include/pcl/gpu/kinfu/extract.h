/*
*  tsdf.h
*	Simple OpenCL configuration
*
*  AUTHOR : haksoo.moon <haksoo.moon@lge.com>
*	UPDATE : 2015-02-11
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __EXTRACT_H
#define __EXTRACT_H
#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"
#include <assert.h>

using namespace pcl::gpu;
using namespace pcl::device;

typedef struct {
	float3 VOLUME;
	float3 cell_size;
} ExtractParams;

class cl_extract
{

	enum
	{
		CTA_SIZE_X = 32,
		CTA_SIZE_Y = 8,
		CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,
		MAX_LOCAL_POINTS = 3
	};
private:
	opencl_utils *clData;

	cl_program program;
	cl_kernel extractKernel;
	cl_kernel extractNormalsKernel;
	cl_kernel extractNormals2Kernel;

	cl_mem cl_mem_cell_size; 
	cl_mem cl_mem_extract_params;


public:
	cl_extract(opencl_utils *_clData);
	~cl_extract();

	void extractInit();
	void extractNormals(const CLDeviceArray2D<int> volume, const float3 volume_size, const CLDeviceArray<float4>& input, CLDeviceArray<float4>& output);
	void extractNormals2(const CLDeviceArray2D<int> volume, const float3 volume_size, const CLDeviceArray<float4>& input, CLDeviceArray<float8>& output);
	size_t extractCloud(const CLDeviceArray2D<int> volume, const float3 volume_size, CLDeviceArray<float4>& output);
};

#endif /* __EXTRACT_H */