/*
*  ray_caster.cl.h
*	ray_caster.cu OpenCL implementation
*
*  AUTHOR : SeungYong Woo <sy.woo@lge.com>
*	UPDATE : 2015-02-08
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __cl_ray_caster_H__
#define __cl_ray_caster_H__

#include <CL/cl.h>
#include <pcl/gpu/containers/device_array_cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"
#include <assert.h>

using namespace pcl::device;
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };
struct RayCasterParams_CL
{

	
	pcl::device::Mat33 Rcurr;
	float3 tcurr;
	
	float time_step;
	float3 volume_size;
	
	float3 cell_size;
	
	pcl::device::Intr intr;

	int VOLUME_X;
	int VOLUME_Y;
	int VOLUME_Z;
};

class cl_ray_caster
{
private:
	opencl_utils *clData;

	cl_program program;
	cl_kernel rayCastKernel;
	cl_mem cl_mem_params;
public:
	cl_ray_caster(opencl_utils *_clData);
	~cl_ray_caster();

	void setclData(opencl_utils *);
	void init();
	void raycast(const pcl::device::Intr& intr, const pcl::device::Mat33& Rcurr, const float3& tcurr,
		float tranc_dist, const float3& volume_size,
		const CLDeviceArray2D<int>& volume, MapArr& vmap, MapArr& nmap);
};

#endif /* __cl_ray_caster_H__ */