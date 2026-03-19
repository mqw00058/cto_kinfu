/*
*  tsdf.h
*	Simple OpenCL configuration
*
*  AUTHOR : haksoo.moon <haksoo.moon@lge.com>
*	UPDATE : 2015-02-11
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __TSDF_H
#define __TSDF_H
#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"
#include <assert.h>

using namespace pcl::gpu;
using namespace pcl::device;

struct TsdfParams
{
	float resolution[3];
	float volume[3];
	float tranc_dist;
	int width;
	int height;
	Intr intr;
	Mat33 Rcurr_inv;
	float3 tcurr;
};
class cl_tsdf_volume
{
private:
	opencl_utils *clData;

	cl_program program;
	cl_kernel tsdfKernel;
	cl_kernel scaleDepthKernel;
	cl_kernel initKernel;
	cl_mem cl_mem_Intr_inv;
	cl_mem cl_mem_params_integrateTsdfVolume;
	cl_mem cl_mem_hvolume_size;
public:
	cl_tsdf_volume(opencl_utils *_clData);
	~cl_tsdf_volume();

	void tsdfInit();
	void integrateTsdfVolume(CLDeviceArray2D<ushort> src, const Intr& intr, const float3& volume_size,
		const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, CLPtrStep<short2> volume, CLDeviceArray2D<float> scaled);
	void initVolume(CLDeviceArray2D<int> volume, const int* volume_size);
};

#endif /* _TSDF_H */