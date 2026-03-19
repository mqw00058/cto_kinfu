/*
*  tsdf.h
*	Simple OpenCL configuration
*
*  AUTHOR : haksoo.moon <haksoo.moon@lge.com>
*	UPDATE : 2015-02-11
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __COROR_H
#define __COROR_H 
#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"
#include <assert.h>

using namespace pcl::gpu;
using namespace pcl::device;
typedef struct {
	float3 VOLUME;
	float3 cell_size;
} extractColorParams;
struct colorParams
{
	float resolution[3];
	float volume[3];
	float tranc_dist;
	int weight;
	int width;
	int height;

	Intr intr;
};
class cl_color_volume
{
private:
	opencl_utils *clData;
	cl_mem cl_mem_extractColor_params;
	cl_program program;
	cl_kernel colorKernel;
	cl_kernel initcolorKernel;
	cl_kernel extractColorsKernel;
public:
	cl_color_volume(opencl_utils *_clData);
	~cl_color_volume();

	void colorVolumeClInit();
	void updateColorVolume(const Intr& intr, float tranc_dist, const Mat33& Rcurr_inv, const float3& tcurr, const MapArr& vmap, const CLPtrStepSz<uchar3> colors, const float3& volume_size, CLDeviceArray2D<int> color_volume, int max_weight = 1);
	void initColorVolume(CLDeviceArray2D<int> color_volume, const int* volume_size);
	void extractColors(CLDeviceArray2D<int> color_volume, const float3& volume_size, const CLDeviceArray<float4>& points, CLDeviceArray<uchar4>& colors);
};

#endif /* __COROR_H */