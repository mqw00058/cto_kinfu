/*
*  tsdf.h
*	Simple OpenCL configuration
*
*  AUTHOR : haksoo.moon <haksoo.moon@lge.com>
*	UPDATE : 2015-02-11
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __GI_H
#define __GI_H
#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include <pcl/gpu/kinfu/pixel_rgb.h>
#include "internal.h"
#include <assert.h>

using namespace pcl::device;

struct lightParams
{
	float pos[3];
	int number;

};
class cl_generateImage
{
private:
	opencl_utils *clData;

	cl_program program;
	cl_kernel GIKernel;
	cl_kernel GIKernel2;
	cl_kernel paint3DViewKernel;
public:
	cl_generateImage(opencl_utils *_clData);
	~cl_generateImage();

	void GIInit();
	void generateImage(const MapArr& vmaps_g_prev_, const MapArr& nmaps_g_prev_, pcl::device::LightSource light, CLPtrStepSz<uchar3> view);
#ifdef __ANDROID__
	void generateImage(const MapArr& vmaps_g_prev_, const MapArr& nmaps_g_prev_, pcl::device::LightSource light, MapArr& view);
#endif
	void paint3DView(const CLPtrStep<uchar3> colors, CLPtrStepSz<uchar3> dst, float colors_weight = 0.5f);
};


#endif /* __GI_H */