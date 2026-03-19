/*
*  opencl_utils.h
*	Simple OpenCL configuration
*
*  AUTHOR : Jeongyoun Yi <jeongyoun.yi@lge.com>
*	UPDATE : 2015-02-08
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __BILATERAL_H
#define __BILATERAL_H

#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"

using namespace pcl::device;
class cl_bilateral_pyrdown
{
private:
	enum { LEVELS = 3 };
	opencl_utils *clData;

	cl_program program;
	cl_kernel bilateralKernel;
	cl_kernel truncateDepthKernel;
	cl_kernel pyrDownKernel;
	cl_mem max_distance;
public:
	cl_bilateral_pyrdown(opencl_utils *_clData);
	~cl_bilateral_pyrdown();

	void bilateralInit();
	void clBilateralFilter(const DepthMap& rawDapth, DepthMap& depths_curr_);
	void clTruncateDepth(DepthMap& depths_curr_, float _max_distance);
	void clpyrDown(const DepthMap& src, DepthMap& dst);
};


#endif /* __BILATERAL_H */