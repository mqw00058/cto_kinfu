/*
*  estimate_combined.h
*	Simple OpenCL configuration
*
*  AUTHOR : Joon Yong Ji <jy.ji@lge.com>
*	UPDATE : 2015-02-08
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __ESTIMATE_COMBINED_H
#define __ESTIMATE_COMBINED_H

#include <CL/cl.h>
#include <iostream>
#include <pcl/gpu/containers/kernel_containers_cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"

typedef float float_type;

using namespace pcl::device;

class cl_estimate_combined
{
private:
	opencl_utils *clData;
	
	cl_program program;
	cl_kernel combinedKernel;
	cl_kernel TransformKernel;
	cl_mem params_estimated_combined;
	cl_mem cl_mem_params_TranformReduction;

public:
	cl_estimate_combined(opencl_utils *_clData);
	~cl_estimate_combined();

	void estimateCombinedInit();
	void estimateCombined(const Mat33& Rcurr, const float3& tcurr,
		MapArr& vmaps_curr_, MapArr& nmaps_curr_,
		const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
		MapArr& vmaps_g_prev_, MapArr& nmaps_g_prev_,
		float distThres, float angleThres,
		BufArr& gbuf_, BufArr& mbuf_,
		//DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, 
		float_type* matrixA_host, float_type* vectorB_host);
};

struct Combined
{
	enum
	{
		CTA_SIZE_X = 32,
		CTA_SIZE_Y = 8,
		CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
	};
};

struct CombinedParams
{
	float distThres;
	float angleThres;
	Intr intr;
	Mat33 Rcurr;
	float3 tcurr;
	Mat33 Rprev_inv;
	float3 tprev;
};

struct TranformReduction
{
	enum
	{
		CTA_SIZE = 512,
		STRIDE = CTA_SIZE,

		B = 6, COLS = 6, ROWS = 6, DIAG = 6,
		UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
		TOTAL = UPPER_DIAG_MAT + B,

		GRID_X = TOTAL
	};
};

struct TranformReductionParams
{
	int step;
	int length;
};

#endif /* __ESTIMATE_COMBINED_H */