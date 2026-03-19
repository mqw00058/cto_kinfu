/*
*  opencl_utils.h
*	Simple OpenCL configuration
*
*  AUTHOR : Jeongyoun Yi <jeongyoun.yi@lge.com>
*	UPDATE : 2015-02-08
*
*  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
*/

#ifndef __MAPS_H
#define __MAPS_H

#include <CL/cl.h>
#include <pcl/gpu/kinfu/opencl_utils.h>
#include "internal.h"

using namespace pcl::device;

struct TranformMapParams
{
	Mat33 Rmat;
	float3 tvec;
};

class cl_maps
{
private:
	opencl_utils *clData;

	cl_program program;
	cl_kernel computeVmapKernel;
	cl_kernel computeNmapKernel;
	cl_kernel transformVMapsKernel;
	cl_kernel transformNMapsKernel;
	cl_kernel resizeVMapKernel;
	cl_kernel resizeNMapKernel;
	cl_kernel resizeDepthMapKernel;
	cl_kernel resizeDepthMapKernel2;
	cl_kernel convertMapKernel;
	cl_kernel convertMap2Kernel;
	cl_kernel mergePointNormalKernel;
	cl_mem cl_mem_Intr_inv;
	cl_mem cl_mem_params_cltranformMap;
public:
	cl_maps(opencl_utils *_clData);
	~cl_maps();
	void mapsInit();

	void clCreateVMap(const Intr& intr, const DepthMap& src, MapArr& dst);
	void clCreateNMap(const MapArr& vmap, MapArr& nmap);
	void cltranformMap(const MapArr& vmaps_curr_, const MapArr& nmaps_curr_, const Mat33& Rmat, const float3& tvec, MapArr& vmaps_g_prev_, MapArr& nmaps_g_prev_);
	void clresizeVMap(const MapArr& input, MapArr& output);
	void clresizeNMap(const MapArr& input, MapArr& output);
	void clresizeDepthMap(const DepthMap& input, DepthMap& output);
	void clresizeDepthMap2(const MapArr& input, DepthMap& output);
	void convert(const MapArr& vmap, CLDeviceArray2D<float4>& output);
	void convert(const MapArr& vmap, CLDeviceArray2D<float8>& output);
	void mergePointNormal(const CLDeviceArray<float4>& cloud, const CLDeviceArray<float8>& normals, const CLDeviceArray<float12>& output);
};

#endif /* __MAPS_H */