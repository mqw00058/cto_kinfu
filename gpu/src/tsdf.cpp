#include <pcl/gpu/kinfu/tsdf.h>
#ifdef __ANDROID__
#include "cl/tsdf.cl.h"
#endif

cl_tsdf_volume::cl_tsdf_volume(opencl_utils *_clData) {
	program = NULL;
	tsdfKernel = NULL;
	scaleDepthKernel = NULL;
	initKernel = NULL;
	clData = _clData;
	tsdfInit();
}

cl_tsdf_volume::~cl_tsdf_volume()
{
	clReleaseMemObject(cl_mem_params_integrateTsdfVolume);
	clReleaseMemObject(cl_mem_hvolume_size);
	clReleaseKernel(tsdfKernel);
	clReleaseKernel(scaleDepthKernel);
	clReleaseKernel(initKernel);
	clReleaseProgram(program);
}

void cl_tsdf_volume::tsdfInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)tsdf_cl, sizeof(tsdf_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/tsdf.cl");
#endif
	tsdfKernel = clData->compileKernelFromFile(program, "TSDF", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	initKernel = clData->compileKernelFromFile(program, "initializeVolume", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	scaleDepthKernel = clData->compileKernelFromFile(program, "scaleDepth", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_params_integrateTsdfVolume = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(TsdfParams), NULL, &err);
#else
	cl_mem_params_integrateTsdfVolume = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(TsdfParams), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_Intr_inv = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(Intr), NULL, &err);
#else
	cl_mem_Intr_inv = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(Intr), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_hvolume_size = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*3, NULL, &err);
#else
	cl_mem_hvolume_size = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(int)*3, NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
}

void cl_tsdf_volume::integrateTsdfVolume(CLDeviceArray2D<ushort> src, const Intr& intr, const float3& volume_size,
	const Mat33& Rcurr_inv, const float3& tcurr, float tranc_dist, CLPtrStep<short2> volume, CLDeviceArray2D<float> scaled)
{
	cl_int err;
	Intr intr_inv = intr(-1);
#ifdef ALLOC_HOST_MEMORY
	Intr* intr_inv_params = (Intr*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_Intr_inv, CL_TRUE, CL_MAP_WRITE, 0, sizeof(Intr), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	*intr_inv_params = intr_inv;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_Intr_inv, intr_inv_params, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_Intr_inv, CL_TRUE, 0, sizeof(Intr), &intr_inv, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	cl_uint j = 0;
	err = clSetKernelArg(scaleDepthKernel, j++, sizeof(cl_mem), (void*)src.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(scaleDepthKernel, j++, sizeof(cl_mem), (void*)scaled.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(scaleDepthKernel, j++, sizeof(cl_mem), (void*)&cl_mem_Intr_inv);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)(src.cols()), (size_t)(src.rows()) };
	size_t local_ws[2] = { 16, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, scaleDepthKernel, 2, 0, global_ws, local_ws, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);



#ifdef ALLOC_HOST_MEMORY
	TsdfParams* pParams = (TsdfParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_params_integrateTsdfVolume, CL_TRUE, CL_MAP_WRITE, 0, sizeof(TsdfParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->resolution[0] = VOLUME_X;
	pParams->resolution[1] = VOLUME_Y;
	pParams->resolution[2] = VOLUME_Z;
	pParams->volume[0] = volume_size.x;
	pParams->volume[1] = volume_size.y;
	pParams->volume[2] = volume_size.z;
	pParams->tranc_dist = tranc_dist;
	pParams->width = src.cols();
	pParams->height = src.rows();
	pParams->intr = intr;        
	pParams->Rcurr_inv = Rcurr_inv;
	pParams->tcurr = tcurr;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_params_integrateTsdfVolume, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	TsdfParams params;
	params.resolution[0] = VOLUME_X;
	params.resolution[1] = VOLUME_Y;
	params.resolution[2] = VOLUME_Z;
	params.volume[0] = volume_size.x;
	params.volume[1] = volume_size.y;
	params.volume[2] = volume_size.z;
	params.tranc_dist = tranc_dist;
	params.width = src.cols();
	params.height = src.rows();
	params.intr = intr;
	params.Rcurr_inv = Rcurr_inv;
	params.tcurr = tcurr;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_params_integrateTsdfVolume, CL_TRUE, 0, sizeof(TsdfParams), &params, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	j = 0;
	err = clSetKernelArg(tsdfKernel, j++, sizeof(cl_mem), (void*)&volume.handle);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(tsdfKernel, j++, sizeof(cl_mem), (void*)scaled.handle()); //depth map
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(tsdfKernel, j++, sizeof(cl_mem), (void*)&cl_mem_params_integrateTsdfVolume);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws2[2] = { VOLUME_X, VOLUME_Y };
	size_t local_ws2[2] = { 16, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, tsdfKernel, 2, 0, global_ws2, local_ws2, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_tsdf_volume::initVolume(CLDeviceArray2D<int> volume, const int* volume_size)
{
	cl_int err;
#ifdef ALLOC_HOST_MEMORY
	int* pParams = (int*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_hvolume_size, CL_TRUE, CL_MAP_WRITE, 0, sizeof(int)* 3, 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams[0] = volume_size[0];
	pParams[1] = volume_size[1];
	pParams[2] = volume_size[2];
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_hvolume_size, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_hvolume_size, CL_TRUE, 0, sizeof(int)*3, volume_size, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	cl_uint j = 0;
	err = clSetKernelArg(initKernel, j++, sizeof(cl_mem), (void*)volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(initKernel, j++, sizeof(cl_mem), (void*)&cl_mem_hvolume_size);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { VOLUME_X, VOLUME_Y };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, initKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
}
