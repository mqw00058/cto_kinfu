#include <pcl/gpu/kinfu/bilateral.h>
#ifdef __ANDROID__
#include "cl/bilateral.cl.h"
#endif

cl_bilateral_pyrdown::cl_bilateral_pyrdown(opencl_utils *_clData) {
	program = NULL;
	bilateralKernel = NULL;
	truncateDepthKernel = NULL;
	pyrDownKernel = NULL;
	clData = _clData;
	bilateralInit();
}

cl_bilateral_pyrdown::~cl_bilateral_pyrdown() {
	clReleaseMemObject(max_distance);
	clReleaseKernel(bilateralKernel);
	clReleaseKernel(truncateDepthKernel);
	clReleaseKernel(pyrDownKernel);
	clReleaseProgram(program);
}

void cl_bilateral_pyrdown::bilateralInit()
{
	cl_int ret;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)bilateral_cl, sizeof(bilateral_cl));
#else
	//program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/bilateral.cl");
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/bilateral.cl");
#endif
	bilateralKernel = clData->compileKernelFromFile(program, "bilateralF", &ret);
	CHK_ERR("clCreateKernel Failed", ret);
	truncateDepthKernel = clData->compileKernelFromFile(program, "truncateDepth", &ret);
	CHK_ERR("clCreateKernel Failed", ret);
	pyrDownKernel = clData->compileKernelFromFile(program, "pyrDown", &ret);
	CHK_ERR("clCreateKernel Failed", ret);
#ifdef ALLOC_HOST_MEMORY
	max_distance = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float), NULL, &ret);
#else
	max_distance = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(float), NULL, &ret);
#endif
	CHK_ERR("clCreateBuffer Failed", ret);
}

void cl_bilateral_pyrdown::clBilateralFilter(const DepthMap& rawDepth, DepthMap& depths_curr_) {
	cl_int ret;

	const float sigma_color = 30;		// in mm
	const float sigma_space = 4.5;		// in pixels

	cl_uint j = 0;
	ret = clSetKernelArg(bilateralKernel, j++, sizeof(cl_mem), (void*)rawDepth.handle());
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(bilateralKernel, j++, sizeof(cl_mem), (void*)depths_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	size_t globalWorkSize[2] = { (size_t)rawDepth.cols(), (size_t)rawDepth.rows() };
	const size_t localWorkSize[2] = { 16, 16 };
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, bilateralKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	
	/* Finalization */
	ret = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	return;
}

void cl_bilateral_pyrdown::clTruncateDepth(DepthMap& depths_curr_, float _max_distance)
{
	cl_int ret;
#ifdef ALLOC_HOST_MEMORY
	float* max_distance_h = (float*)clEnqueueMapBuffer(clData->m_command_queue, max_distance, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float), 0, NULL, NULL, &ret);
	CHK_ERR("clEnqueueMapBuffer Failed", ret);
	*max_distance_h = _max_distance;
	ret = clEnqueueUnmapMemObject(clData->m_command_queue, max_distance, max_distance_h, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", ret);
#else
	float max_distance_h = _max_distance;
	ret = clEnqueueWriteBuffer(clData->m_command_queue, max_distance, CL_TRUE, 0, sizeof(float), &max_distance_h, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", ret);
	clFlush(clData->m_command_queue);
#endif

	/* Set OpenCL Kernel Parameters */
	cl_uint j = 0;
	ret = clSetKernelArg(truncateDepthKernel, j++, sizeof(cl_mem), (void*)depths_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(truncateDepthKernel, j++, sizeof(cl_mem), (void*)&max_distance);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	size_t globalWorkSize[2] = { (size_t)depths_curr_.cols(), (size_t)depths_curr_.rows() };
	const size_t localWorkSize[2] = { 32, 8 };
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, truncateDepthKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);

	ret = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	return;
}

void cl_bilateral_pyrdown::clpyrDown(const DepthMap& src, DepthMap& dst)
{
	cl_int ret;

	cl_uint j = 0;
	ret = clSetKernelArg(pyrDownKernel, j++, sizeof(cl_mem), (void*)src.handle());
	ret |= clSetKernelArg(pyrDownKernel, j++, sizeof(cl_mem), (void*)dst.handle());

	/* Execute OpenCL Kernel */
	size_t globalWorkSize[2] = { (size_t)dst.cols(), (size_t)dst.rows() };
	const size_t localWorkSize[2] = { 32, 8 };
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, pyrDownKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);

	ret = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	return;
}