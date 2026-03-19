#include <pcl/gpu/kinfu/maps.h>
#include <assert.h>
#ifdef __ANDROID__
#include "cl/maps.cl.h"
#endif
#ifdef CL_GL_INTEROP //For CL GL Interoperation
#include <GLES3/gl3.h>
#include <CL/cl_gl.h>
#endif

cl_maps::cl_maps(opencl_utils *_clData) {
	program = NULL;
	computeVmapKernel = NULL;
	computeNmapKernel = NULL;
	transformVMapsKernel = NULL;
	transformNMapsKernel = NULL; 
	resizeVMapKernel = NULL;
	resizeNMapKernel = NULL;
	resizeDepthMapKernel = NULL;
	resizeDepthMapKernel2 = NULL;
	convertMapKernel = NULL;
	convertMap2Kernel = NULL;
	mergePointNormalKernel = NULL;
	clData = _clData;
	mapsInit();
}

cl_maps::~cl_maps()
{
	clReleaseMemObject(cl_mem_Intr_inv);
	clReleaseMemObject(cl_mem_params_cltranformMap);
	clReleaseKernel(computeVmapKernel);
	clReleaseKernel(computeNmapKernel);
	clReleaseKernel(transformVMapsKernel);
	clReleaseKernel(transformNMapsKernel);
	clReleaseKernel(resizeVMapKernel);
	clReleaseKernel(resizeNMapKernel);
	clReleaseKernel(resizeDepthMapKernel);
	clReleaseKernel(resizeDepthMapKernel2);
	clReleaseKernel(convertMapKernel);
	clReleaseKernel(convertMap2Kernel);
	clReleaseKernel(mergePointNormalKernel);
	clReleaseProgram(program);
}

void cl_maps::mapsInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)maps_cl, sizeof(maps_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/maps.cl");
#endif
	computeVmapKernel = clData->compileKernelFromFile(program, "computeVmap", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	computeNmapKernel = clData->compileKernelFromFile(program, "computeNmap", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	transformVMapsKernel = clData->compileKernelFromFile(program, "transformVMap", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	transformNMapsKernel = clData->compileKernelFromFile(program, "transformNMap", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	resizeVMapKernel = clData->compileKernelFromFile(program, "resizeVMaps", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	resizeNMapKernel = clData->compileKernelFromFile(program, "resizeNMaps", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	resizeDepthMapKernel = clData->compileKernelFromFile(program, "resizeDepthMaps", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	resizeDepthMapKernel2 = clData->compileKernelFromFile(program, "resizeDepthMaps2", &err);
	CHK_ERR("compileKernelFromFile2 Failed", err);
convertMapKernel = clData->compileKernelFromFile(program, "convertMap", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	convertMap2Kernel = clData->compileKernelFromFile(program, "convertMap2", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	mergePointNormalKernel = clData->compileKernelFromFile(program, "mergePointNormal", &err);
	CHK_ERR("compileKernelFromFile Failed", err);

#ifdef ALLOC_HOST_MEMORY
	cl_mem_params_cltranformMap = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(TranformMapParams), NULL, &err);
#else
	cl_mem_params_cltranformMap = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(TranformMapParams), NULL, &err);
#endif
#ifdef ALLOC_HOST_MEMORY
	cl_mem_Intr_inv = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(Intr), NULL, &err);
#else
	cl_mem_Intr_inv = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(Intr), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);

}

void cl_maps::clCreateVMap(const pcl::device::Intr& intr, const DepthMap& src, MapArr& dst)
{
	cl_int err;
	int cols = src.cols();
	int rows = src.rows();
	Intr intr_inv = intr(-1);

#ifdef ALLOC_HOST_MEMORY
	Intr* intr_inv_params = (Intr*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_Intr_inv, CL_TRUE, CL_MAP_WRITE, 0, sizeof(Intr), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	* intr_inv_params  = intr_inv;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_Intr_inv, intr_inv_params, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_Intr_inv, CL_TRUE, 0, sizeof(Intr), &intr_inv, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	/* Set OpenCL Kernel Parameters */
	cl_uint j = 0;
	err = clSetKernelArg(computeVmapKernel, j++, sizeof(cl_mem), src.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(computeVmapKernel, j++, sizeof(cl_mem), dst.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(computeVmapKernel, j++, sizeof(cl_mem), (void*)&cl_mem_Intr_inv);
	CHK_ERR("clSetKernelArg Failed", err);

	/* Execute OpenCL Kernel */
	size_t globalWorkSize[2] = { (size_t)cols, (size_t)rows };
	const size_t localWorkSize[2] = { 32, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, computeVmapKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	/* Finalization */
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_maps::clCreateNMap(const MapArr& vmap, MapArr& nmap)
{
	cl_int err;
	int width = vmap.width();
	int height = vmap.height();

	/* Set OpenCL Kernel Parameters */
	cl_uint j = 0;
	err = clSetKernelArg(computeNmapKernel, j++, sizeof(cl_mem), vmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(computeNmapKernel, j++, sizeof(cl_mem), nmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	/* Execute OpenCL Kernel */
	size_t globalWorkSize[2] = { (size_t)width, (size_t)height };
	const size_t localWorkSize[2] = { 32, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, computeNmapKernel, 2, 0, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	/* Finalization */
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_maps::cltranformMap(const MapArr& vmaps_curr_, const MapArr& nmaps_curr_, const Mat33& Rmat, const float3& tvec, MapArr& vmaps_g_prev_, MapArr& nmaps_g_prev_)
{
	cl_int err;
	int width = vmaps_curr_.width();
	int height = vmaps_curr_.height();


#ifdef ALLOC_HOST_MEMORY
	TranformMapParams* pParams = (TranformMapParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_params_cltranformMap, CL_TRUE, CL_MAP_WRITE, 0, sizeof(TranformMapParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->Rmat = Rmat;
	pParams->tvec = tvec;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_params_cltranformMap, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	TranformMapParams params;
	params.Rmat = Rmat;
	params.tvec = tvec;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_params_cltranformMap, CL_TRUE, 0, sizeof(TranformMapParams), &params, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	cl_uint j = 0;
	err = clSetKernelArg(transformVMapsKernel, j++, sizeof(cl_mem), vmaps_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(transformVMapsKernel, j++, sizeof(cl_mem), vmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(transformVMapsKernel, j++, sizeof(cl_mem), (void*)&cl_mem_params_cltranformMap);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)width, (size_t)height };
	const size_t local_ws[2] = { 32, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, transformVMapsKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	///////////////////////////////////////////////////////////////////////////////////
	j = 0;
	err = clSetKernelArg(transformNMapsKernel, j++, sizeof(cl_mem), nmaps_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(transformNMapsKernel, j++, sizeof(cl_mem), nmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(transformNMapsKernel, j++, sizeof(cl_mem), (void*)&cl_mem_params_cltranformMap);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws2[2] = { (size_t)width, (size_t)height };
	const size_t local_ws2[2] = { 32, 8 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, transformNMapsKernel, 2, 0, global_ws2, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	///////////////////////////////////////////////////////////////////////////////////
	return;
}

void cl_maps::clresizeVMap(const MapArr& input, MapArr& output)
{
	cl_int err;
	int out_width = output.width();
	int out_height = output.height();

	cl_uint j = 0;
	err = clSetKernelArg(resizeVMapKernel, j++, sizeof(cl_mem), input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(resizeVMapKernel, j++, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)out_width, (size_t)out_height };
#ifdef KINFU_USE_MULTIQUEUE
	if (clData->m_event_wait == true)
	{
		err = clEnqueueNDRangeKernel(clData->m_command_queue, resizeVMapKernel, 2, 0, global_ws, NULL, 3, clData->m_event_wait_list, NULL);
		clData->m_event_wait = false;
	}
	else
#endif
		err = clEnqueueNDRangeKernel(clData->m_command_queue, resizeVMapKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_maps::clresizeNMap(const MapArr& input, MapArr& output)
{
	cl_int err;
	int out_width = output.width();
	int out_height = output.height();

	cl_uint j = 0;
	err = clSetKernelArg(resizeNMapKernel, j++, sizeof(cl_mem), input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(resizeNMapKernel, j++, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)out_width, (size_t)out_height };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, resizeNMapKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	return;
}

void cl_maps::clresizeDepthMap(const DepthMap& input, DepthMap& output)
{
	cl_int err;
	int out_width = output.cols();
	int out_height = output.rows();

	cl_uint j = 0;
	err = clSetKernelArg(resizeDepthMapKernel, j++, sizeof(cl_mem), input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(resizeDepthMapKernel, j++, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)out_width, (size_t)out_height };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, resizeDepthMapKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_maps::clresizeDepthMap2(const MapArr& input, DepthMap& output)
{
	cl_int err;
	int out_width = output.cols();
	int out_height = output.rows();

#ifdef CL_GL_INTEROP
	glFinish();
	err = clEnqueueAcquireGLObjects(clData->m_command_queue, 1, input.handle(), 0, NULL, NULL);
#endif
	cl_uint j = 0;
	err = clSetKernelArg(resizeDepthMapKernel2, j++, sizeof(cl_mem), input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(resizeDepthMapKernel2, j++, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)out_width, (size_t)out_height };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, resizeDepthMapKernel2, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

#ifdef CL_GL_INTEROP
	clFinish(clData->m_command_queue);
	clEnqueueReleaseGLObjects(clData->m_command_queue, 1, input.handle(), 0, NULL, NULL);
#endif
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	return;
}

void cl_maps::convert(const MapArr& vmap, CLDeviceArray2D<float4>& output)
{
	cl_int err;

	int w = vmap.width();
	int h = vmap.height();

	err = clSetKernelArg(convertMapKernel, 0, sizeof(cl_mem), vmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(convertMapKernel, 1, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)w, (size_t)h };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, convertMapKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	return;
}

void cl_maps::convert(const MapArr& vmap, CLDeviceArray2D<float8>& output)
{
	cl_int err;

	int w = vmap.width();
	int h = vmap.height();

	err = clSetKernelArg(convertMap2Kernel, 0, sizeof(cl_mem), vmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(convertMap2Kernel, 1, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)w, (size_t)h };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, convertMap2Kernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	return;
}

void cl_maps::mergePointNormal(const CLDeviceArray<float4>& cloud, const CLDeviceArray<float8>& normals, const CLDeviceArray<float12>& output)
//void cl_maps::mergePointNormal(const CLDeviceArray<PointXYZ>& cloud, const CLDeviceArray<Normal>& normals, CLDeviceArray<PointNormal>& output)
{
	cl_int err;
	const int block = 256;
	size_t total = output.size();

	err = clSetKernelArg(mergePointNormalKernel, 0, sizeof(cl_mem), cloud.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(mergePointNormalKernel, 1, sizeof(cl_mem), normals.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(mergePointNormalKernel, 2, sizeof(cl_mem), output.handle());
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws = total;
	err = clEnqueueNDRangeKernel(clData->m_command_queue, mergePointNormalKernel, 1, 0, &global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	return;
}
