#include <pcl/gpu/kinfu/extract.h>
#ifdef __ANDROID__
#include "cl/extract.cl.h"
#endif

cl_extract::cl_extract(opencl_utils *_clData) {
	program = NULL;
	extractKernel = NULL;
	extractNormalsKernel = NULL;
	extractNormals2Kernel = NULL;
	clData = _clData;
	extractInit();
}

cl_extract::~cl_extract()
{
	clReleaseMemObject(cl_mem_cell_size);
	clReleaseMemObject(cl_mem_extract_params);
	clReleaseKernel(extractKernel);
	clReleaseKernel(extractNormalsKernel);
	clReleaseKernel(extractNormals2Kernel);
	clReleaseProgram(program);
}

void cl_extract::extractInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)extract_cl, sizeof(extract_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/extract.cl");
#endif
	extractKernel = clData->compileKernelFromFile(program, "extractCloud", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	extractNormalsKernel = clData->compileKernelFromFile(program, "extractNormals", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	extractNormals2Kernel = clData->compileKernelFromFile(program, "extractNormals2", &err);
	CHK_ERR("compileKernelFromFile Failed", err);

#ifdef ALLOC_HOST_MEMORY
	cl_mem_cell_size = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float3), NULL, &err);
#else
	cl_mem_cell_size = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(float3), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_extract_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(ExtractParams), NULL, &err);
#else
	cl_mem_extract_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(ExtractParams), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
}

void cl_extract::extractNormals(const CLDeviceArray2D<int> volume, const float3 volume_size, const CLDeviceArray<float4>& input, CLDeviceArray<float4>& output)
{
	cl_int err;
	float3 cell_size;
	cell_size.x = volume_size.x / VOLUME_X;
	cell_size.y = volume_size.y / VOLUME_Y;
	cell_size.z = volume_size.z / VOLUME_Z;
#ifdef ALLOC_HOST_MEMORY
	int* pParams = (int*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_cell_size, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float3), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams[0] = volume_size.x / VOLUME_X;
	pParams[1] = volume_size.y / VOLUME_Y;
	pParams[2] = volume_size.z / VOLUME_Z;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_cell_size, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_cell_size, CL_TRUE, 0, sizeof(float3), (void*)&cell_size, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	cl_uint j = 0;
	err = clSetKernelArg(extractNormalsKernel, j++, sizeof(cl_mem), (void*)volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormalsKernel, j++, sizeof(cl_mem), (void*)input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormalsKernel, j++, sizeof(cl_mem), (void*)output.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormalsKernel, j++, sizeof(cl_mem), (void*)&cl_mem_cell_size);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[1] = { (size_t)input.size() };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, extractNormalsKernel, 1, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
}
void cl_extract::extractNormals2(const CLDeviceArray2D<int> volume, const float3 volume_size, const CLDeviceArray<float4>& input, CLDeviceArray<float8>& output)
{
	cl_int err;
	float3 cell_size;
	cell_size.x = volume_size.x / VOLUME_X;
	cell_size.y = volume_size.y / VOLUME_Y;
	cell_size.z = volume_size.z / VOLUME_Z;
#ifdef ALLOC_HOST_MEMORY
	ExtractParams* pParams = (ExtractParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_extract_params, CL_TRUE, CL_MAP_WRITE, 0, sizeof(ExtractParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->cell_size = cell_size;
	//pParams->cell_size.x = volume_size.x / VOLUME_X;
	//pParams->cell_size.y = volume_size.y / VOLUME_Y;
	//pParams->cell_size.z = volume_size.z / VOLUME_Z;
	pParams->VOLUME.x = VOLUME_X;
	pParams->VOLUME.y = VOLUME_Y;
	pParams->VOLUME.z = VOLUME_Z;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_extract_params, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	ExtractParams pParams;
	pParams.cell_size = cell_size;
	//pParams.cell_size.x = volume_size.x / VOLUME_X;
	//pParams.cell_size.y = volume_size.y / VOLUME_Y;
	//pParams.cell_size.z = volume_size.z / VOLUME_Z;
	pParams.VOLUME.x = VOLUME_X;
	pParams.VOLUME.y = VOLUME_Y;
	pParams.VOLUME.z = VOLUME_Z;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_extract_params, CL_TRUE, 0, sizeof(ExtractParams), (void*)&pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif

	cl_uint j = 0;
	err = clSetKernelArg(extractNormals2Kernel, j++, sizeof(cl_mem), (void*)volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormals2Kernel, j++, sizeof(cl_mem), (void*)input.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormals2Kernel, j++, sizeof(cl_mem), (void*)output.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractNormals2Kernel, j++, sizeof(cl_mem), (void*)&cl_mem_extract_params);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[1] = { (size_t)input.size() };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, extractNormals2Kernel, 1, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
}
size_t  cl_extract::extractCloud(const CLDeviceArray2D<int> volume, const float3 volume_size, CLDeviceArray<float4>& output)
{
	cl_int err;
	float3 cell_size;
	cell_size.x = volume_size.x / VOLUME_X;
	cell_size.y = volume_size.y / VOLUME_Y;
	cell_size.z = volume_size.z / VOLUME_Z;
#ifdef ALLOC_HOST_MEMORY
	ExtractParams* pParams = (ExtractParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_extract_params, CL_TRUE, CL_MAP_WRITE, 0, sizeof(ExtractParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->cell_size = cell_size;
	//pParams->cell_size.x = volume_size.x / VOLUME_X;
	//pParams->cell_size.y = volume_size.y / VOLUME_Y;
	//pParams->cell_size.z = volume_size.z / VOLUME_Z;
	pParams->VOLUME.x = VOLUME_X;
	pParams->VOLUME.y = VOLUME_Y;
	pParams->VOLUME.z = VOLUME_Z;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_extract_params, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	ExtractParams pParams;
	pParams.cell_size = cell_size;
	//pParams.cell_size.x = volume_size.x / VOLUME_X;
	//pParams.cell_size.y = volume_size.y / VOLUME_Y;
	//pParams.cell_size.z = volume_size.z / VOLUME_Z;
	pParams.VOLUME.x = VOLUME_X;
	pParams.VOLUME.y = VOLUME_Y;
	pParams.VOLUME.z = VOLUME_Z;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_extract_params, CL_TRUE, 0, sizeof(ExtractParams), (void*)&pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif

	size_t kernel_group_size;
	clGetKernelWorkGroupInfo(extractKernel, clData->m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void*)&kernel_group_size, NULL);

	cl_mem global_count = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	CHK_ERR("clCreateBuffer Failed", err);
	cl_mem output_count = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	CHK_ERR("clCreateBuffer Failed", err);
	cl_mem blocks_done = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	CHK_ERR("clCreateBuffer Failed", err);
	int count = 0;
	err = clEnqueueWriteBuffer(clData->m_command_queue, global_count, CL_TRUE, 0, sizeof(int), (void*)&count, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(clData->m_command_queue, output_count, CL_TRUE, 0, sizeof(int), (void*)&count, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(clData->m_command_queue, blocks_done, CL_TRUE, 0, sizeof(unsigned int), (void*)&count, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	size_t localSize = CTA_SIZE *  sizeof(cl_int);
	size_t localSize2 = CTA_SIZE * MAX_LOCAL_POINTS * sizeof(cl_float);
	cl_uint j = 0;
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)output.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)&global_count);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)&output_count);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)&blocks_done);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, localSize, NULL);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, localSize2, NULL);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, localSize2, NULL);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, localSize2, NULL);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractKernel, j++, sizeof(cl_mem), (void*)&cl_mem_extract_params);
	CHK_ERR("clSetKernelArg Failed", err);
	dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
	dim3 grid(divUp(VOLUME_X, block.x), divUp(VOLUME_Y, block.y));
	//size_t global_ws[2] = { (size_t)VOLUME_X, (size_t)VOLUME_Y };
	
	size_t global_ws[2] = { (size_t)grid.x*block.x, (size_t)grid.y*block.y }; //256.258
	size_t local_ws[2] = { (size_t)block.x, (size_t)block.y }; //32.6
	err = clEnqueueNDRangeKernel(clData->m_command_queue, extractKernel, 2, 0, global_ws, local_ws, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
	int result;
	err = clEnqueueReadBuffer(clData->m_command_queue, output_count, CL_TRUE, 0, sizeof(int), (void*)&result, 0, NULL, NULL);
	CHK_ERR("clEnqueueReadBuffer Failed", err);
	clFinish(clData->m_command_queue);
	CHK_ERR("clFinish Failed", err);
	CL_RELEASE_MEM(global_count);
	CL_RELEASE_MEM(output_count);
	CL_RELEASE_MEM(blocks_done);
	return (size_t)result;
}