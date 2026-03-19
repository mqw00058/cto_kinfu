#include <pcl/gpu/kinfu/ray_caster.cl.h>
#ifdef __ANDROID__
#include "cl/ray_caster.cl.h"
#endif

cl_ray_caster::cl_ray_caster(opencl_utils *_clData)
{
	program = NULL;
	rayCastKernel = NULL;
	clData = _clData;
	init();
}

cl_ray_caster::~cl_ray_caster()
{
	clReleaseMemObject(cl_mem_params);
	clReleaseKernel(rayCastKernel);
	clReleaseProgram(program);
}

void cl_ray_caster::init()
{
	opencl_utils* clData = opencl_utils::get();

	cl_int err;
	char* option = "-DUSE_MULTIQUEUE";
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)ray_caster_cl, sizeof(ray_caster_cl), option);
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/ray_caster.cl", option);
#endif
	rayCastKernel = clData->compileKernelFromFile(program, "rayCastKernel", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(RayCasterParams_CL), NULL, &err);
#else
	cl_mem_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(RayCasterParams_CL), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
}

void cl_ray_caster::raycast(const pcl::device::Intr& intr, const pcl::device::Mat33& Rcurr, const float3& tcurr,
	float tranc_dist, const float3& volume_size,
	const CLDeviceArray2D<int>& volume, MapArr& vmap, MapArr& nmap)
{
	int width = vmap.width();
	int height = vmap.height();

	opencl_utils* clData = opencl_utils::get();
	cl_int err = CL_SUCCESS;
#ifdef ALLOC_HOST_MEMORY
	RayCasterParams_CL* pParams = (RayCasterParams_CL*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_params, CL_TRUE, CL_MAP_WRITE, 0, sizeof(RayCasterParams_CL), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->Rcurr = Rcurr;
	pParams->tcurr = tcurr;

	pParams->time_step = tranc_dist * 0.8f;

	pParams->volume_size = volume_size;

	pParams->cell_size.x = volume_size.x / pcl::device::VOLUME_X;
	pParams->cell_size.y = volume_size.y / pcl::device::VOLUME_Y;
	pParams->cell_size.z = volume_size.z / pcl::device::VOLUME_Z;

	pParams->intr = intr;

	pParams->VOLUME_X = pcl::device::VOLUME_X;
	pParams->VOLUME_Y = pcl::device::VOLUME_Y;
	pParams->VOLUME_Z = pcl::device::VOLUME_Z;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_params, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	RayCasterParams_CL rc;
	rc.Rcurr = Rcurr;
	rc.tcurr = tcurr;

	rc.time_step = tranc_dist * 0.8f;

	rc.volume_size = volume_size;

	rc.cell_size.x = volume_size.x / pcl::device::VOLUME_X;
	rc.cell_size.y = volume_size.y / pcl::device::VOLUME_Y;
	rc.cell_size.z = volume_size.z / pcl::device::VOLUME_Z;

	rc.intr = intr;

	rc.VOLUME_X = pcl::device::VOLUME_X;
	rc.VOLUME_Y = pcl::device::VOLUME_Y;
	rc.VOLUME_Z = pcl::device::VOLUME_Z;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_params, CL_TRUE, 0, sizeof(RayCasterParams_CL), &rc, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif

	err = clSetKernelArg(rayCastKernel, 0, sizeof(cl_mem), (void *)volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(rayCastKernel, 1, sizeof(cl_mem), vmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(rayCastKernel, 2, sizeof(cl_mem), nmap.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(rayCastKernel, 3, sizeof(cl_mem), (void *)&cl_mem_params);
	CHK_ERR("clSetKernelArg Failed", err);
	
	dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
	dim3 grid(pcl::device::divUp(width, block.x), pcl::device::divUp(height, block.y));

	/* Execute OpenCL Kernel */
#ifdef KINFU_USE_MULTIQUEUE
	size_t globalWorkSize[2] = { (size_t)grid.x*block.x/2, (size_t)grid.y*block.y/2 };
	size_t localWorkSize[2] = { (size_t)block.x, (size_t)block.y };
	size_t globalWorkOffset[2] = { 0, 0 };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, rayCastKernel, 2, globalWorkOffset, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	globalWorkOffset[0] = globalWorkSize[0];
	globalWorkOffset[1] = 0;
	err = clEnqueueNDRangeKernel(clData->m_command_queue_sub[0], rayCastKernel, 2, globalWorkOffset, globalWorkSize, NULL, 0, NULL, &clData->m_event_wait_list[0]);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue_sub[0]);
	CHK_ERR("clFlush Failed", err);

	globalWorkOffset[0] = 0;
	globalWorkOffset[1] = globalWorkSize[1];
	err = clEnqueueNDRangeKernel(clData->m_command_queue_sub[1], rayCastKernel, 2, globalWorkOffset, globalWorkSize, NULL, 0, NULL, &clData->m_event_wait_list[1]);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue_sub[1]);
	CHK_ERR("clFlush Failed", err);

	globalWorkOffset[0] = globalWorkSize[0];
	globalWorkOffset[1] = globalWorkSize[1];
	err = clEnqueueNDRangeKernel(clData->m_command_queue_sub[2], rayCastKernel, 2, globalWorkOffset, globalWorkSize, NULL, 0, NULL, &clData->m_event_wait_list[2]);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue_sub[2]);
	CHK_ERR("clFlush Failed", err);
#else
	size_t globalWorkSize[2] = { (size_t)grid.x*block.x, (size_t)grid.y*block.y };
	size_t localWorkSize[2] = { (size_t)block.x, (size_t)block.y };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, rayCastKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
}
