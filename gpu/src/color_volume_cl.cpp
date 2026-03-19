#include <pcl/gpu/kinfu/color_volume_cl.h>
#ifdef __ANDROID__
#include "cl/colors.cl.h"
#endif

cl_color_volume::cl_color_volume(opencl_utils *_clData) {
	program = NULL;
	colorKernel = NULL;
	initcolorKernel = NULL;
	clData = _clData;
	colorVolumeClInit();
}

cl_color_volume::~cl_color_volume()
{
	clReleaseMemObject(cl_mem_extractColor_params);
	clReleaseKernel(colorKernel);
	clReleaseKernel(initcolorKernel);
	clReleaseKernel(extractColorsKernel);
	clReleaseProgram(program);
}

void cl_color_volume::colorVolumeClInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)colors_cl, sizeof(colors_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/colors.cl");
#endif
	colorKernel = clData->compileKernelFromFile(program, "updateColorVolumeKernel", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	initcolorKernel = clData->compileKernelFromFile(program, "initColorVolumeKernel", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	extractColorsKernel = clData->compileKernelFromFile(program, "extractColorsKernel", &err);
	CHK_ERR("compileKernelFromFile Failed", err);

#ifdef ALLOC_HOST_MEMORY
	cl_mem_extractColor_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(extractColorParams), NULL, &err);
#else
	cl_mem_extractColor_params = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(extractColorParams), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);

}

void cl_color_volume::updateColorVolume(const Intr& intr, float tranc_dist, const Mat33& Rcurr_inv, const float3& tcurr,
	const MapArr& vmap, const CLPtrStepSz<uchar3> colors, const float3& volume_size, CLDeviceArray2D<int> color_volume, int max_weight)
{
	cl_int err;
	colorParams params;
	/*settint params*/
	params.resolution[0] = VOLUME_X;
	params.resolution[1] = VOLUME_Y;
	params.resolution[2] = VOLUME_Z;
	params.volume[0] = volume_size.x;
	params.volume[1] = volume_size.y;
	params.volume[2] = volume_size.z;
	params.tranc_dist = tranc_dist;
	params.weight = max_weight < 0 ? 0 : (max_weight > 255 ? 255 : max_weight);
	params.width = colors.cols;
	params.height = colors.rows;
	params.intr = intr;

	Mat33 Rcurr_inv_cl = Rcurr_inv;
	
	cl_mem color_params_d_ = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(colorParams), &params, &err);
	CHK_ERR("clCreateBuffer Failed", err);
	cl_mem cam_pose_R_inv_d = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Mat33)* 3, &Rcurr_inv_cl, &err);
	CHK_ERR("clCreateBuffer Failed", err);
	cl_mem cam_pose_t_d = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float3), (void*)&tcurr, &err);
	CHK_ERR("clCreateBuffer Failed", err);

	cl_uint j = 0;
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), (void*)color_volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), (void*)&colors.handle); //depth map
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), vmap.handle()); //vertex map
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), (void*)&cam_pose_R_inv_d);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), (void*)&cam_pose_t_d);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(colorKernel, j++, sizeof(cl_mem), (void*)&color_params_d_);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)VOLUME_X, (size_t)VOLUME_Y };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, colorKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#if 0 //for test
	cl_short2* tsdf_h = new cl_short2[(int)(VOLUME_X*VOLUME_Y*VOLUME_Z)];
	clEnqueueReadBuffer(clData->m_command_queue, tsdf_volume_->map, CL_TRUE, 0, sizeof(cl_short2)*VOLUME_X*VOLUME_Y*VOLUME_Z, tsdf_h, 0, NULL, NULL);
	CHK_ERR("clEnqueueReadBuffer Failed", err);
	
	FILE *file_out = fopen("tsdf2.txt", "w");
	for (int i = VOLUME_X*VOLUME_Y*(VOLUME_Z / 4); i < VOLUME_X*VOLUME_Y*(VOLUME_Z / 2); i++)
	{
		fprintf(file_out, "i = %d | tsdf = %f, weight = %d \n", i, (float)(tsdf_h[i].s[0]) / (float)(32767), tsdf_h[i].s[1]);
	}
	fclose(file_out);
	delete[] tsdf_h;
#endif
	err = clReleaseMemObject(color_params_d_);
	CHK_ERR("clReleaseMemObject Failed", err);
	err = clReleaseMemObject(cam_pose_R_inv_d);
	CHK_ERR("clReleaseMemObject Failed", err);
	err = clReleaseMemObject(cam_pose_t_d);
	CHK_ERR("clReleaseMemObject Failed", err);
	return;
}

void cl_color_volume::initColorVolume(CLDeviceArray2D<int> color_volume, const int* volume_size)
{
	cl_int err;
	cl_mem hvolume_size = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)* 3, (void*)volume_size, &err);
	CHK_ERR("clCreateBuffer Failed", err);

	cl_uint j = 0;
	err = clSetKernelArg(initcolorKernel, j++, sizeof(cl_mem), (void*)color_volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(initcolorKernel, j++, sizeof(cl_mem), (void*)&hvolume_size);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)VOLUME_X, (size_t)VOLUME_Y };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, initcolorKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	err = clReleaseMemObject(hvolume_size);
	CHK_ERR("clReleaseMemObject Failed", err);
}
void cl_color_volume::extractColors(CLDeviceArray2D<int> color_volume, const float3& volume_size, const CLDeviceArray<float4>& points, CLDeviceArray<uchar4>& colors)
{
	cl_int err;
	float3 cell_size;
	cell_size.x = volume_size.x / VOLUME_X;
	cell_size.y = volume_size.y / VOLUME_Y;
	cell_size.z = volume_size.z / VOLUME_Z;
#ifdef ALLOC_HOST_MEMORY
	extractColorParams* pParams = (extractColorParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_extractColor_params, CL_TRUE, CL_MAP_WRITE, 0, sizeof(extractColorParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams->cell_size = cell_size;
	//pParams->cell_size.x = volume_size.x / VOLUME_X;
	//pParams->cell_size.y = volume_size.y / VOLUME_Y;
	//pParams->cell_size.z = volume_size.z / VOLUME_Z;
	pParams->VOLUME.x = VOLUME_X;
	pParams->VOLUME.y = VOLUME_Y;
	pParams->VOLUME.z = VOLUME_Z;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_extractColor_params, pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	extractColorParams pParams;
	pParams.cell_size = cell_size;
	//pParams.cell_size.x = volume_size.x / VOLUME_X;
	//pParams.cell_size.y = volume_size.y / VOLUME_Y;
	//pParams.cell_size.z = volume_size.z / VOLUME_Z;
	pParams.VOLUME.x = VOLUME_X;
	pParams.VOLUME.y = VOLUME_Y;
	pParams.VOLUME.z = VOLUME_Z;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_extractColor_params, CL_TRUE, 0, sizeof(extractColorParams), (void*)&pParams, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif

	cl_uint j = 0;
	err = clSetKernelArg(extractColorsKernel, j++, sizeof(cl_mem), (void*)color_volume.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractColorsKernel, j++, sizeof(cl_mem), (void*)points.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractColorsKernel, j++, sizeof(cl_mem), (void*)colors.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(extractColorsKernel, j++, sizeof(cl_mem), (void*)&cl_mem_extractColor_params);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[1] = { (size_t)points.size() };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, extractColorsKernel, 1, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
}
