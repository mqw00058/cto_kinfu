#include <pcl/gpu/kinfu/generateImage.h>
#ifdef __ANDROID__
#include "cl/generateImage.cl.h"
#endif

cl_generateImage::cl_generateImage(opencl_utils *_clData)
{
	program = NULL;
	GIKernel = NULL;
	GIKernel2 = NULL;
	paint3DViewKernel = NULL;
	clData = _clData;
	GIInit();
}

cl_generateImage::~cl_generateImage()
{
	clReleaseKernel(GIKernel);
	clReleaseKernel(GIKernel2);
	clReleaseKernel(paint3DViewKernel);
	clReleaseProgram(program);
}

void cl_generateImage::GIInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)generateImage_cl, sizeof(generateImage_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/generateImage.cl");
#endif
	GIKernel = clData->compileKernelFromFile(program, "GI", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	GIKernel2 = clData->compileKernelFromFile(program, "GI2", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	paint3DViewKernel = clData->compileKernelFromFile(program, "paint3DView", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
}

void cl_generateImage::generateImage(const MapArr& vmaps_g_prev_, const MapArr& nmaps_g_prev_, LightSource light, CLPtrStepSz<uchar3> dst) {
	cl_int err;
	int width = vmaps_g_prev_.width();
	int height = vmaps_g_prev_.height();

	/* Create Memory Buffer */
	lightParams params;
	params.pos[0] = light.pos[0].x;
	params.pos[1] = light.pos[0].y;
	params.pos[2] = light.pos[0].z;
	params.number = light.number;
	cl_mem light_params_d_ = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(lightParams), &params, &err);
	CHK_ERR("clCreateBuffer Failed", err);

	cl_uint j = 0;
	err = clSetKernelArg(GIKernel, j++, sizeof(cl_mem), vmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel, j++, sizeof(cl_mem), nmaps_g_prev_.handle()); //depth map
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel, j++, sizeof(cl_mem), (void*)&light_params_d_);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel, j++, sizeof(cl_mem), (void*)&dst.handle);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)width, (size_t)height };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, GIKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#if 0 //for test
	unsigned char* GI_h = new unsigned char[cols * rows];
	clEnqueueReadBuffer(clData->m_command_queue, dst, CL_TRUE, 0, sizeof(unsigned char)* cols * (rows), GI_h, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	FILE *file_out = fopen("GI2.txt", "w");
	for (int i = 0; i < cols; i++)
	for (int j = 0; j <(rows / 3); j++)
	{
		fprintf(file_out, "%d | x = %d, y = %d, z = %d\n", j * cols + i, GI_h[j * cols + i], GI_h[(j + rows / 3) * cols + i], GI_h[(j + (2 * rows / 3)) * cols + i] );
	}
	fclose(file_out);
#endif
	err = clReleaseMemObject(light_params_d_);
	CHK_ERR("clReleaseMemObject Failed", err);
}


#ifdef __ANDROID__
void cl_generateImage::generateImage(const MapArr& vmaps_g_prev_, const MapArr& nmaps_g_prev_, LightSource light, MapArr& dst) {
	cl_int err;
	int width = vmaps_g_prev_.width();
	int height = vmaps_g_prev_.height();

	/* Create Memory Buffer */
	lightParams params;
	params.pos[0] = light.pos[0].x;
	params.pos[1] = light.pos[0].y;
	params.pos[2] = light.pos[0].z;
	params.number = light.number;
	cl_mem light_params_d_ = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(lightParams), &params, &err);
	CHK_ERR("clCreateBuffer Failed", err);

	cl_uint j = 0;
	err = clSetKernelArg(GIKernel2, j++, sizeof(cl_mem), vmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel2, j++, sizeof(cl_mem), nmaps_g_prev_.handle()); //depth map
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel2, j++, sizeof(cl_mem), (void*)&light_params_d_);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(GIKernel2, j++, sizeof(cl_mem), dst.handle() );
	CHK_ERR("clSetKernelArg Failed", err);

	//glFinish();
	err = clEnqueueAcquireGLObjects(clData->m_command_queue, 1, dst.handle(), 0, NULL, NULL );

	size_t global_ws[2] = { (size_t)width, (size_t)height };
	err = clEnqueueNDRangeKernel(clData->m_command_queue, GIKernel2, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#if 0 //for test
	unsigned char* GI_h = new unsigned char[cols * rows];
	clEnqueueReadBuffer(clData->m_command_queue, dst, CL_TRUE, 0, sizeof(unsigned char)* cols * (rows), GI_h, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	FILE *file_out = fopen("GI2.txt", "w");
	for (int i = 0; i < cols; i++)
		for (int j = 0; j <(rows / 3); j++)
		{
			fprintf(file_out, "%d | x = %d, y = %d, z = %d\n", j * cols + i, GI_h[j * cols + i], GI_h[(j + rows / 3) * cols + i], GI_h[(j + (2 * rows / 3)) * cols + i] );
		}
	fclose(file_out);
#endif
	err = clReleaseMemObject(light_params_d_);
	CHK_ERR("clReleaseMemObject Failed", err);
}
#endif


void cl_generateImage::paint3DView(const CLPtrStep<uchar3> colors, CLPtrStepSz<uchar3> dst, float colors_weight){
	cl_int err;

	int cols = dst.cols;
	int rows = dst.rows;


	cl_uint j = 0;
	err = clSetKernelArg(paint3DViewKernel, j++, sizeof(cl_mem), (void*)&colors.handle);
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(paint3DViewKernel, j++, sizeof(cl_mem), (void*)&dst.handle);
	CHK_ERR("clSetKernelArg Failed", err);
	//err = clSetKernelArg(paint3DViewKernel, j++, sizeof(cl_mem), (void*)&paint_params_d_);
	//CHK_ERR("clSetKernelArg Failed", err);

	size_t global_ws[2] = { (size_t)cols, (size_t)rows};
	err = clEnqueueNDRangeKernel(clData->m_command_queue, paint3DViewKernel, 2, 0, global_ws, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);

	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#if 0 //for test
	unsigned char* GI_h = new unsigned char[cols * rows];
	clEnqueueReadBuffer(clData->m_command_queue, dst, CL_TRUE, 0, sizeof(unsigned char)* cols * (rows), GI_h, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	FILE *file_out = fopen("GI2.txt", "w");
	for (int i = 0; i < cols; i++)
	for (int j = 0; j <(rows / 3); j++)
	{
		fprintf(file_out, "%d | x = %d, y = %d, z = %d\n", j * cols + i, GI_h[j * cols + i], GI_h[(j + rows / 3) * cols + i], GI_h[(j + (2 * rows / 3)) * cols + i]);
	}
	fclose(file_out);
#endif
}

