#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include "internal.h"
#include <pcl/gpu/kinfu/connexe.h>
#ifdef __ANDROID__
#include "cl/marchingcube.cl.h"
#endif

using namespace pcl;
using namespace pcl::gpu;
using pcl::device::device_cast;

#define CL_IMAGE3D_UNSUPPORTED

pcl::gpu::MarchingCubes::MarchingCubes() :
verticesBuffer(opencl_utils::get()->m_context),
isolevel(51)
{
	clData = opencl_utils::get();
	init();
}

template <class T>
inline std::string to_string(const T& t) {
	std::stringstream ss;
	ss << t;
	return ss.str();
}

void pcl::gpu::MarchingCubes::init()
{
	cl_int ret;
	std::string buildOptions = "-DSIZE=" + to_string(VOLUME_X);
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)marchingcube_cl, sizeof(marchingcube_cl), buildOptions.c_str());
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/marchingcube.cl", buildOptions.c_str());
#endif
	constructHPLevelKernel = clData->compileKernelFromFile(program, "constructHPLevel", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
#ifdef CL_IMAGE3D_UNSUPPORTED
	classifyCubesKernel = clData->compileKernelFromFile(program, "classifyCubesFromBuffer", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
	traverseHPKernel = clData->compileKernelFromFile(program, "traverseHPFromBuffer", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
#else
	classifyCubesKernel = clData->compileKernelFromFile(program, "classifyCubes", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
	traverseHPKernel = clData->compileKernelFromFile(program, "traverseHP", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
#endif

	constructHPLevelCharCharKernel = clData->compileKernelFromFile(program, "constructHPLevelCharChar", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
	constructHPLevelCharShortKernel = clData->compileKernelFromFile(program, "constructHPLevelCharShort", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
	constructHPLevelShortShortKernel = clData->compileKernelFromFile(program, "constructHPLevelShortShort", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);
	constructHPLevelShortIntKernel = clData->compileKernelFromFile(program, "constructHPLevelShortInt", &ret);
	CHK_ERR("compileKernelFromFile Failed", ret);

	cl_mem_flags flag;
#ifdef ALLOC_HOST_MEMORY
	flag = CL_MEM_ALLOC_HOST_PTR;
#else
	flag = 0;
#endif
	cl_mem buffer;
	int bufferSize = VOLUME_X*VOLUME_Y*VOLUME_Z;
	buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(char)*bufferSize, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
	buffers.push_back(buffer);
	bufferSize /= 8;
	buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(char)*bufferSize, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
	buffers.push_back(buffer);
	bufferSize /= 8;
	buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(short)*bufferSize, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
	buffers.push_back(buffer);
	bufferSize /= 8;
	buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(short)*bufferSize, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
	buffers.push_back(buffer);
	bufferSize /= 8;
	buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(short)*bufferSize, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
	buffers.push_back(buffer);
	bufferSize /= 8;
	for (int i = 5; i < log2((float)VOLUME_X); i++) {
		buffer = clCreateBuffer(clData->m_context, CL_MEM_READ_WRITE | flag, sizeof(int)*bufferSize, NULL, &ret);
		CHK_ERR("clCreateBuffer Failed", ret);
		buffers.push_back(buffer);
		bufferSize /= 8;
	}

#ifdef CL_IMAGE3D_UNSUPPORTED
	cubeIndexesImage = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | flag, sizeof(unsigned char)*VOLUME_X*VOLUME_Y*VOLUME_Z, NULL, &ret);
	CHK_ERR("clCreateImage3D Failed", ret);

	volumeDataImage = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | flag, sizeof(unsigned char)*VOLUME_X*VOLUME_Y*VOLUME_Z, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
#else
	cl_image_format format;
	format.image_channel_order = CL_R;
	format.image_channel_data_type = CL_UNSIGNED_INT8;
#ifdef CL_VERSION_1_2
	cl_image_desc desc;
	desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	desc.image_width = VOLUME_X;
	desc.image_height = VOLUME_Y;
	desc.image_depth = VOLUME_Z;
	desc.image_array_size = 1;
	desc.image_row_pitch = NULL;
	desc.image_slice_pitch = NULL;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;

	cubeIndexesImage = clCreateImage(clData->m_context, CL_MEM_READ_ONLY | flag, &format, &desc, NULL, &ret);
	CHK_ERR("clCreateImage Failed", ret);

	volumeDataImage = clCreateImage(clData->m_context, CL_MEM_READ_ONLY | flag, &format, &desc, NULL, &ret);
	CHK_ERR("clCreateImage Failed", ret);
#else
	cubeIndexesImage = clCreateImage3D(clData->m_context, CL_MEM_READ_ONLY | flag, &format, VOLUME_X, VOLUME_Y, VOLUME_Z, 0, 0, NULL, &ret);
	CHK_ERR("clCreateImage3D Failed", ret);

	volumeDataImage = clCreateImage3D(clData->m_context, CL_MEM_READ_ONLY | flag, &format, VOLUME_X, VOLUME_Y, VOLUME_Z, 0, 0, NULL, &ret);
	CHK_ERR("clCreateImage3D Failed", ret);
#endif
#endif
}

pcl::gpu::MarchingCubes::~MarchingCubes()
{
	while (buffers.empty() == false)
	{
		clReleaseMemObject(buffers.back());
		buffers.pop_back();
	}
	clReleaseMemObject(cubeIndexesImage);
	clReleaseMemObject(volumeDataImage);
	clReleaseKernel(constructHPLevelKernel);
	clReleaseKernel(classifyCubesKernel);
	clReleaseKernel(traverseHPKernel);
	clReleaseKernel(constructHPLevelCharCharKernel);
	clReleaseKernel(constructHPLevelCharShortKernel);
	clReleaseKernel(constructHPLevelShortShortKernel);
	clReleaseKernel(constructHPLevelShortIntKernel);
	clReleaseProgram(program);
}

void pcl::gpu::MarchingCubes::histoPyramidConstruction() {
	cl_int ret;
	updateScalarField();

	ret = clSetKernelArg(constructHPLevelCharCharKernel, 0, sizeof(cl_mem), &buffers[0]);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(constructHPLevelCharCharKernel, 1, sizeof(cl_mem), &buffers[1]);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	size_t globalWorkSize[3] = { VOLUME_X / 2, VOLUME_Y / 2, VOLUME_Z / 2 };
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelCharCharKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	int previous = VOLUME_X / 2;

	ret = clSetKernelArg(constructHPLevelCharShortKernel, 0, sizeof(cl_mem), &buffers[1]);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(constructHPLevelCharShortKernel, 1, sizeof(cl_mem), &buffers[2]);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	globalWorkSize[0] = previous / 2;
	globalWorkSize[1] = previous / 2;
	globalWorkSize[2] = previous / 2;
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelCharShortKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	previous /= 2;

	ret = clSetKernelArg(constructHPLevelShortShortKernel, 0, sizeof(cl_mem), &buffers[2]);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(constructHPLevelShortShortKernel, 1, sizeof(cl_mem), &buffers[3]);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	globalWorkSize[0] = previous / 2;
	globalWorkSize[1] = previous / 2;
	globalWorkSize[2] = previous / 2;
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelShortShortKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	previous /= 2;

	ret = clSetKernelArg(constructHPLevelShortShortKernel, 0, sizeof(cl_mem), &buffers[3]);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(constructHPLevelShortShortKernel, 1, sizeof(cl_mem), &buffers[4]);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	globalWorkSize[0] = previous / 2;
	globalWorkSize[1] = previous / 2;
	globalWorkSize[2] = previous / 2;
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelShortShortKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	previous /= 2;

	ret = clSetKernelArg(constructHPLevelShortIntKernel, 0, sizeof(cl_mem), &buffers[4]);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(constructHPLevelShortIntKernel, 1, sizeof(cl_mem), &buffers[5]);
	CHK_ERR("clSetKernelArg Failed", ret);

	/* Execute OpenCL Kernel */
	globalWorkSize[0] = previous / 2;
	globalWorkSize[1] = previous / 2;
	globalWorkSize[2] = previous / 2;
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelShortIntKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);

	previous /= 2;

	// Run level 2 to top level
	for (int i = 5; i < log2((float)VOLUME_X) - 1; i++) {
		ret = clSetKernelArg(constructHPLevelKernel, 0, sizeof(cl_mem), &buffers[i]);
		CHK_ERR("clSetKernelArg Failed", ret);
		ret = clSetKernelArg(constructHPLevelKernel, 1, sizeof(cl_mem), &buffers[i + 1]);
		CHK_ERR("clSetKernelArg Failed", ret);

		previous /= 2;

		/* Execute OpenCL Kernel */
		globalWorkSize[0] = previous;
		globalWorkSize[1] = previous;
		globalWorkSize[2] = previous;
		ret = clEnqueueNDRangeKernel(clData->m_command_queue, constructHPLevelKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
		CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
		clFlush(clData->m_command_queue);
		CHK_ERR("clFlush Failed", ret);
	}
}

void pcl::gpu::MarchingCubes::updateScalarField() {
	cl_int ret;
	cl_mem_flags flag;
#ifdef ALLOC_HOST_MEMORY
	flag = CL_MEM_ALLOC_HOST_PTR;
#else
	flag = 0;
#endif
#ifdef CL_IMAGE3D_UNSUPPORTED
#else
	cubeIndexesBuffer = clCreateBuffer(clData->m_context, CL_MEM_WRITE_ONLY | flag, sizeof(char)*VOLUME_X * VOLUME_Y * VOLUME_Z, NULL, &ret);
	CHK_ERR("clCreateBuffer Failed", ret);
#endif
	ret = clSetKernelArg(classifyCubesKernel, 0, sizeof(cl_mem), &buffers[0]);
	CHK_ERR("clSetKernelArg Failed", ret);
#ifdef CL_IMAGE3D_UNSUPPORTED
	ret = clSetKernelArg(classifyCubesKernel, 1, sizeof(cl_mem), &cubeIndexesImage);
	CHK_ERR("clSetKernelArg Failed", ret);
#else
	ret = clSetKernelArg(classifyCubesKernel, 1, sizeof(cl_mem), &cubeIndexesBuffer);
	CHK_ERR("clSetKernelArg Failed", ret);
#endif
	ret = clSetKernelArg(classifyCubesKernel, 2, sizeof(cl_mem), &volumeDataImage);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(classifyCubesKernel, 3, sizeof(int), &isolevel);
	CHK_ERR("clSetKernelArg Failed", ret);

	size_t globalWorkSize[3] = { VOLUME_X, VOLUME_Y, VOLUME_Z };
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, classifyCubesKernel, 3, 0, globalWorkSize, NULL, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);
#ifdef CL_IMAGE3D_UNSUPPORTED
#else
	size_t offset[3] = { 0, 0, 0 };
	size_t region[3] = { VOLUME_X, VOLUME_Y, VOLUME_Z };

	ret = clEnqueueCopyBufferToImage(clData->m_command_queue, cubeIndexesBuffer, cubeIndexesImage, 0, offset, region, 0, NULL, NULL);
	CHK_ERR("clEnqueueCopyBufferToImage Failed", ret);
	clFinish(clData->m_command_queue);
	CHK_ERR("clFinish Failed", ret);
	clReleaseMemObject(cubeIndexesBuffer);
#endif
}

void pcl::gpu::MarchingCubes::histoPyramidTraversal(int sum) {
	// Make OpenCL buffer from OpenGL buffer
	unsigned int i = 0;
	cl_int ret;
	ret = clSetKernelArg(traverseHPKernel, 0, sizeof(cl_mem), &volumeDataImage);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(traverseHPKernel, 1, sizeof(cl_mem), &cubeIndexesImage);
	CHK_ERR("clSetKernelArg Failed", ret);
	for (i = 0; i < buffers.size(); i++) {
		ret = clSetKernelArg(traverseHPKernel, i + 2, sizeof(cl_mem), &buffers[i]);
		CHK_ERR("clSetKernelArg Failed", ret);
	}
	i += 2;

	ret = clSetKernelArg(traverseHPKernel, i, sizeof(cl_mem), verticesBuffer.handle());
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(traverseHPKernel, i + 1, sizeof(int), &isolevel);
	CHK_ERR("clSetKernelArg Failed", ret);
	ret = clSetKernelArg(traverseHPKernel, i + 2, sizeof(int), &sum);
	CHK_ERR("clSetKernelArg Failed", ret);

	// Increase the global_work_size so that it is divideable by 64
	size_t global_work_size = sum + 64 - (sum - 64 * (sum / 64));
	size_t local_work_size = 64;
	// Run a NDRange kernel over this buffer which traverses back to the base level
	ret = clEnqueueNDRangeKernel(clData->m_command_queue, traverseHPKernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", ret);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", ret);
}

void pcl::gpu::MarchingCubes::convertVolume(const TsdfVolume& tsdf_volume)
{
	cl_int ret;
#ifdef ALLOC_HOST_MEMORY
	cl_short2* pshort2 = (cl_short2*)clEnqueueMapBuffer(clData->m_command_queue, *tsdf_volume.data().handle(), CL_TRUE, CL_MAP_READ, 0, sizeof(cl_short2)*VOLUME_X*VOLUME_Y*VOLUME_Z, 0, NULL, NULL, &ret);
	CHK_ERR("clEnqueueMapBuffer Failed", ret);
#ifdef CL_IMAGE3D_UNSUPPORTED
	unsigned char* puchar = (unsigned char*)clEnqueueMapBuffer(clData->m_command_queue, volumeDataImage, CL_TRUE, CL_MAP_WRITE, 0, sizeof(unsigned char)*VOLUME_X*VOLUME_Y*VOLUME_Z, 0, NULL, NULL, &ret);
	CHK_ERR("clEnqueueMapBuffer Failed", ret);
#else
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {VOLUME_X, VOLUME_Y, VOLUME_Z};
	size_t row_pitch = VOLUME_X;
	size_t slice_pitch = VOLUME_X*VOLUME_Y;
	unsigned char* puchar = (unsigned char*)clEnqueueMapImage(clData->m_command_queue, volumeDataImage, CL_TRUE, CL_MAP_WRITE, origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &ret);
	__android_log_print(ANDROID_LOG_ERROR, "Kinfu", "convertVolume clEnqueueMapImage End ptr:%d row_picth:%d slice_pitch:%d\n", puchar, row_pitch, slice_pitch);
	CHK_ERR("clEnqueueMapImage Failed", ret);
#endif

#ifdef NO_POST_DENOISING
	for (int i = 0; i < tsdf_volume.data().rows(); i++)
	{
		for (int j = 0; j < tsdf_volume.data().cols(); j++)
		{
			int index = i*tsdf_volume.data().cols() + j;
#ifdef __ANDROID__
			//this if is for making single-sided mesh, not double sided, sidee ffect is hole-filling
			if (pshort2[index].s[0] == 0 && pshort2[index].s[1] == 0)
				pshort2[index].s[0] = pshort2[index-1].s[0];
#endif
			if (pshort2[index].s[0] < 0)
				puchar[index] = -pshort2[index].s[0] / 128;
			else
				puchar[index] = 0;// -pshort2[index].s[0] / 128;
		}
	}
#endif
	//for (int x = 0; x < VOLUME_X; x++)
	//{
	//	for (int y = 0; y < VOLUME_Y; y++)
	//	{
	//		int pos = (y * VOLUME_X + x) ;
	//		int elem_step = VOLUME_Y * VOLUME_X;
	//		int pos_prev = (y * VOLUME_X + x) ;
	//		for (int z = 0; z < VOLUME_Z; z++, pos += elem_step)
	//		{
	//			int index = z * VOLUME_X *VOLUME_Y + y * VOLUME_Y + x;
	//			if (z == 0)
	//			{
	//				puchar[index] = 0;
	//			}
	//			else if (pshort2[pos].s[0] < 0 && pshort2[pos_prev].s[0] > 0 )

	//			{
	//				puchar[index] = -pshort2[index].s[0] / pshort2[index].s[1];
	//				pos_prev = pos;
	//			}
	//			else
	//				puchar[index] = 0;
	//		}
	//	}
	//}

#ifdef CONNECTED_COMPONENTS_DENOISING_3D
	int bufferDims[3] = { tsdf_volume.data().cols(), tsdf_volume.data().cols(), tsdf_volume.data().cols() };
	unsigned char *bufferIn = (unsigned char*)malloc(bufferDims[0] * bufferDims[1] * bufferDims[2] * sizeof(unsigned char));
	unsigned char *bufferOut = (unsigned char*)malloc(bufferDims[0] * bufferDims[1] * bufferDims[2] * sizeof(unsigned char));
	
	Connexe_SetMinimumSizeOfComponents(1000);
	Connexe_SetMaximumNumberOfComponents(-1);
	Connexe_SetConnectivity(26);

	for (int i = 0; i < tsdf_volume.data().rows(); i++)
	{
		for (int j = 0; j < tsdf_volume.data().cols(); j++)
		{
			int index = i*tsdf_volume.data().cols() + j;
			int ii = i%tsdf_volume.data().cols();
			int jj = i/tsdf_volume.data().cols();
			int kk = j; 
			int index2 = ii + jj*tsdf_volume.data().cols() + kk*tsdf_volume.data().cols()*tsdf_volume.data().cols();
			if (pshort2[index].s[0] < 0 ) 
			{
				bufferIn[index2] = 1;
			}
			else 
			{
				bufferIn[index2] = 0;
			}
		}
	}
	if (CountConnectedComponents(bufferIn, UCHAR, bufferOut, UCHAR, bufferDims) > 0) 
	{
		for (int i = 0; i < tsdf_volume.data().rows(); i++)
		{
			for (int j = 0; j < tsdf_volume.data().cols(); j++)
			{
				int index = i*tsdf_volume.data().cols() + j;
				int ii = i%tsdf_volume.data().cols();
				int jj = i/tsdf_volume.data().cols();
				int kk = j; 
				int index2 = ii + jj*tsdf_volume.data().cols() + kk*tsdf_volume.data().cols()*tsdf_volume.data().cols();
				if (pshort2[index].s[0] < 0 ) {
					puchar[index] = -pshort2[index].s[0] / 128;
				}
				else {
					puchar[index] = 0;
				}

				if( bufferOut[index2] == 0 )
				{
					puchar[index] = 0;
				}
			}
		}
	}

	free(bufferIn);
	free(bufferOut);
#endif

#ifdef CONNECTED_COMPONENTS_DENOISING_2D
	unsigned char **bitmap = NULL;
	int **labelmap = NULL;
	int* labelmapcounter = NULL;

	bitmap = (unsigned char**)malloc(tsdf_volume.data().rows() * sizeof(unsigned char*));
	labelmap = (int**)malloc(tsdf_volume.data().rows() * sizeof(int*));
	for (int y = 0; y < tsdf_volume.data().rows(); y++) 
	{
		bitmap[y] = (unsigned char*)malloc(tsdf_volume.data().cols() * sizeof(unsigned char));
		labelmap[y] = (int*)malloc(tsdf_volume.data().cols() * sizeof(int));
	}

	for (int i = 0; i < tsdf_volume.data().rows(); i++)
	{
		for (int j = 0; j < tsdf_volume.data().cols(); j++)
		{
			int index = i*tsdf_volume.data().cols() + j;
			if (pshort2[index].s[0] < 0)
				bitmap[i][j] = 1;
			else
				bitmap[i][j] = 0;
			labelmap[i][j] = 0;
		}
	}

	int labelsize = ConnectedComponentLabeling(tsdf_volume, bitmap, labelmap, &labelmapcounter);
	if( labelsize > 0 ) 
	{
		for (int i = 0; i < tsdf_volume.data().rows(); i++)
		{
			for (int j = 0; j < tsdf_volume.data().cols(); j++)
			{
				int index = i*tsdf_volume.data().cols() + j;
				if (pshort2[index].s[0] < 0) 
				{
					puchar[index] = -pshort2[index].s[0] / 128;
				}
				else {
					puchar[index] = 0;
				}

				if(labelmap[i][j] > 0 && labelmap[i][j] < labelsize) 
				{
					if(labelmapcounter[labelmap[i][j]] < 50) 
					{
						puchar[index] = 0;
					}
				}
			}
		}
	}

	for (int idx = 0; idx < tsdf_volume.data().rows(); idx++)
	{
		free(bitmap[idx]);
		free(labelmap[idx]);
	}
	free(bitmap);
	free(labelmap);
	free(labelmapcounter);
#endif

	ret = clEnqueueUnmapMemObject(clData->m_command_queue, volumeDataImage, puchar, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", ret);
	ret = clEnqueueUnmapMemObject(clData->m_command_queue, *tsdf_volume.data().handle(), pshort2, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", ret);
#else
	cl_short2* pshort2 = new cl_short2[tsdf_volume.data().rows()*tsdf_volume.data().cols()];
	unsigned char* puchar = new unsigned char[tsdf_volume.data().rows()*tsdf_volume.data().cols()];
	tsdf_volume.data().download(clData->m_command_queue, pshort2, 0);
	for (int i = 0; i < tsdf_volume.data().rows(); i++)
	{
		for (int j = 0; j < tsdf_volume.data().cols(); j++)
		{
			int index = i*tsdf_volume.data().cols() + j;
#ifdef __ANDROID__
			//this if is for making single-sided mesh, not double sided, sidee ffect is hole-filling
			if (pshort2[index].s[0] == 0 && pshort2[index].s[1] == 0)
				pshort2[index].s[0] = pshort2[index-1].s[0];
#endif
			if (pshort2[index].s[0] < 0)
				puchar[index] = -pshort2[index].s[0] / 128;
			else
				puchar[index] = 0;// -pshort2[index].s[0] / 128;
		}
	}
#ifdef CL_IMAGE3D_UNSUPPORTED
	ret = clEnqueueWriteBuffer(clData->m_command_queue, volumeDataImage, CL_TRUE, 0, VOLUME_X*VOLUME_X*VOLUME_Y, puchar, 0, NULL, NULL);
#else
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { VOLUME_X, VOLUME_Y, VOLUME_Z };
	ret = clEnqueueWriteImage(clData->m_command_queue, volumeDataImage, CL_TRUE, origin, region, VOLUME_X, VOLUME_X*VOLUME_Y, puchar, 0, NULL, NULL);
	CHK_ERR("clEnqueueMapBuffer Failed", ret);
#endif
	delete[] puchar;
	delete[] pshort2;
#endif

	//char filename[256];
	//sprintf(filename, "tsdf1130_1.txt");
	//FILE *file_out = fopen(filename, "w");
	//cl_short2* pushort = new cl_short2[tsdf_volume.data().rows()*tsdf_volume.data().cols()];
	//int row = tsdf_volume.data().rows();
	//int col = tsdf_volume.data().cols();
	//tsdf_volume.data().download(clData->m_command_queue, pushort, 0);
	//for (int i = 0; i < tsdf_volume.data().rows(); i++)
	//{
	//	for (int j = 0; j < tsdf_volume.data().cols(); j++)
	//	{
	//		int index = i*tsdf_volume.data().cols() + j;
	//		//if (pushort[index].s[0] != 0)
	//		//if ( (j == (tsdf_volume.data().cols() / 2)) && ((i % tsdf_volume.data().cols()) == (tsdf_volume.data().cols() / 2)) )
	//		if ((j == 128) && ((i % tsdf_volume.data().cols()) == 128))
	//		{
	//			fprintf(file_out, "i = %d | tsdf = %f, weight = %d, org_tsdf = %d \n", index, (float)(pushort[index].s[0]) / (float)(32767), pushort[index].s[1], pushort[index].s[0]);
	//		}
	//	}
	//}
	//fclose(file_out);
	//delete[] pushort;
}

CLDeviceArray<cl_float>
pcl::gpu::MarchingCubes::run(const TsdfVolume& tsdf_volume, CLDeviceArray<PointType>& triangles_buffer)
{
	cl_int ret;

	convertVolume(tsdf_volume);
	histoPyramidConstruction();

	// Read top of histoPyramid an use this size to allocate VBO below
#ifdef ALLOC_HOST_MEMORY
	int * sum = (int*)clEnqueueMapBuffer(clData->m_command_queue, buffers[buffers.size() - 1], CL_TRUE, CL_MAP_READ, 0, sizeof(int)*8, 0, NULL, NULL, &ret);
	CHK_ERR("clEnqueueMapBuffer Failed", ret);
#else
	int * sum = new int[8];
	ret = clEnqueueReadBuffer(clData->m_command_queue, buffers[buffers.size() - 1], CL_TRUE, 0, sizeof(int)* 8, sum, 0, NULL, NULL);
	CHK_ERR("clEnqueueReadBuffer Failed", ret);
#endif
	int totalSum;
	totalSum = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#ifdef __ANDROID__
	__android_log_print(ANDROID_LOG_ERROR, "Kinfu", "MarchingCubes num triangles: %d\n", totalSum);
#else
	std::cout << "MarchingCubes num triangles: " << totalSum << std::endl;
#endif
#ifdef ALLOC_HOST_MEMORY
	ret = clEnqueueUnmapMemObject(clData->m_command_queue, buffers[buffers.size() - 1], sum, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", ret);
#endif
	if (totalSum == 0) {
		std::cout << "No triangles were extracted. Check isovalue." << std::endl;
		return 0;
	}
	verticesBuffer.create(totalSum * 18);

	// Traverse the histoPyramid
	histoPyramidTraversal(totalSum);
	clFinish(clData->m_command_queue);
	CHK_ERR("clFinish Failed", ret);

	return CLDeviceArray<cl_float>(verticesBuffer);
}




#ifdef CONNECTED_COMPONENTS_DENOISING_2D
void pcl::gpu::MarchingCubes::Tracer(int *cy, int *cx, int *tracingdirection, unsigned char** bitmap, int ** labelmap)
{
	int SearchDirection[8][2] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
	int i, y, x;

	for (i = 0; i < 7; i++)
	{
		y = *cy + SearchDirection[*tracingdirection][0];
		x = *cx + SearchDirection[*tracingdirection][1];

		if (bitmap[y][x] == 0)
		{
			labelmap[y][x] = -1;
			*tracingdirection = (*tracingdirection + 1) % 8;
		}
		else
		{
			*cy = y;
			*cx = x;
			break;
		}
	}
}

void pcl::gpu::MarchingCubes::ContourTracing(int cy, int cx, int labelindex, int tracingdirection, unsigned char** bitmap, int ** labelmap)
{
	int SearchDirection[8][2] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 1, -1 }, { 0, -1 }, { -1, -1 }, { -1, 0 }, { -1, 1 } };
	char tracingstopflag = 0, SearchAgain = 1;
	int fx, fy, sx = cx, sy = cy;

	Tracer(&cy, &cx, &tracingdirection, bitmap,labelmap);

	if (cx != sx || cy != sy)
	{
		fx = cx;
		fy = cy;

		while (SearchAgain)
		{
			tracingdirection = (tracingdirection + 6) % 8;
			labelmap[cy][cx] = labelindex;
			Tracer(&cy, &cx, &tracingdirection, bitmap, labelmap);

			if (cx == sx && cy == sy)
			{
				tracingstopflag = 1;
			}
			else if (tracingstopflag)
			{
				if (cx == fx && cy == fy)
				{
					SearchAgain = 0;
				}
				else
				{
					tracingstopflag = 0;
				}
			}
		}
	}
}

int pcl::gpu::MarchingCubes::ConnectedComponentLabeling(const TsdfVolume& tsdf_volume, unsigned char** bitmap, int** labelmap, int** labelmapcounter)
{
	int height, width, cx, cy, tracingdirection, ConnectedComponentsCount = 0, labelindex = 0;

	height = tsdf_volume.data().rows();
	width = tsdf_volume.data().cols();

	for (cy = 1; cy < height - 1; cy++)
	{
		for (cx = 1, labelindex = 0; cx < width - 1; cx++)
		{
			if (bitmap[cy][cx] == 1)// black pixel
			{
				if (labelindex != 0)// use pre-pixel label
				{
					labelmap[cy][cx] = labelindex;
				}
				else
				{
					labelindex = labelmap[cy][cx];

					if (labelindex == 0)
					{
						labelindex = ++ConnectedComponentsCount;
						tracingdirection = 0;
						ContourTracing(cy, cx, labelindex, tracingdirection, bitmap, labelmap);// external contour
						labelmap[cy][cx] = labelindex;
					}
				}
			}
			else if (labelindex != 0)// white pixel & pre-pixel has been labeled
			{
				if (labelmap[cy][cx] == 0)
				{
					tracingdirection = 1;
					ContourTracing(cy, cx - 1, labelindex, tracingdirection, bitmap, labelmap);// internal contour
				}

				labelindex = 0;
			}
		}
	}

	int labelsize = -1;
	for (cy = 1; cy < height - 1; cy++)
	{
		for (cx = 1; cx < width - 1; cx++)
		{
			if (labelsize <= labelmap[cy][cx])
				labelsize = labelmap[cy][cx];
		}
	}
	if (labelsize > 0) {
		(*labelmapcounter) = (int*)malloc(labelsize * sizeof(int));
	}
	for (cx = 0; cx < labelsize; cx++) {
		(*labelmapcounter)[cx] = 0;
	}
	for (cy = 1; cy < height - 1; cy++)
	{
		for (cx = 1; cx < width - 1; cx++)
		{
			if (labelmap[cy][cx] > 0) {
				(*labelmapcounter)[labelmap[cy][cx]]++;
			}
		}
	}

	return labelsize;
}
#endif