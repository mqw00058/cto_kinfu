#include <pcl/gpu/kinfu/estimate_combined.h>
#include <vector>
#ifdef __ANDROID__
#include "cl/estimate_combined.cl.h"
#include <android/log.h>
#endif

cl_estimate_combined::cl_estimate_combined(opencl_utils *_clData) {
	program = NULL;
	combinedKernel = NULL;
	TransformKernel = NULL;
	clData = _clData;
	estimateCombinedInit();
}

cl_estimate_combined::~cl_estimate_combined() {
	clReleaseMemObject(cl_mem_params_TranformReduction);
	clReleaseMemObject(params_estimated_combined);
	clReleaseKernel(combinedKernel);
	clReleaseKernel(TransformKernel);
	clReleaseProgram(program);
}

void cl_estimate_combined::estimateCombinedInit()
{
	cl_int err;
#ifdef __ANDROID__
	program = clData->buildProgram((const char*)estimate_combined_cl, sizeof(estimate_combined_cl));
#else
	program = clData->buildProgramFromFile("../../../../../gpu/kinfu_opencl/src/cl/estimate_combined.cl");
#endif
	combinedKernel = clData->compileKernelFromFile(program, "combinedKernel", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
	TransformKernel = clData->compileKernelFromFile(program, "TransformEstimatorKernel2", &err);
	CHK_ERR("compileKernelFromFile Failed", err);
#ifdef ALLOC_HOST_MEMORY
	params_estimated_combined = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(CombinedParams), NULL, &err);
#else
	params_estimated_combined = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(CombinedParams), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
#ifdef ALLOC_HOST_MEMORY
	cl_mem_params_TranformReduction = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(TranformReductionParams), NULL, &err);
#else
	cl_mem_params_TranformReduction = clCreateBuffer(clData->m_context, CL_MEM_READ_ONLY, sizeof(TranformReductionParams), NULL, &err);
#endif
	CHK_ERR("clCreateBuffer Failed", err);
}


void
cl_estimate_combined::estimateCombined(const Mat33& Rcurr, const float3& tcurr,
MapArr& vmaps_curr_, MapArr& nmaps_curr_,
const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
MapArr& vmaps_g_prev_, MapArr& nmaps_g_prev_,
float distThres, float angleThres,
BufArr& gbuf_, BufArr& mbuf_,
float_type* matrixA_host, float_type* vectorB_host)
{

	int width = vmaps_curr_.width();
	int height = vmaps_curr_.height();


	dim3 block(Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y / 2);
	dim3 grid(1, 1, 1);
	grid.x = divUp(width, block.x);
	grid.y = divUp(height, block.y);
	int gbuf_step = grid.x * grid.y;
	cl_int err;
	{
		//pcl::ScopeTime time("estimateCombined params");
#ifdef ALLOC_HOST_MEMORY
		CombinedParams* pParams = (CombinedParams*)clEnqueueMapBuffer(clData->m_command_queue, params_estimated_combined, CL_TRUE, CL_MAP_WRITE, 0, sizeof(CombinedParams), 0, NULL, NULL, &err);
		CHK_ERR("clEnqueueMapBuffer Failed", err);
		pParams->distThres = distThres;
		pParams->angleThres = angleThres;
		pParams->intr = intr;
		pParams->Rcurr = Rcurr;
		pParams->tcurr = tcurr;
		pParams->Rprev_inv = Rprev_inv;
		pParams->tprev = tprev;
		err = clEnqueueUnmapMemObject(clData->m_command_queue, params_estimated_combined, pParams, 0, NULL, NULL);
		CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
		CombinedParams pParams;
		pParams.distThres = distThres;
		pParams.angleThres = angleThres;
		pParams.intr = intr;
		pParams.Rcurr = Rcurr;
		pParams.tcurr = tcurr;
		pParams.Rprev_inv = Rprev_inv;
		pParams.tprev = tprev;
		
		err = clEnqueueWriteBuffer(clData->m_command_queue, params_estimated_combined, CL_TRUE, 0, sizeof(CombinedParams), &pParams, 0, NULL, NULL);
		CHK_ERR("clEnqueueWriteBuffer Failed", err);
		clFlush(clData->m_command_queue);
#endif
	}
	// Set the kernel arguments
	cl_uint j = 0;
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), gbuf_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), nmaps_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), vmaps_curr_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), nmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), vmaps_g_prev_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(combinedKernel, j++, sizeof(cl_mem), (void*)&params_estimated_combined);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t globalWorkSize[2] = { (size_t)grid.x*block.x, (size_t)grid.y*block.y };
	size_t localWorkSize[2] = { (size_t)block.x, (size_t)block.y };

	// Queue the kernel up for execution across the array
	err = clEnqueueNDRangeKernel(clData->m_command_queue, combinedKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);


#ifdef ALLOC_HOST_MEMORY
	TranformReductionParams* pParams2 = (TranformReductionParams*)clEnqueueMapBuffer(clData->m_command_queue, cl_mem_params_TranformReduction, CL_TRUE, CL_MAP_WRITE, 0, sizeof(TranformReductionParams), 0, NULL, NULL, &err);
	CHK_ERR("clEnqueueMapBuffer Failed", err);
	pParams2->step = grid.x * grid.y;
	pParams2->length = grid.x * grid.y;
	err = clEnqueueUnmapMemObject(clData->m_command_queue, cl_mem_params_TranformReduction, pParams2, 0, NULL, NULL);
	CHK_ERR("clEnqueueUnmapMemObject Failed", err);
#else
	TranformReductionParams params2;
	params2.step = grid.x * grid.y;
	params2.length = grid.x * grid.y;
	err = clEnqueueWriteBuffer(clData->m_command_queue, cl_mem_params_TranformReduction, CL_TRUE, 0, sizeof(TranformReductionParams), &params2, 0, NULL, NULL);
	CHK_ERR("clEnqueueWriteBuffer Failed", err);
	clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);
#endif
	j = 0;
	err = clSetKernelArg(TransformKernel, j++, sizeof(cl_mem), gbuf_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(TransformKernel, j++, sizeof(cl_mem), mbuf_.handle());
	CHK_ERR("clSetKernelArg Failed", err);
	err = clSetKernelArg(TransformKernel, j++, sizeof(cl_mem), &cl_mem_params_TranformReduction);
	CHK_ERR("clSetKernelArg Failed", err);

	size_t globalWorkSize2[1] = { TranformReduction::TOTAL * TranformReduction::CTA_SIZE / 4 };
	size_t localWorkSize2[1] = { TranformReduction::CTA_SIZE / 4 };

	// Queue the kernel up for execution across the array
	err = clEnqueueNDRangeKernel(clData->m_command_queue, TransformKernel, 1, NULL, globalWorkSize2, localWorkSize2, 0, NULL, NULL);
	CHK_ERR("clEnqueueNDRangeKernel Failed", err);
	err = clFlush(clData->m_command_queue);
	CHK_ERR("clFlush Failed", err);

	std::vector<float_type> v_mbuf;
	int v_mbuf_size = TranformReduction::TOTAL;
	v_mbuf.resize(v_mbuf_size);
	err = clEnqueueReadBuffer(clData->m_command_queue, *(mbuf_.handle()), CL_TRUE, 0, sizeof(float_type)*v_mbuf.size(), (void*)v_mbuf.data(), 0, NULL, NULL);
	CHK_ERR("clEnqueueReadBuffer Failed", err);

	int shift = 0;
	for (int i = 0; i < 6; ++i)  //rows
	for (int j = i; j < 7; ++j)    // cols + b
	{
		float_type value = v_mbuf.data()[shift++];
		if (j == 6)       // vector b
			vectorB_host[i] = value;
		else
			matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
	}
}
