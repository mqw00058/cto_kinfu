/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <algorithm>

#include <pcl/gpu/kinfu/time.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#endif
#define USE_OPENCL_DEBUG_FRAME_COUNT 5
#ifdef __ANDROID__
const char* OPENCL_DEBUG_PATH_PREFIX = "/sdcard/Download/";
#else
const char* OPENCL_DEBUG_PATH_PREFIX = "./Debug/";
#endif

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;
int time_ms_ = 0;
cl_generateImage *KinfuTracker::clGenerateImage;
cl_maps *KinfuTracker::clMaps;
namespace pcl
{
	namespace gpu
	{
		PCL_EXPORTS Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker(
#ifdef CL_GL_INTEROP
	GLuint* vmaps_g_curr_tex_, GLuint* nmaps_g_curr_tex_, GLuint* vmaps_g_prev_tex_, GLuint* nmaps_g_prev_tex_, GLuint* vmaps_curr_tex_, GLuint* nmaps_curr_tex_,
#endif
	Config config)
	: clData(opencl_utils::get(config.getCLDevice())), depthRawScaled_(opencl_utils::get(config.getCLDevice())->m_context),
	rows_(config.getRows()), cols_(config.getCols()), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), disable_icp_(false), min_delta_(config.getMinDelta()), start_level_(0), doreset(false)

{
	levels_ = config.getLevels();
#ifdef __ANDROID__
	LOGD("levels_", levels_);
#endif
	icp_iterations_ = new int[levels_];

	clBilateralPyrdown = new cl_bilateral_pyrdown(clData);
	clMaps = new cl_maps(clData);
	clTsdf = new cl_tsdf_volume(clData);
	clEstimateCombined = new cl_estimate_combined(clData);
	clGenerateImage = new cl_generateImage(clData);
	clRayCaster = new cl_ray_caster(clData);
	clExtract = new cl_extract(clData);
	clColorVolume = new cl_color_volume(clData);
	

	const Vector3f volume_size = Vector3f::Constant(VOLUME_SIZE);
	const Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);

	tsdf_volume_ = TsdfVolume::Ptr(new TsdfVolume(volume_resolution, clTsdf, clExtract));
	tsdf_volume_->setSize(volume_size);

	setFocalLength(config);
	setDepthIntrinsics(focalLen_depth_x_, focalLen_depth_y_); // default values, can be overwritten

	init_Rcam_ = Eigen::Matrix3f::Identity();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
	init_tcam_ = volume_size * 0.5f - Vector3f(0, 0, volume_size(2) / 2 * 1.2f);
	
	int* iters = config.getIters();
	std::copy(iters, iters + levels_, icp_iterations_);
	icp_iterations_MAX_ = /*icp_iterations_[0] +*/ icp_iterations_[1] /*+ icp_iterations_[2]*/;
	const float default_distThres = 0.10f; //meters
	const float default_angleThres = sin(20.f * 3.14159254f / 180.f);
	const float default_tranc_dist = 0.03f; //meters

	setIcpCorespFilteringParams(default_distThres, default_angleThres);
	tsdf_volume_->setTsdfTruncDist(default_tranc_dist);

#ifdef CL_GL_INTEROP
	vmaps_g_prev_tex = new GLuint[levels_];	
	nmaps_g_prev_tex = new GLuint[levels_];
	setGLTextureIDs(vmaps_g_curr_tex_, nmaps_g_curr_tex_, vmaps_g_prev_tex_, nmaps_g_prev_tex_, vmaps_curr_tex_, nmaps_curr_tex_);
#endif
	allocateBufffers(rows_, cols_);
	rmats_.reserve(30000);
	tvecs_.reserve(30000);
	R_gyro_.reserve(4);
	reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthIntrinsics(float fx, float fy, float cx, float cy)
{
	fx_ = fx;
	fy_ = fy;
	cx_ = (cx == -1) ? cols_ / 2 - 0.5f : cx;
	cy_ = (cy == -1) ? rows_ / 2 - 0.5f : cy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getDepthIntrinsics(float& fx, float& fy, float& cx, float& cy)
{
	fx = fx_;
	fy = fy_;
	cx = cx_;
	cy = cy_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setInitalCameraPose(const Eigen::Affine3f& pose)
{
	init_Rcam_ = pose.rotation();
	init_tcam_ = pose.translation();
	reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthTruncationForICP(float max_icp_distance)
{
	max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
	integration_metric_threshold_ = threshold;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setFocalLength(Config config)
{
	FocalLen f = config.getFocalLength();
	focalLen_rgb_x_ = f.rgbX;
	focalLen_rgb_y_ = f.rgbY;
	focalLen_depth_x_ = f.depthX;
	focalLen_depth_y_ = f.depthY;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setIcpCorespFilteringParams(float distThreshold, float sineOfAngle)
{
	distThres_ = distThreshold; //mm
	angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::cols()
{
	return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::rows()
{
	return (rows_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::reset()
{
	if (global_time_)
	{
		cout << "Reset" << endl;
#ifdef __ANDROID__
		LOGD("RESET");
#endif
	}
	doreset = false;
	global_time_ = 0;
	rmats_.clear();
	tvecs_.clear();

	rmats_.push_back(init_Rcam_);
	tvecs_.push_back(init_tcam_);
	cout << init_Rcam_ << endl;
	cout << init_tcam_ << endl;
	R_gyro_.clear();
	R_gyro_.insert(R_gyro_.begin(), Vector3f(0.f, 0.f, 0.f));
	R_gyro_.insert(R_gyro_.begin(), Vector3f(0.f, 0.f, 0.f));
	R_gyro_.insert(R_gyro_.begin(), Vector3f(0.f, 0.f, 0.f));
	R_gyro_prev_ = Eigen::Matrix3f::Identity();
	tsdf_volume_->reset();

	if (color_volume_) // color integration mode is enabled
		color_volume_->reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CL_GL_INTEROP
void
pcl::gpu::KinfuTracker::setGLTextureIDs(GLuint* vmaps_g_curr_tex_, 
GLuint* nmaps_g_curr_tex_, 
GLuint* vmaps_g_prev_tex_, 
GLuint* nmaps_g_prev_tex_, 
GLuint* vmaps_curr_tex_, 
GLuint* nmaps_curr_tex_ ) 
{
	for (int i = 0; i < levels_; ++i)
	{
		//vmaps_g_curr_tex[i] = vmaps_g_curr_tex_[i];
		//nmaps_g_curr_tex[i] = nmaps_g_curr_tex_[i];
		vmaps_g_prev_tex[i] = vmaps_g_prev_tex_[i];
		nmaps_g_prev_tex[i] = nmaps_g_prev_tex_[i];
		//vmaps_curr_tex[i] = vmaps_curr_tex_[i];
		//nmaps_curr_tex[i] = nmaps_curr_tex_[i];
	}
}
#endif

void
pcl::gpu::KinfuTracker::allocateBufffers(int rows, int cols)
{
	depths_curr_.resize(levels_, DepthMap(opencl_utils::get()->m_context));

	vmaps_g_curr_.resize(levels_, MapArr(opencl_utils::get()->m_context));
	nmaps_g_curr_.resize(levels_, MapArr(opencl_utils::get()->m_context));

	vmaps_g_prev_.resize(levels_, MapArr(opencl_utils::get()->m_context));
	nmaps_g_prev_.resize(levels_, MapArr(opencl_utils::get()->m_context));

	vmaps_curr_.resize(levels_, MapArr(opencl_utils::get()->m_context));
	nmaps_curr_.resize(levels_, MapArr(opencl_utils::get()->m_context));

	coresps_.resize(levels_, CorespMap(opencl_utils::get()->m_context));

	gbuf_.resize(levels_, BufArr(opencl_utils::get()->m_context));
	sumbuf_.resize(levels_, BufArr(opencl_utils::get()->m_context));

	for (int i = 0; i < levels_; ++i)
	{
		int pyr_rows = rows >> i;
		int pyr_cols = cols >> i;
		int pyr_width = pyr_cols;
		int pyr_height = pyr_rows;
		depths_curr_[i].create(pyr_rows, pyr_cols);

		cl_image_format imageDepthFormat;
		imageDepthFormat.image_channel_data_type = CL_UNSIGNED_INT16;
		imageDepthFormat.image_channel_order = CL_R;
		cl_image_format imageMapFormat;
		imageMapFormat.image_channel_data_type = CL_FLOAT;
		imageMapFormat.image_channel_order = CL_RGBA;
#ifndef CL_GL_INTEROP
		vmaps_g_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);
		nmaps_g_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);

		vmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat);
		nmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat);

		vmaps_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);
		nmaps_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);
#else
		vmaps_g_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);//,vmaps_g_curr_tex[i]);
		nmaps_g_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);//,nmaps_g_curr_tex[i]);
		if( i == 0 ) {
			bool ret;
			ret = vmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat, vmaps_g_prev_tex[i]);			
			nmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat, nmaps_g_prev_tex[i]);
			
		} else {
			vmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat);
			nmaps_g_prev_[i].create(pyr_width, pyr_height, &imageMapFormat);
		}

		vmaps_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);//,vmaps_curr_tex[i]);
		nmaps_curr_[i].create(pyr_width, pyr_height, &imageMapFormat);//,nmaps_curr_tex[i]);
#endif
		dim3 block(Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y / 2);
		dim3 grid(1, 1, 1);
		grid.x = divUp(pyr_cols, block.x);
		grid.y = divUp(pyr_rows, block.y);
		gbuf_[i].create(grid.y, grid.x*TranformReduction::TOTAL);
		sumbuf_[i].create(TranformReduction::TOTAL, 1);
		coresps_[i].create(pyr_rows, pyr_cols);
	}
	depthRawScaled_.create(rows, cols);

	//// see estimate tranform for the magic numbers
	//gbuf_.create(27, 20 * 60);
	//sumbuf_.create(27);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void
pcl::gpu::KinfuTracker::convertTransforms(Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in_1, Vector3f& translation_in_2, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out_1, float3& translation_out_2)
{
	rotation_out_1 = device_cast<Mat33> (rotation_in_1);
	rotation_out_2 = device_cast<Mat33> (rotation_in_2);
	translation_out_1 = device_cast<float3>(translation_in_1);
	translation_out_2 = device_cast<float3>(translation_in_2);
}

inline void
pcl::gpu::KinfuTracker::convertTransforms(Matrix3frm& rotation_in_1, Matrix3frm& rotation_in_2, Vector3f& translation_in, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out)
{
	rotation_out_1 = device_cast<Mat33> (rotation_in_1);
	rotation_out_2 = device_cast<Mat33> (rotation_in_2);
	translation_out = device_cast<float3>(translation_in);
}

inline void
pcl::gpu::KinfuTracker::convertTransforms(Matrix3frm& rotation_in, Vector3f& translation_in, Mat33& rotation_out, float3& translation_out)
{
	rotation_out = device_cast<Mat33> (rotation_in);
	translation_out = device_cast<float3>(translation_in);
}


inline bool
pcl::gpu::KinfuTracker::checkIntegration(Matrix3frm& rotation_curr, Vector3f& translation_curr, Matrix3frm& rotation_prev, Vector3f& translation_prev, const float alpha)
{
	float rnorm = rodrigues2(rotation_curr.inverse() * rotation_prev).norm();
	float tnorm = (translation_curr - translation_prev).norm();

	return ((rnorm + alpha * tnorm) / 2 >= integration_metric_threshold_);
}

//////////////////////////////////////////////
inline void
pcl::gpu::KinfuTracker::prepareMaps(const DepthMap& depth_raw, const Intr& cam_intrinsics)
{
	// blur raw map
	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr[0]);

		clBilateralPyrdown->clBilateralFilter(depth_raw, depths_curr_[0]);

#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclBilateralFilter_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			ushort* pushort = new ushort[depths_curr_[0].rows()*depths_curr_[0].cols()*sizeof(ushort)];
			depths_curr_[0].download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < depths_curr_[0].rows(); i++)
			{
				for (int j = 0; j < depths_curr_[0].cols(); j++)
				{
					int index = i*depths_curr_[0].cols() + j;
					fprintf(file_out, "depths_curr_[0][%d] = %d\n", index, pushort[index]);
				}
			}
			fclose(file_out);
			delete[] pushort;
		}
#endif
		if (max_icp_distance_ > 0)
			clBilateralPyrdown->clTruncateDepth(depths_curr_[0], max_icp_distance_);
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclTruncateDepth_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			ushort* pushort = new ushort[depths_curr_[0].rows()*depths_curr_[0].cols()*sizeof(ushort)];
			depths_curr_[0].download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < depths_curr_[0].rows(); i++)
			{
				for (int j = 0; j < depths_curr_[0].cols(); j++)
				{
					int index = i*depths_curr_[0].cols() + j;
					fprintf(file_out, "depths_curr_[0][%d] = %d\n", index, pushort[index]);
				}
			}
			fclose(file_out);
			delete[] pushort;
		}
#endif
		for (int i = 0; i < levels_ - 1; ++i)
			clBilateralPyrdown->clpyrDown(depths_curr_[i], depths_curr_[i + 1]);
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			for (int level = 1; level < levels_; ++level)
			{
				char filename[256];
				sprintf(filename, "%sclpyrDown_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				FILE *file_out = fopen(filename, "w");
				ushort* pushort = new ushort[depths_curr_[level].rows()*depths_curr_[level].cols()*sizeof(ushort)];
				depths_curr_[level].download(clData->m_command_queue, pushort, 0);
				for (int i = 0; i < depths_curr_[level].rows(); i++)
				{
					for (int j = 0; j < depths_curr_[level].cols(); j++)
					{
						int index = i*depths_curr_[level].cols() + j;
						fprintf(file_out, "depths_curr_level[%d] = %d\n", index, pushort[index]);
					}
				}
				fclose(file_out);
				delete[] pushort;
			}
		}
#endif

		for (int i = start_level_; i < levels_; ++i)
		{
			clMaps->clCreateVMap(cam_intrinsics(i), depths_curr_[i], vmaps_curr_[i]);
			clMaps->clCreateNMap(vmaps_curr_[i], nmaps_curr_[i]);
			//computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			//for (int level = 0; level < levels_; ++level)
			int level = 1;
			{
				float* pfloat = new float[vmaps_curr_[level].rows()*vmaps_curr_[level].cols() * 4];

				char filename[256];
				sprintf(filename, "%sclCreateVMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				FILE *file_out = fopen(filename, "w");
				vmaps_curr_[level].download(clData->m_command_queue, pfloat, 0);
				for (int i = 0; i < vmaps_curr_[level].rows(); i++)
				{
					for (int j = 0; j < vmaps_curr_[level].cols() * 4; j++)
					{
						int index = i*vmaps_curr_[level].cols() * 4 + j;
						fprintf(file_out, "vmaps_curr_[%d] = %f\n", index, pfloat[index]);
					}
				}
				fclose(file_out);

				sprintf(filename, "%sclCreateNMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				file_out = fopen(filename, "w");
				nmaps_curr_[level].download(clData->m_command_queue, pfloat, 0);
				for (int i = 0; i < nmaps_curr_[level].rows(); i++)
				{
					for (int j = 0; j < nmaps_curr_[level].cols() * 4; j++)
					{
						int index = i*nmaps_curr_[level].cols() * 4 + j;
						fprintf(file_out, "nmaps_curr_[%d] = %f\n", index, pfloat[index]);
					}
				}
				fclose(file_out);

				delete[] pfloat;
			}
		}

#endif
	}

}

inline bool
pcl::gpu::KinfuTracker::performICP(const Intr& cam_intrinsics, Matrix3frm& Rprev, Vector3f& tprev, Matrix3frm& Rcurr, Vector3f& tcurr, Eigen::Affine3f *hint)
{
	Matrix3frm Rprev_inv = Rprev.inverse();
	Mat33 device_Rprev_inv;
	float3 device_tprev;
	bool checkICPSucces = false;
	if (hint)
	{
		Eigen::Vector3f poseVec;
		Matrix3frm R_gyro_inc = hint->rotation().matrix() * R_gyro_prev_.inverse();
		R_gyro_prev_ = hint->rotation().matrix();
		poseVec = R_gyro_inc.eulerAngles(0, 1, 2);

		if (poseVec(0) > (M_PI / 2))
		{
			poseVec(0) = poseVec(0) - M_PI;
		}
		else if (poseVec(0) < (-M_PI / 2))
		{
			poseVec(0) = poseVec(0) + M_PI;
		}

		if (poseVec(1) > (M_PI / 2))
		{
			poseVec(1) = poseVec(1) - M_PI;
		}
		else if (poseVec(1) < (-M_PI / 2))
		{
			poseVec(1) = poseVec(1) + M_PI;
		}
		if (poseVec(2) > (M_PI / 2))
		{
			poseVec(2) = poseVec(2) - M_PI;
		}
		else if (poseVec(2) < (-M_PI / 2))
		{
			poseVec(2) = poseVec(2) + M_PI;
		}

		R_gyro_.insert(R_gyro_.begin(), poseVec);

		Eigen::Vector3f Rgyro1 = 0.25 * R_gyro_.at(0) + 0.25 * R_gyro_.at(1);
		R_gyro_.pop_back();
		Eigen::Vector3f poseVec2;
		if (abs(Rgyro1(0)) >0.15 || abs(Rgyro1(1)) > 0.15 || abs(Rgyro1(2)) > 0.15)
		{
			poseVec2 = Eigen::Vector3f(0, 0, 0);
		}
		else
		{
			poseVec2 = Rgyro1;
		}

		Matrix3frm R_gyro_inc2 = (Eigen::Matrix3f)AngleAxisf(poseVec2(0), Vector3f::UnitX()) * AngleAxisf(poseVec2(1), Vector3f::UnitY()) * AngleAxisf(poseVec2(2), Vector3f::UnitZ());
		Rcurr = R_gyro_inc2 * Rprev;
		tcurr = tprev;
	}
	else
	{
		Rcurr = Rprev; // tranform to global coo for ith camera pose
		tcurr = tprev;
	}
/*
	FILE* file_out = fopen("/sdcard/test.ply", "w");

    	        fprintf(file_out, "ply\n");
                fprintf(file_out, "format ascii 1.0\n");

                fprintf(file_out, "element vertex %d",   vmaps_curr_[0].rows()*vmaps_curr_[0].cols());
                fprintf(file_out, "property float32 x\n");
                fprintf(file_out, "property float32 y\n");
                fprintf(file_out, "property float32 z\n");
                fprintf(file_out, "property float32 nx\n");
                fprintf(file_out, "property float32 ny\n");
                fprintf(file_out, "property float32 nz\n");
                fprintf(file_out, "property list uchar int vertex_indices\n");
                fprintf(file_out, "end_header\n");

    	float* pfloatn = new float[vmaps_curr_[0].rows()*vmaps_curr_[0].cols() * 4];
    	float* pfloatv = new float[vmaps_curr_[0].rows()*vmaps_curr_[0].cols() * 4];

    				vmaps_curr_[level].download(clData->m_command_queue, pfloatv, 0);
    				nmaps_curr_[0].download(clData->m_command_queue, pfloatn, 0);

    				for (int i = 0; i < nmaps_curr_[0].rows(); i++)
    				{
    					for (int j = 0; j < nmaps_curr_[0].cols(); j++)
    					{

                            for (int k = 0; k < 3; k++)
                            {
                                int index = i*vmaps_curr_[0].cols() * k + j;
                                fprintf(file_out, "%f ", pfloatv[index]);
                            }
                            for (int k = 0; k < 3; k++)
                            {
                                int index = i*nmaps_curr_[0].cols() * k + j;
                                fprintf(file_out, "%f ", pfloatn[index]);
                            }
    					}
    				}
    				fclose(file_out);
    				free(pfloatv);
    				free(pfloatn);

*/

	convertTransforms(Rprev_inv, tprev, device_Rprev_inv, device_tprev);
	{
		//ScopeTime time("icp-all");
		for (int level_index = levels_ - 1; level_index >= start_level_; --level_index)
		{
			int iter_num = icp_iterations_[level_index];
			for (int iter = 0; iter < iter_num; ++iter)
			{
				Mat33  device_Rcurr;// = device_cast<Mat33> (Rcurr);
				float3 device_tcurr;// = device_cast<float3>(tcurr);
				convertTransforms(Rcurr, tcurr, device_Rcurr, device_tcurr);
				Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A;
				Eigen::Matrix<float, 6, 1> b;

				clEstimateCombined->estimateCombined(device_Rcurr, device_tcurr, vmaps_curr_[level_index], nmaps_curr_[level_index], device_Rprev_inv, device_tprev, cam_intrinsics(level_index),
					vmaps_g_prev_[level_index], nmaps_g_prev_[level_index], distThres_, angleThres_, gbuf_[level_index], sumbuf_[level_index],/*gbuf__, sumbuf__,*/ A.data(), b.data());
#ifdef USE_OPENCL_DEBUG
				if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
				{
					char filename[256];
					sprintf(filename, "%sclestimateCombined_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
					FILE *file_out = fopen(filename, "w");
					for (int i = 0; i < A.rows(); i++)
					{
						for (int j = 0; j < A.cols(); j++)
						{
							int index = i*A.cols() + j;
							fprintf(file_out, "A[%d] = %f\n", index, A.data()[index]);
						}
					}
					for (int i = 0; i < b.rows(); i++)
					{
						for (int j = 0; j < b.cols(); j++)
						{
							int index = i*b.cols() + j;
							fprintf(file_out, "b[%d] = %f\n", index, b.data()[index]);
						}
					}
					fclose(file_out);
				}
#endif
				//checking nullspace
				double det = A.determinant();

				if (fabs(det) < 1e-15 || pcl_isnan(det))
				{
					if (pcl_isnan(det))
					{
						cout << "ICP qnan" << endl;
#ifdef __ANDORID__
						LOGD("ICP qnan");
#endif
					}
					return (false);
			}
				//float maxc = A.maxCoeff();

				Eigen::Matrix<float, 6, 1> result = A.llt().solve(b).cast<float>();
				//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

				float alpha = result(0);
				float beta = result(1);
				float gamma = result(2);

				Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf(gamma, Vector3f::UnitZ()) * AngleAxisf(beta, Vector3f::UnitY()) * AngleAxisf(alpha, Vector3f::UnitX());
				Vector3f tinc = result.tail<3>();

				//compose
				tcurr = Rinc * tcurr + tinc;
				Rcurr = Rinc * Rcurr;
				double Err = max((Rinc - Eigen::Matrix3f::Identity()).norm(), tinc.norm());
				double min_delta = min_delta_;

				if (Err < min_delta)
				{
					if (level_index == start_level_)
					{
						checkICPSucces = true;
					}
					break;
				}
			}
		}
		/** position Not found even ICP worked **/
		if (!checkICPSucces)
		{
			cout << "ICP icp_iterations_MAX_" << endl;
#ifdef __ANDORID__
			LOGD("ICP icp_iterations_MAX_!!");
#endif
		}
	}
	return checkICPSucces;
}
/////////////////////////////////////////////////////////////////////////////////
// mergesort

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge(int arr[], int l, int m, int r)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	/* create temp arrays */
	int *L = new int[n1];
	int *R = new int[n2];

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0; // Initial index of first subarray
	j = 0; // Initial index of second subarray
	k = l; // Initial index of merged subarray
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of L[], if there
	are any */
	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there
	are any */
	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}

	delete [] L;
	delete [] R;
}

/* l is for left index and r is right index of the
sub-array of arr to be sorted */
void mergeSort(int arr[], int l, int r)
{
	if (l < r)
	{
		// Same as (l+r)/2, but avoids overflow for
		// large l and h
		int m = l + (r - l) / 2;

		// Sort first and second halves
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);

		merge(arr, l, m, r);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (
const DepthMap& depth_raw, Eigen::Affine3f *hint)
{
#ifdef __ANDROID__
	LOGD("start");
#endif

	if (doreset){
#ifdef __ANDROID__
		LOGD("doreset");
#endif
		reset();
		return (false);
	}

	/*Common Values*/
	device::Intr intr(fx_, fy_, cx_, cy_);
	float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

	prepareMaps(depth_raw, intr);
#if 0

	{
		//ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
		//depth_raw.copyTo(depths_curr[0]);
		clBilateralPyrdown->clBilateralFilter(depth_raw, depths_curr_[0]);
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclBilateralFilter_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			ushort* pushort = new ushort[depths_curr_[0].rows()*depths_curr_[0].cols()*sizeof(ushort)];
			depths_curr_[0].download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < depths_curr_[0].rows(); i++)
			{
				for (int j = 0; j < depths_curr_[0].cols(); j++)
				{
					int index = i*depths_curr_[0].cols() + j;
					fprintf(file_out, "depths_curr_[0][%d] = %d\n", index, pushort[index]);
				}
			}
			fclose(file_out);
			delete[] pushort;
		}
#endif
		if (max_icp_distance_ > 0)
			clBilateralPyrdown->clTruncateDepth(depths_curr_[0], max_icp_distance_);
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclTruncateDepth_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			ushort* pushort = new ushort[depths_curr_[0].rows()*depths_curr_[0].cols()*sizeof(ushort)];
			depths_curr_[0].download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < depths_curr_[0].rows(); i++)
			{
				for (int j = 0; j < depths_curr_[0].cols(); j++)
				{
					int index = i*depths_curr_[0].cols() + j;
					fprintf(file_out, "depths_curr_[0][%d] = %d\n", index, pushort[index]);
				}
			}
			fclose(file_out);
			delete[] pushort;
		}
#endif
		for (int i = 0; i < levels_ - 1; ++i)
			clBilateralPyrdown->clpyrDown(depths_curr_[i], depths_curr_[i + 1]);
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			for (int level = 1; level < levels_; ++level)
			{
				char filename[256];
				sprintf(filename, "%sclpyrDown_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				FILE *file_out = fopen(filename, "w");
				ushort* pushort = new ushort[depths_curr_[level].rows()*depths_curr_[level].cols()*sizeof(ushort)];
				depths_curr_[level].download(clData->m_command_queue, pushort, 0);
				for (int i = 0; i < depths_curr_[level].rows(); i++)
				{
					for (int j = 0; j < depths_curr_[level].cols(); j++)
					{	
						int index = i*depths_curr_[level].cols() + j;
						fprintf(file_out, "depths_curr_level[%d] = %d\n", index, pushort[index]);
					}
				}
				fclose(file_out);
				delete[] pushort;
			}
		}
#endif

		for (int i = start_level_; i < levels_; ++i)
		{
			clMaps->clCreateVMap(intr(i), depths_curr_[i], vmaps_curr_[i]);
			clMaps->clCreateNMap(vmaps_curr_[i], nmaps_curr_[i]);
			//computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
		}
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			//for (int level = 0; level < levels_; ++level)
			int level = 1;
			{
				float* pfloat = new float[vmaps_curr_[level].rows()*vmaps_curr_[level].cols() * 4];

				char filename[256];
				sprintf(filename, "%sclCreateVMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				FILE *file_out = fopen(filename, "w");
				vmaps_curr_[level].download(clData->m_command_queue, pfloat, 0);
				for (int i = 0; i < vmaps_curr_[level].rows(); i++)
				{
					for (int j = 0; j < vmaps_curr_[level].cols() * 4; j++)
					{
						int index = i*vmaps_curr_[level].cols() * 4 + j;
						fprintf(file_out, "vmaps_curr_[%d] = %f\n", index, pfloat[index]);
					}
				}
				fclose(file_out);

				sprintf(filename, "%sclCreateNMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				file_out = fopen(filename, "w");
				nmaps_curr_[level].download(clData->m_command_queue, pfloat, 0);
				for (int i = 0; i < nmaps_curr_[level].rows(); i++)
				{
					for (int j = 0; j < nmaps_curr_[level].cols() * 4; j++)
					{
						int index = i*nmaps_curr_[level].cols() * 4 + j;
						fprintf(file_out, "nmaps_curr_[%d] = %f\n", index, pfloat[index]);
					}
				}
				fclose(file_out);

				delete[] pfloat;
					}
				}

#endif
	}
#endif // 0


	//can't perform more on first frame
	if (global_time_ == 0)
	{
#ifdef __ANDROID__
		LOGD("global_time_ == 0");
#endif
		Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
		Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose
		Matrix3frm init_Rcam_inv = init_Rcam.inverse();
		Mat33  device_Rcam, device_Rcam_inv;
		float3 device_tcam;
		
		// Convert pose to device types
		convertTransforms(init_Rcam, init_Rcam_inv, init_tcam, device_Rcam, device_Rcam_inv, device_tcam);
		
		clTsdf->integrateTsdfVolume(depths_curr_[start_level_]/*depth_raw*/, intr(start_level_), device_volume_size, device_Rcam_inv, device_tcam,
			tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_); //tsdf OCL main
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclintegrateTsdfVolume_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			cl_short2* pushort = new cl_short2[tsdf_volume_->data().rows()*tsdf_volume_->data().cols()*sizeof(ushort)];
			tsdf_volume_->data().download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < tsdf_volume_->data().rows(); i++)
			{
				for (int j = 0; j < tsdf_volume_->data().cols(); j++)
				{
					int index = i*tsdf_volume_->data().cols() + j;
					if (pushort[index].s[0] != 0)
						fprintf(file_out, "i = %d | tsdf = %f, weight = %d \n", index, (float)(pushort[index].s[0]) / (float)(32767), pushort[index].s[1]);
				}
			}
			fclose(file_out);
			delete[] pushort;
			}
#endif
		
		for (int i = start_level_; i < levels_; ++i)
		{
#ifdef __ANDROID__
			LOGD("cltranformMap start_level_", start_level_);
#endif
			clMaps->cltranformMap(vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
		}
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			//for (int level = 0; level < levels_; ++level)
			int level = 1;
			{
				float* pfloat = new float[vmaps_g_prev_[level].rows()*vmaps_g_prev_[level].cols() * 4];
				float* pfloat2 = new float[nmaps_g_prev_[level].rows()*nmaps_g_prev_[level].cols() * 4];
				char filename[256];
				sprintf(filename, "%scltranformMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
				FILE *file_out = fopen(filename, "w");
				vmaps_g_prev_[level].download(clData->m_command_queue, pfloat, 0);
				nmaps_g_prev_[level].download(clData->m_command_queue, pfloat2, 0);
				for (int i = 0; i < vmaps_g_prev_[level].rows(); i++)
				{
					for (int j = 0; j < vmaps_g_prev_[level].cols() * 4; j++)
					{
						fprintf(file_out, "vmaps_g_prev_,nmaps_g_prev_[%d] = %f,%f\n",
							i*vmaps_g_prev_[level].cols() * 4 + j, pfloat[i*vmaps_g_prev_[level].cols() * 4 + j], pfloat2[i*nmaps_g_prev_[level].cols() * 4 + j]);
					}
				}
				fclose(file_out);
				delete[] pfloat2;
				delete[] pfloat;
			}
		}
#endif
#ifdef __ANDROID__
		LOGD("KinfuTracker::operator cltranformMap end");
#endif
		++global_time_;
		//return (false);
		return true; //inyeop- first frame 에도 has_image=true -> WriteColorandPos() 수행되도록
	}


	Matrix3frm Rprev = rmats_[global_time_ - 1];//  [Ri|ti] - pos of camera, i.e.
	Vector3f   tprev = tvecs_[global_time_ - 1];//  tranfrom from camera to global coo space for (i-1)th camera pose
	Matrix3frm Rcurr;
	Vector3f tcurr;

	if (!disable_icp_)
	{

		bool PerformICPResult = false; //check ICP result

		/**perform ICP scanning and odometry**/
		PerformICPResult = performICP(intr, Rprev, tprev, Rcurr, tcurr, hint);
		if (!PerformICPResult)
		{
			return (false);
		}
		//save tranform
		rmats_.push_back(Rcurr);
		tvecs_.push_back(tcurr);
	}
	else /* if (disable_icp_) */
	{
		if (global_time_ == 0)
			++global_time_;

		Matrix3frm Rcurr = rmats_[global_time_ - 1];
		Vector3f   tcurr = tvecs_[global_time_ - 1];

		rmats_.push_back(Rcurr);
		tvecs_.push_back(tcurr);
	}


	const float alpha = 1.f;
	bool integrate = checkIntegration(Rcurr, tcurr, Rprev, tprev, alpha);
	if (global_time_ < 3)
		integrate = true;
	if (!integrate)
	{
		rmats_.pop_back();
		tvecs_.pop_back();
		return (true);
	}
	if (disable_icp_)
		integrate = true;

	///////////////////////////////////////////////////////////////////////////////////////////
	// Volume integration

	Matrix3frm Rcurr_inv = Rcurr.inverse();
	Mat33  device_Rcurr_inv, device_Rcurr;
	float3 device_tcurr;
	convertTransforms(Rcurr_inv, Rcurr, tcurr, device_Rcurr_inv, device_Rcurr, device_tcurr);
	if (integrate)
	{
		//ScopeTime time("tsdf");
		clTsdf->integrateTsdfVolume(depths_curr_[start_level_]/*depth_raw*/, intr(start_level_), device_volume_size, device_Rcurr_inv, device_tcurr,
			tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_); //tsdf OCL main
#ifdef USE_OPENCL_DEBUG
		if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
		{
			char filename[256];
			sprintf(filename, "%sclintegrateTsdfVolume_%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_);
			FILE *file_out = fopen(filename, "w");
			cl_short2* pushort = new cl_short2[tsdf_volume_->data().rows()*tsdf_volume_->data().cols()];
			tsdf_volume_->data().download(clData->m_command_queue, pushort, 0);
			for (int i = 0; i < tsdf_volume_->data().rows(); i++)
			{
				for (int j = 0; j < tsdf_volume_->data().cols(); j++)
				{
					int index = i*tsdf_volume_->data().cols() + j;
					if (pushort[index].s[0] != 0)
						fprintf(file_out, "i = %d | tsdf = %f, weight = %d \n", index, (float)(pushort[index].s[0]) / (float)(32767), pushort[index].s[1]);
				}
			}
			fclose(file_out);
			delete[] pushort;
	}
#endif

		///////////////////////////////////////////////////////////////////////////////////////////
		// Ray casting
//		Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
		{
			//ScopeTime time("ray-cast-all");
			clRayCaster->raycast(intr(start_level_), device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[start_level_], nmaps_g_prev_[start_level_]);
		

			float* pfloat = new float[vmaps_g_prev_[0].rows()*vmaps_g_prev_[0].cols() * 4];

			vmaps_g_prev_[0].download(clData->m_command_queue, pfloat, 0);
		/*	// 5x5 median filter //inyeop
           	 int window[25];
           	for(int i = 2; i < 480; i++){
           		 for(int j = 2; j < 640; j++){
           			window[0] = pfloat[4*((i - 2) * 640 + j - 2)+2];
           			window[1] = pfloat[4*((i - 2) * 640 + j - 1)+2];
           			window[2] = pfloat[4*((i - 2) * 640 + j - 0)+2];
           			window[3] = pfloat[4*((i - 2) * 640 + j + 1)+2];
           			window[4] = pfloat[4*((i - 2) * 640 + j + 2)+2];

           			window[5] = pfloat[4*((i - 1) * 640 + j - 2)+2];
           			window[6] = pfloat[4*((i - 1) * 640 + j - 1)+2];
           			window[7] = pfloat[4*((i - 1) * 640 + j - 0)+2];
           			window[8] = pfloat[4*((i - 1) * 640 + j + 1)+2];
           			window[9] = pfloat[4*((i - 1) * 640 + j + 2)+2];

                      window[10] = pfloat[4*((i - 0) * 640 + j - 2)+2];
           		   window[11] = pfloat[4*((i - 0) * 640 + j - 1)+2];
           		   window[12] = pfloat[4*((i - 0) * 640 + j - 0)+2];
           		   window[13] = pfloat[4*((i - 0) * 640 + j + 1)+2];
           		   window[14] = pfloat[4*((i - 0) * 640 + j + 2)+2];

           		   window[15] = pfloat[4*((i + 1) * 640 + j - 2)+2];
           		   window[16] = pfloat[4*((i + 1) * 640 + j - 1)+2];
           		   window[17] = pfloat[4*((i + 1) * 640 + j - 0)+2];
           		   window[18] = pfloat[4*((i + 1) * 640 + j + 1)+2];
           		   window[19] = pfloat[4*((i + 1) * 640 + j + 2)+2];

           		   window[20] = pfloat[4*((i + 2) * 640 + j - 2)+2];
           		   window[21] = pfloat[4*((i + 2) * 640 + j - 1)+2];
           		   window[22] = pfloat[4*((i + 2) * 640 + j - 0)+2];
           		   window[23] = pfloat[4*((i + 2) * 640 + j + 1)+2];
           		   window[24] = pfloat[4*((i + 2) * 640 + j + 2)+2];

           			int arr_size = sizeof(window)/sizeof(window[0]);
           			mergeSort(window, 0, arr_size - 1);

           			pfloat[i * 640 + j] = window[12];
           		 }
           	}

		cl_image_format imageMapFormat;
		imageMapFormat.image_channel_data_type = CL_FLOAT;
		imageMapFormat.image_channel_order = CL_RGBA;
            vmaps_g_prev_[0].upload(clData->m_command_queue, pfloat, 0, 640,480, &imageMapFormat);
            */
			for (int i = start_level_; i < levels_ - 1; ++i) 
			{
				clMaps->clresizeVMap(vmaps_g_prev_[i], vmaps_g_prev_[i + 1]);
				clMaps->clresizeNMap(nmaps_g_prev_[i], nmaps_g_prev_[i + 1]);

		}
#ifdef USE_OPENCL_DEBUG
			if (global_time_ < USE_OPENCL_DEBUG_FRAME_COUNT)
			{
				for (int level = 1; level < levels_; ++level)
				{
					float* pfloat = new float[vmaps_g_prev_[level].rows()*vmaps_g_prev_[level].cols() * 4];
					float* pfloat2 = new float[nmaps_g_prev_[level].rows()*nmaps_g_prev_[level].cols() * 4];
					char filename[256];
					sprintf(filename, "%sclresizeVMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
					FILE *file_out = fopen(filename, "w");
					vmaps_g_prev_[level].download(clData->m_command_queue, pfloat, 0);
					nmaps_g_prev_[level].download(clData->m_command_queue, pfloat2, 0);
					for (int i = 0; i < vmaps_g_prev_[level].rows(); i++)
					{
						for (int j = 0; j < vmaps_g_prev_[level].cols() * 4; j++)
						{
							fprintf(file_out, "vmaps_g_prev_[%d] = %f\n", i*vmaps_g_prev_[level].cols() * 4 + j, pfloat[i*vmaps_g_prev_[level].cols() * 4 + j]);
						}
					}
					fclose(file_out);
					sprintf(filename, "%sclresizeNMap_%d_level%d.txt", OPENCL_DEBUG_PATH_PREFIX, global_time_, level);
					file_out = fopen(filename, "w");
					for (int i = 0; i < vmaps_g_prev_[level].rows(); i++)
					{
						for (int j = 0; j < vmaps_g_prev_[level].cols() * 4; j++)
						{
							fprintf(file_out, "nmaps_g_prev_[%d] = %f\n", i*vmaps_g_prev_[level].cols() * 4 + j, pfloat2[i*nmaps_g_prev_[level].cols() * 4 + j]);
						}
					}
					fclose(file_out);
					delete[] pfloat2;
					delete[] pfloat;
				}
			}
#endif
		}
	}
	++global_time_;
	return (true);
		}

void pcl::gpu::KinfuTracker::exportSTL(const char *FileName)
{
	MarchingCubes* marching_cubes_ = new MarchingCubes();
	CLDeviceArray<PointXYZ> triangles_buffer_device_;
	CLDeviceArray<float> triangles_device = marching_cubes_->run(*tsdf_volume_, triangles_buffer_device_);

#ifdef ALLOC_HOST_MEMORY
	cl_int ret;
	cl_float* pfloat = (cl_float*)clEnqueueMapBuffer(clData->m_command_queue, *triangles_device.handle(), CL_TRUE, CL_MAP_READ, 0, triangles_device.size(), 0, NULL, NULL, &ret);
#else
	cl_float* pfloat = new cl_float[triangles_device.size()];
	triangles_device.download(clData->m_command_queue, pfloat);
#endif
	{
		int pointSize = triangles_device.sizeBytes() / 4 / 3;
		char filename[256];
		WriteAsciiPLY(pfloat, pointSize, FileName);

		//sprintf(filename, "%sresult.stl", OPENCL_DEBUG_PATH_PREFIX);
		//WriteBinarySTL(pfloat, pointSize, filename);
	}
#ifdef ALLOC_HOST_MEMORY
	ret = clEnqueueUnmapMemObject(clData->m_command_queue, *triangles_device.handle(), pfloat, 0, NULL, NULL);
#else
	delete[] pfloat;
#endif
	delete marching_cubes_;
}

void pcl::gpu::KinfuTracker::ComputeNormal(const float v1[3], const float v2[3], const float v3[3], float n[3])
{
	float ax, ay, az, bx, by, bz;
	float length;

	// order is important!!! maintain consistency with triangle vertex order
	ax = v3[0] - v2[0]; ay = v3[1] - v2[1]; az = v3[2] - v2[2];
	bx = v1[0] - v2[0]; by = v1[1] - v2[1]; bz = v1[2] - v2[2];

	n[0] = (ay * bz - az * by);
	n[1] = (az * bx - ax * bz);
	n[2] = (ax * by - ay * bx);

	if ((length = sqrt((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))) != 0.0)
	{
		n[0] /= length;
		n[1] /= length;
		n[2] /= length;
	}
}


void pcl::gpu::KinfuTracker::WriteAsciiPLY(const float *vertices, const int point_size, const char *FileName)
{
	FILE *fp;
	float v1[3], v2[3], v3[3];

	if ((fp = fopen(FileName, "w")) == NULL)
	{
		printf("WriteAsciiSTL Couldn't open file: %s", FileName);
		return;
	}

	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", point_size / 2);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "element face %d\n", point_size / 6);
	fprintf(fp, "property list uchar int vertex_index\n");
	fprintf(fp, "end_header\n");

	//known issues: not able to get correct vertices and indices for mesh
	for (int i = 0; i < point_size; i += 6)
	{
		for (int j = 0; j < 3; ++j) {
			v1[j] = vertices[i * 3 + j] * volume().getVoxelSize()[0];
			v2[j] = vertices[(i + 2) * 3 + j] * volume().getVoxelSize()[1];
			v3[j] = vertices[(i + 4) * 3 + j] * volume().getVoxelSize()[2];
		}
		fprintf(fp, "%f %f %f\n", v1[0], v1[2], v1[1]);
		fprintf(fp, "%f %f %f\n", v2[0], v2[2], v2[1]);
		fprintf(fp, "%f %f %f\n", v3[0], v3[2], v3[1]);
	}
	for (int i = 0; i < point_size; i += 6)
	{
		fprintf(fp, "3 ");
		fprintf(fp, "%d %d %d\n", i / 2, i / 2 + 1, i / 2 + 2);
	}

	fclose(fp);
}

void pcl::gpu::KinfuTracker::WriteAsciiSTL(const float *vertices, const int point_size, const char *FileName)
{
	FILE *fp;
	float n[3], v1[3], v2[3], v3[3];

	if ((fp = fopen(FileName, "w")) == NULL)
	{
		printf("WriteAsciiSTL Couldn't open file: %s", FileName);
		return;
	}

	if (fprintf(fp, "solid ascii\n") < 0)
	{
		fclose(fp);
		printf("WriteAsciiSTL Couldn't fprintf : %s 1", FileName);
		return;
	}

	// i ������ ��ǲ�� ���� �ٲ�. point 3���� �� for������ ó��
	for (int i = 0; i < point_size; i += 6)
	{
		for (int j = 0; j < 3; ++j) {
			v1[j] = vertices[i * 3 + j];
			v2[j] = vertices[(i + 2) * 3 + j];
			v3[j] = vertices[(i + 4) * 3 + j];
		}

		ComputeNormal(v1, v3, v2, n);

		if (fprintf(fp, " facet normal %.6g %.6g %.6g\n  outer loop\n", n[0], n[1], n[2]) < 0)
		{
			fclose(fp);
			printf("WriteAsciiSTL Couldn't fprintf : %s 2", FileName);
			return;
		}

		if (fprintf(fp, "   vertex %.6g %.6g %.6g\n", v1[0], v1[1], v1[2]) < 0)
		{
			fclose(fp);
			printf("WriteAsciiSTL Couldn't fprintf : %s 3", FileName);
			return;
		}
		if (fprintf(fp, "   vertex %.6g %.6g %.6g\n", v2[0], v2[1], v2[2]) < 0)
		{
			fclose(fp);
			printf("WriteAsciiSTL Couldn't fprintf : %s 4", FileName);
			return;
		}
		if (fprintf(fp, "   vertex %.6g %.6g %.6g\n", v3[0], v3[1], v3[2]) < 0)
		{
			fclose(fp);
			printf("WriteAsciiSTL Couldn't fprintf : %s 5", FileName);
			return;
		}

		if (fprintf(fp, "  endloop\n endfacet\n") < 0)
		{
			fclose(fp);
			printf("WriteAsciiSTL Couldn't fprintf : %s 6", FileName);
			return;
		}
	}

	if (fprintf(fp, "endsolid\n") < 0)
	{
		printf("WriteAsciiSTL Couldn't fprintf : %s 7", FileName);
	}

	fclose(fp);
}

void pcl::gpu::KinfuTracker::WriteBinarySTL(const float *vertices, const int point_size, const char *FileName)
{
	FILE *fp;
	float dn[3], v1[3], v2[3], v3[3];

	unsigned long ulint;
	unsigned short ibuff2 = 0;

	if ((fp = fopen(FileName, "wb")) == NULL)
	{
		printf("Couldn't open file: %s\n", FileName);
		return;
	}

	//  Write header
	//
	char szHeader[80 + 1];

	memset(szHeader, 32, 80);  // fill with space (ASCII=>32)
	sprintf(szHeader, "%s", "STL file made by LGE");

	if (fwrite(szHeader, 1, 80, fp) < 80)
	{
		fclose(fp);
		return;
	}

	ulint = (unsigned long int) point_size / 6;
	if (fwrite(&ulint, 1, 4, fp) < 4)
	{
		fclose(fp);
		return;
	}

	//  Write out triangle polygons.  In not a triangle polygon, only first
	//  three vertices are written.
	for (int i = 0; i < point_size; i += 6)
	{
		for (int j = 0; j < 3; ++j) {
			v1[j] = vertices[i * 3 + j];
			v2[j] = vertices[(i + 2) * 3 + j];
			v3[j] = vertices[(i + 4) * 3 + j];
		}

		ComputeNormal(v1, v3, v2, dn);
		float n[3];
		n[0] = (float)dn[0];
		n[1] = (float)dn[1];
		n[2] = (float)dn[2];
		if (fwrite(n, 4, 3, fp) < 3)
		{
			fclose(fp);
			return;
		}

		n[0] = (float)v1[0];  n[1] = (float)v1[1];  n[2] = (float)v1[2];
		if (fwrite(n, 4, 3, fp) < 3)
		{
			fclose(fp);
			return;
		}

		n[0] = (float)v2[0];  n[1] = (float)v2[1];  n[2] = (float)v2[2];
		if (fwrite(n, 4, 3, fp) < 3)
		{
			fclose(fp);
			return;
		}

		n[0] = (float)v3[0];  n[1] = (float)v3[1];  n[2] = (float)v3[2];
		if (fwrite(n, 4, 3, fp) < 3)
		{
			fclose(fp);
			return;
		}

		if (fwrite(&ibuff2, 2, 1, fp) < 1)
		{
			fclose(fp);
			return;
		}
	}

	fclose(fp);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::KinfuTracker::getCameraPose(int time) const
{
	if (time > (int)rmats_.size() || time < 0)
		time = rmats_.size() - 1;

	Eigen::Affine3f aff;
	aff.linear() = rmats_[time];
	aff.translation() = tvecs_[time];
	return (aff);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
pcl::gpu::KinfuTracker::getNumberOfPoses() const
{
	return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume&
pcl::gpu::KinfuTracker::volume() const
{
	return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume&
pcl::gpu::KinfuTracker::volume()
{
	return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const ColorVolume&
pcl::gpu::KinfuTracker::colorVolume() const
{
	return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume&
pcl::gpu::KinfuTracker::colorVolume()
{
	return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getImage(View& view) const
{
	//Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
	Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size() - 1];

	device::LightSource light;
	light.number = 1;
	light.pos[0] = device_cast<const float3>(light_source_pose);

	view.create(rows_ >> start_level_, cols_ >> start_level_);
	clGenerateImage->generateImage(vmaps_g_prev_[start_level_], nmaps_g_prev_[start_level_], light, view);
}

#ifdef CL_GL_INTEROP
void pcl::gpu::KinfuTracker::getImage(ViewImage& view) const
{
	Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size() - 1];

	device::LightSource light;
	light.number = 1;
	light.pos[0] = device_cast<const float3>(light_source_pose);

	clGenerateImage->generateImage(vmaps_g_prev_[start_level_], nmaps_g_prev_[start_level_], light, view);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud(CLDeviceArray2D<PointType>& cloud) const
{
	cloud.create(rows_, cols_);
	CLDeviceArray2D<float4>& c = (CLDeviceArray2D<float4>&)cloud;
	clMaps->convert(vmaps_g_prev_[0], c);	//device::convert(vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameNormals(CLDeviceArray2D<NormalType>& normals) const
{
	normals.create(rows_, cols_);
	CLDeviceArray2D<float8>& n = (CLDeviceArray2D<float8>&)normals;
	clMaps->convert(nmaps_g_prev_[0], n);//device::convert(nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::disableIcp() { disable_icp_ = true; }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{
	color_volume_ = pcl::gpu::ColorVolume::Ptr(new ColorVolume(*tsdf_volume_, clColorVolume, max_weight));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef __ANDROID__
bool
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{
	bool res = (*this)(depth);

	if (res && color_volume_)
	{
		const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
		device::Intr intr(fx_, fy_, cx_, cy_);

		Matrix3frm R_inv = rmats_.back().inverse();
		Vector3f   t = tvecs_.back();

		Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
		float3& device_tcurr = device_cast<float3> (t);
		clColorVolume->updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0],
			colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
	}

	return res;
}
#endif
void
pcl::gpu::KinfuTracker::setdoreset(bool r)
{
	doreset = r;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
	namespace gpu
	{
		PCL_EXPORTS void
			paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
		{
			KinfuTracker::clGenerateImage->paint3DView(rgb24, view, colors_weight);//device::paint3DView(rgb24, view, colors_weight);// need implement
		}

		PCL_EXPORTS void
			mergePointNormal(const CLDeviceArray<PointXYZ>& cloud, const CLDeviceArray<Normal>& normals, CLDeviceArray<PointNormal>& output)
		{
			const size_t size = min(cloud.size(), normals.size());
			output.create(size);

			const CLDeviceArray<float4>& c = (const CLDeviceArray<float4>&)cloud;
			const CLDeviceArray<float8>& n = (const CLDeviceArray<float8>&)normals;
			const CLDeviceArray<float12>& o = (const CLDeviceArray<float12>&)output;
			KinfuTracker::clMaps->mergePointNormal(c, n, o);//device::mergePointNormal(c, n, o);
		}

		PCL_EXPORTS Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
		{
			Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
			Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

			double rx = R(2, 1) - R(1, 2);
			double ry = R(0, 2) - R(2, 0);
			double rz = R(1, 0) - R(0, 1);

			double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
			double c = (R.trace() - 1) * 0.5;
			c = c > 1. ? 1. : c < -1. ? -1. : c;

			double theta = acos(c);

			if (s < 1e-5)
			{
				double t;

				if (c > 0)
					rx = ry = rz = 0;
				else
				{
					t = (R(0, 0) + 1)*0.5;
					rx = sqrt(std::max(t, 0.0));
					t = (R(1, 1) + 1)*0.5;
					ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
					t = (R(2, 2) + 1)*0.5;
					rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

					if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0))
						rz = -rz;
					theta /= sqrt(rx*rx + ry*ry + rz*rz);
					rx *= theta;
					ry *= theta;
					rz *= theta;
				}
			}
			else
			{
				double vth = 1 / (2 * s);
				vth *= theta;
				rx *= vth; ry *= vth; rz *= vth;
			}
			return Eigen::Vector3d(rx, ry, rz).cast<float>();
		}
	}
}
