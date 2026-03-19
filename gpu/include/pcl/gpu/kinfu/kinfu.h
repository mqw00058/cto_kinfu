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

#ifndef PCL_KINFU_KINFUTRACKER_HPP_
#define PCL_KINFU_KINFUTRACKER_HPP_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array_cl.h>
#include <pcl/gpu/kinfu/pixel_rgb.h>
#include <pcl/gpu/kinfu/tsdf_volume.h>
#include <pcl/gpu/kinfu/color_volume.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <vector>
#include "kinfu_config.h"

#include <pcl/gpu/kinfu/opencl_utils.h>
#include <pcl/gpu/kinfu/bilateral.h>
#include <pcl/gpu/kinfu/maps.h>
#include <pcl/gpu/kinfu/estimate_combined.h>
#include <pcl/gpu/kinfu/ray_caster.cl.h>
#include <pcl/gpu/kinfu/tsdf.h>
#include <pcl/gpu/kinfu/generateImage.h>
#include <pcl/gpu/kinfu/color_volume_cl.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/kinfu/extract.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "kinfu", __VA_ARGS__) 
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , "kinfu", __VA_ARGS__) 
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , "kinfu", __VA_ARGS__) 
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , "kinfu", __VA_ARGS__) 
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "kinfu", __VA_ARGS__) 
#else
#define LOGV(...)
#define LOGD(...) 
#define LOGI(...) 
#define LOGW(...) 
#define LOGE(...) 
#endif

#define PI 3.14159265358979323846264

namespace pcl
{
	namespace gpu
	{
		/** \brief KinfuTracker class encapsulates implementation of Microsoft Kinect Fusion algorithm
		  * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
		  */
		class PCL_EXPORTS KinfuTracker
		{
		public:
			/** \brief Pixel type for rendered image. */
			typedef pcl::gpu::PixelRGB PixelRGB;

			typedef CLDeviceArray2D<PixelRGB> View;
			typedef CLDeviceArray2D<unsigned short> DepthMap;
			typedef pcl::PointXYZ PointType;
			typedef pcl::Normal NormalType;

			/** \brief Constructor
			  * \param[in] rows height of depth image
			  * \param[in] cols width of depth image
			  */
			KinfuTracker(
#ifdef CL_GL_INTEROP
						GLuint*, GLuint*, GLuint*, GLuint*, GLuint*, GLuint*,
#endif
						Config config);

						
			/** \brief Sets Depth camera intrinsics
			  * \param[in] fx focal length x
			  * \param[in] fy focal length y
			  * \param[in] cx principal point x
			  * \param[in] cy principal point y
			  */

			void
				setDepthIntrinsics(float fx, float fy, float cx = -1, float cy = -1);

			/** \brief Get Depth camera intrinsics
			  * \param[out] fx focal length x
			  * \param[out] fy focal length y
			  * \param[out] cx principal point x
			  * \param[out] cy principal point y
			  */
			void
				getDepthIntrinsics(float& fx, float& fy, float& cx, float& cy);

			float getFocalLengthRGBx() { return focalLen_rgb_x_; }
			float getFocalLengthRGBy() { return focalLen_rgb_y_; }

			/** \brief Sets initial camera pose relative to volume coordiante space
			  * \param[in] pose Initial camera pose
			  */
			void
				setInitalCameraPose(const Eigen::Affine3f& pose);

			/** \brief Sets truncation threshold for depth image for ICP step only! This helps
			  *  to filter measurements that are outside tsdf volume. Pass zero to disable the truncation.
			  * \param[in] max_icp_distance Maximal distance, higher values are reset to zero (means no measurement).
			  */
			void
				setDepthTruncationForICP(float max_icp_distance = 0.f);

			/** \brief Sets ICP filtering parameters.
			  * \param[in] distThreshold distance.
			  * \param[in] sineOfAngle sine of angle between normals.
			  */
			void
				setIcpCorespFilteringParams(float distThreshold, float sineOfAngle);

			/** \brief Sets integration threshold. TSDF volume is integrated iff a camera movement metric exceedes the threshold value.
			  * The metric represents the following: M = (rodrigues(Rotation).norm() + alpha*translation.norm())/2, where alpha = 1.f (hardcoded constant)
			  * \param[in] threshold a value to compare with the metric. Suitable values are ~0.001
			  */
			void
				setCameraMovementThreshold(float threshold = 0.001f);

			void
				setFocalLength(Config config);

			/** \brief Performs initialization for color integration. Must be called before calling color integration.
			  * \param[in] max_weight max weighe for color integration. -1 means default weight.
			  */
			void
				initColorIntegration(int max_weight = -1);

			/** \brief Returns cols passed to ctor */
			int
				cols();

			/** \brief Returns rows passed to ctor */
			int
				rows();

			/** \brief Processes next frame.
			  * \param[in] depth next frame with values in millimeters
			  * \param hint
			  * \return true if can render 3D view.
			  */
			//bool operator() (const DepthMap& depth, Eigen::Affine3f* hint = NULL);
			bool operator() (
				const DepthMap& depth_raw,

				Eigen::Affine3f* hint = NULL);

			/** \brief Processes next frame (both depth and color integration). Please call initColorIntegration before invpoking this.
			  * \param[in] depth next depth frame with values in millimeters
			  * \param[in] colors next RGB frame
			  * \return true if can render 3D view.
			  */
			bool operator() (const DepthMap& depth, const View& colors);

			/** \brief Returns camera pose at given time, default the last pose
			  * \param[in] time Index of frame for which camera pose is returned.
			  * \return camera pose
			  */
			Eigen::Affine3f
				getCameraPose(int time = -1) const;

			/** \brief Returns number of poses including initial */
			size_t
				getNumberOfPoses() const;

			/** \brief Returns TSDF volume storage */
			const TsdfVolume& volume() const;

			/** \brief Returns TSDF volume storage */
			TsdfVolume& volume();

			/** \brief Returns color volume storage */
			const ColorVolume& colorVolume() const;

			/** \brief Returns color volume storage */
			ColorVolume& colorVolume();

			/** \brief Renders 3D scene to display to human
			  * \param[out] view output array with image
			  */
			void
				getImage(View& view) const;

			/** \brief Returns point cloud abserved from last camera pose
			  * \param[out] cloud output array for points
			  */
			void
				getLastFrameCloud(CLDeviceArray2D<PointType>& cloud) const;

			/** \brief Returns point cloud abserved from last camera pose
			  * \param[out] normals output array for normals
			  */
			void
				getLastFrameNormals(CLDeviceArray2D<NormalType>& normals) const;

			/** \brief Disables ICP forever */
			void disableIcp();

		private:

			/** \brief Number of pyramid levels */
			unsigned int  levels_;			

			/** \brief ICP Correspondences  map type */
			typedef CLDeviceArray2D<int> CorespMap;

			/** \brief Vertex or Normal Map type */
			typedef CLDeviceImage2D MapArr;
			typedef CLDeviceArray2D<float> BufArr;

			typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
			typedef Eigen::Vector3f Vector3f;


			/** \brief helper function that converts transforms from host to device types
			* \param[in] transformIn1 first transform to convert
			* \param[in] transformIn2 second transform to convert
			* \param[in] translationIn1 first translation to convert
			* \param[in] translationIn2 second translation to convert
			* \param[out] transformOut1 result of first transform conversion
			* \param[out] transformOut2 result of second transform conversion
			* \param[out] translationOut1 result of first translation conversion
			* \param[out] translationOut2 result of second translation conversion
			*/
			inline void
				convertTransforms(Matrix3frm& transform_in_1, Matrix3frm& transform_in_2, Eigen::Vector3f& translation_in_1, Eigen::Vector3f& translation_in_2,
				pcl::device::Mat33& transform_out_1, pcl::device::Mat33& transform_out_2, float3& translation_out_1, float3& translation_out_2);

			/** \brief helper function that converts transforms from host to device types
			* \param[in] transformIn1 first transform to convert
			* \param[in] transformIn2 second transform to convert
			* \param[in] translationIn translation to convert
			* \param[out] transformOut1 result of first transform conversion
			* \param[out] transformOut2 result of second transform conversion
			* \param[out] translationOut result of translation conversion
			*/
			inline void
				convertTransforms(Matrix3frm& transform_in_1, Matrix3frm& transform_in_2, Eigen::Vector3f& translation_in,
				pcl::device::Mat33& transform_out_1, pcl::device::Mat33& transform_out_2, float3& translation_out);

			/** \brief helper function that converts transforms from host to device types
			* \param[in] transformIn transform to convert
			* \param[in] translationIn translation to convert
			* \param[out] transformOut result of transform conversion
			* \param[out] translationOut result of translation conversion
			*/
			inline void
				convertTransforms(Matrix3frm& transform_in, Eigen::Vector3f& translation_in,
				pcl::device::Mat33& transform_out, float3& translation_out);
			/** \brief helper function that check integrated or not
			* \param[in] transformIn transform of current
			* \param[in] translationIn translation of current
			* \param[in] transformIn transform of previus
			* \param[in] translationIn translation of previus
			* \param[in] alpha value
			* \return true if have to integrate.
			*/
			inline bool
				checkIntegration(Matrix3frm& rotation_curr, Vector3f& translation_curr, Matrix3frm& rotation_prev, Vector3f& translation_prev, const float alpha);

			/** \brief helper function that check integrated or not
			* \param[in] transformIn transform of current
			* \param[in] translationIn translation of current
			* \param[in] threshhold const value for checking
			* \param[in] alpha const value
			* \return true if have to integrate.
			*/
			inline bool
				checkSaveKeyframe(Matrix3frm& rotation_curr, Vector3f& translation_curr, const float threshold, const float alpha);
			/** \brief helper function that pre-process a raw detph map the kinect fusion algorithm.
			* The raw depth map is first blured, eventually truncated, and downsampled for each pyramid level.
			* Then, vertex and normal maps are computed for each pyramid level.
			* \param[in] depth_raw the raw depth map to process
			* \param[in] cam_intrinsics intrinsics of the camera used to acquire the depth map
			*/
			inline void
				prepareMaps(const DepthMap& depth_raw, const pcl::device::Intr& cam_intrinsics);

			/** \brief helper function that performs GPU-based ICP, using vertex and normal maps stored in v/nmaps_curr_ and v/nmaps_g_prev_
			* The function requires the previous local camera pose (translation and inverted rotation) as well as camera intrinsics.
			* It will return the newly computed pose found as global rotation and translation.
			* \param[in] cam_intrinsics intrinsics of the camera
			* \param[in] cam_intrinsics intrinsics of the camera
			* \param[in] previous_global_rotation previous local rotation of the camera
			* \param[in] previous_global_translation previous local translation of the camera
			* \param[out] current_global_rotation computed global rotation
			* \param[out] current_global_translation computed global translation
			* \param[in] hint rotation and translation computed external gyro sensor
			* \return true if ICP has converged.
			*/
			inline bool
				performICP(const pcl::device::Intr& cam_intrinsics, Matrix3frm& previous_global_rotation, Vector3f& previous_global_translation, Matrix3frm& current_global_rotation, Vector3f& current_global_translation, Eigen::Affine3f *hint = NULL);

			/** \brief Height of input depth image. */
			int rows_;
			/** \brief Width of input depth image. */
			int cols_;
			/** \brief Frame counter */
			int global_time_;
			/** \brief start depth level */
			int start_level_; 
			/** \brief Truncation threshold for depth image for ICP step */
			float max_icp_distance_;

			/** \brief Intrinsic parameters of depth camera. */
			float fx_, fy_, cx_, cy_;

			/** \brief Tsdf volume container. */
			TsdfVolume::Ptr tsdf_volume_;
			ColorVolume::Ptr color_volume_;

			/** \brief Initial camera rotation in volume coo space. */
			Matrix3frm init_Rcam_;

			/** \brief Initial camera position in volume coo space. */
			Vector3f   init_tcam_;

			/** \brief array with IPC iteration numbers for each pyramid level */
			int* icp_iterations_;
			//int* icp_iterations_;
			/** \brief array with IPC iteration Max number for each pyramid level */
			int icp_iterations_MAX_;
			/** \brief distance threshold in correspondences filtering */
			float  distThres_;
			/** \brief angle threshold in correspondences filtering. Represents max sine of angle between normals. */
			float angleThres_;


			/** \brief Depth pyramid. */
			std::vector<DepthMap> depths_curr_;

			/** \brief Vertex maps pyramid for current frame in global coordinate space. */
			std::vector<MapArr> vmaps_g_curr_;
			/** \brief Normal maps pyramid for current frame in global coordinate space. */
			std::vector<MapArr> nmaps_g_curr_;

			/** \brief Vertex maps pyramid for previous frame in global coordinate space. */
			std::vector<MapArr> vmaps_g_prev_;
			/** \brief Normal maps pyramid for previous frame in global coordinate space. */
			std::vector<MapArr> nmaps_g_prev_;
			/** \brief Vertex maps pyramid for current frame in current coordinate space. */
			std::vector<MapArr> vmaps_curr_;
			/** \brief Normal maps pyramid for current frame in current coordinate space. */
			std::vector<MapArr> nmaps_curr_;

///////////////////////////////////
			/** \brief Array of buffers with ICP correspondences for each pyramid level. */
			std::vector<CorespMap> coresps_;

			/** \brief Buffer for storing scaled depth image */
			CLDeviceArray2D<float> depthRawScaled_;

			/** \brief Temporary buffer for ICP */
			std::vector<BufArr> gbuf_; //CLDeviceArray2D<double> gbuf_;
			/** \brief Buffer to store MLS matrix. */
			std::vector<BufArr> sumbuf_; //CLDeviceArray<double> sumbuf_;

			/** \brief Array of camera rotation matrices for each moment of time. */
			std::vector<Matrix3frm> rmats_;

			/** \brief Array of camera rotation matrices for each moment of time. */
			std::vector<Eigen::Vector3f> R_gyro_;
			/** \brief rotation matrices for previus gyroscope */
			Matrix3frm R_gyro_prev_;
			/** \brief Array of camera translations for each moment of time. */
			std::vector<Vector3f> tvecs_;

			/** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
			float integration_metric_threshold_;

			float focalLen_rgb_x_;
			float focalLen_rgb_y_;
			float focalLen_depth_x_;
			float focalLen_depth_y_;

			/** \brief ICP step is completelly disabled. Inly integratio now */
			bool disable_icp_;
			/** \ brief min_delta is value for checking ICP error result*/
			double  min_delta_;
			/** \brief Allocates all GPU internal buffers.
			  * \param[in] rows_arg
			  * \param[in] cols_arg
			  */
			void
				allocateBufffers(int rows_arg, int cols_arg);

			/** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
			  */
			void
				reset();

			
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		private:
			bool doreset;
			opencl_utils *clData;
			cl_bilateral_pyrdown *clBilateralPyrdown;
			
			cl_estimate_combined * clEstimateCombined;
			cl_ray_caster* clRayCaster;
			cl_tsdf_volume *clTsdf;
			cl_extract *clExtract;
		public:
			void setdoreset(bool r);
			static cl_maps *clMaps;
			static cl_generateImage *clGenerateImage;
			cl_color_volume *clColorVolume;
			void exportSTL(const char *FileName);
			void ComputeNormal(const float v1[3], const float v2[3], const float v3[3], float n[3]);
			void WriteAsciiSTL(const float *vertices, const int point_size, const char *FileName);
			void WriteBinarySTL(const float *vertices, const int point_size, const char *FileName);
			void WriteAsciiPLY(const float *vertices, const int point_size, const char *FileName);

#ifdef CL_GL_INTEROP
		public:
			typedef CLDeviceImage2D ViewImage;
			typedef CLDeviceImage2D DepthImage;
			void getImage(ViewImage& view) const;
		private:
			//GLuint* vmaps_g_curr_tex;
			//GLuint* nmaps_g_curr_tex;
			GLuint* vmaps_g_prev_tex;
			GLuint* nmaps_g_prev_tex;
			//GLuint* vmaps_curr_tex;
			//GLuint* nmaps_curr_tex;
		public:
			void setGLTextureIDs(GLuint* ,GLuint* ,GLuint* ,GLuint* ,GLuint* ,GLuint*);
#endif
		};
	}
};

#endif /* PCL_KINFU_KINFUTRACKER_HPP_ */
