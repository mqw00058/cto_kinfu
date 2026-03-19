/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
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
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/


#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>

#include <pcl/gpu/kinfu/opencl_utils.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/gpu/kinfu/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/exceptions.h>

#include "openni_capture.h"
#include <pcl/visualization/point_cloud_color_handlers.h>
#include "evaluation.h"

#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#include "camera_pose.h"

#include <opencv2\opencv.hpp>
#include <math.h>
#include <stdio.h>

#ifdef HAVE_OPENCV  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "video_recorder.h"
#endif
typedef pcl::ScopeTime ScopeTimeT;

//#include "../src/internal.h"

#include "lodepng.h"

const char* OPENCL_DEBUG_PATH_PREFIX = "./Debug/";

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

#include <ShellApi.h>
#include "convexhull.h"

namespace pcl
{
	namespace gpu
	{
		void paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
		void mergePointNormal(const CLDeviceArray<PointXYZ>& cloud, const CLDeviceArray<Normal>& normals, CLDeviceArray<PointNormal>& output);
		Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
	}

	namespace visualization
	{
		//////////////////////////////////////////////////////////////////////////////////////
		/** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
		* fields from an additional cloud as the color at each point.
		* \author Anatoly Baksheev
		* \ingroup visualization
		*/
		template <typename PointT>
		class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT>
		{
			using PointCloudColorHandler<PointT>::capable_;
			using PointCloudColorHandler<PointT>::cloud_;

			typedef typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr PointCloudConstPtr;
			typedef typename pcl::PointCloud<RGB>::ConstPtr RgbCloudConstPtr;

		public:
			typedef boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT> > Ptr;
			typedef boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT> > ConstPtr;

			/** \brief Constructor. */
			PointCloudColorHandlerRGBCloud(const PointCloudConstPtr& cloud, const RgbCloudConstPtr& colors)
				: rgb_(colors)
			{
				cloud_ = cloud;
				capable_ = true;
			}

			/** \brief Obtain the actual color for the input dataset as vtk scalars.
			* \param[out] scalars the output scalars containing the color for the dataset
			* \return true if the operation was successful (the handler is capable and
			* the input cloud was given as a valid pointer), false otherwise
			*/
			virtual bool
				getColor(vtkSmartPointer<vtkDataArray> &scalars) const
			{
				if (!capable_ || !cloud_)
					return (false);

				if (!scalars)
					scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
				scalars->SetNumberOfComponents(3);

				vtkIdType nr_points = vtkIdType(cloud_->points.size());
				reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples(nr_points);
				unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer(0);

				// Color every point
				if (nr_points != int(rgb_->points.size()))
					std::fill(colors, colors + nr_points * 3, static_cast<unsigned char> (0xFF));
				else
					for (vtkIdType cp = 0; cp < nr_points; ++cp)
					{
						int idx = cp * 3;
						colors[idx + 0] = rgb_->points[cp].r;
						colors[idx + 1] = rgb_->points[cp].g;
						colors[idx + 2] = rgb_->points[cp].b;
					}
				return (true);
			}

		private:
			virtual std::string
				getFieldName() const { return ("additional rgb"); }
			virtual std::string
				getName() const { return ("PointCloudColorHandlerRGBCloud"); }

			RgbCloudConstPtr rgb_;
		};
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
vector<string> getPcdFilesInDir(const string& directory)
{
	namespace fs = boost::filesystem;
	fs::path dir(directory);

	std::cout << "path: " << directory << std::endl;
	if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
		PCL_THROW_EXCEPTION(pcl::IOException, "No valid PCD directory given!\n");

	vector<string> result;
	fs::directory_iterator pos(dir);
	fs::directory_iterator end;

	for (; pos != end; ++pos)
		if (fs::is_regular_file(pos->status()))
			if (fs::extension(*pos) == ".pcd")
			{
#if BOOST_FILESYSTEM_VERSION == 3
				result.push_back(pos->path().string());
#else
				result.push_back(pos->path());
#endif
				cout << "added: " << result.back() << endl;
			}

	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose(visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
	Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, -1.f);
	Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
	Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);
	viewer.setCameraPosition(pos_vector[0], pos_vector[1], pos_vector[2],
		look_at_vector[0], look_at_vector[1], look_at_vector[2],
		up_vector[0], up_vector[1], up_vector[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f
getViewerPose(visualization::PCLVisualizer& viewer)
{
	Eigen::Affine3f pose = viewer.getViewerPose();
	Eigen::Matrix3f rotation = pose.linear();

	Matrix3f axis_reorder;
	axis_reorder << 0, 0, 1,
		-1, 0, 0,
		0, -1, 0;

	rotation = rotation * axis_reorder;
	pose.linear() = rotation;
	return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
writeCloudFile(int format, const CloudT& cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
writePolygonMeshFile(int format, const pcl::PolygonMesh& mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{
	typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());

	pcl::copyPointCloud(points, *merged_ptr);
	for (size_t i = 0; i < colors.size(); ++i)
		merged_ptr->points[i].rgba = colors.points[i].rgba;

	return merged_ptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<Eigen::Vector2f> convertToMeshSlice(const CLDeviceArray<float>& triangles, Eigen::Vector3f voxelSize, int pickPoint)
{

	std::vector<Eigen::Vector2f> vPoints;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = (int)triangles.size() / 3 / 2;
	cloud.height = 1;
	cloud.points.resize(cloud.width);
	float* pfloat = new float[triangles.sizeBytes()];
	triangles.download(opencl_utils::get()->m_command_queue, pfloat);

	for (int i = 0; i < cloud.points.size(); i++)
	{
		cloud.points.at(i).data[0] = pfloat[2 * i * 3] * voxelSize[0];
		cloud.points.at(i).data[1] = pfloat[2 * i * 3 + 1] * voxelSize[1];
		cloud.points.at(i).data[2] = pfloat[2 * i * 3 + 2] * voxelSize[2];
		cloud.points.at(i).data[3] = 1.0f;
	}
	delete[] pfloat;

	vPoints.clear();

	for (int i = 0; i < cloud.points.size(); i++)
	{
		if ((int)(cloud.points.at(i).data[1] * VOLUME_Y) == pickPoint)
			vPoints.push_back(Vector2f(cloud.points.at(i).data[0] * 100.f, cloud.points.at(i).data[2] * 100.f));
	}

	return vPoints;
}


boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const CLDeviceArray<float>& triangles, Eigen::Vector3f voxelSize)
{
	if (triangles.empty())
		return boost::shared_ptr<pcl::PolygonMesh>();

	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = (int)triangles.size() / 3 / 2;
	cloud.height = 1;
	cloud.points.resize(cloud.width);
	float* pfloat = new float[triangles.sizeBytes()];
	triangles.download(opencl_utils::get()->m_command_queue, pfloat);

	for (int i = 0; i < cloud.points.size(); i++)
	{
		cloud.points.at(i).data[0] = pfloat[2 * i * 3] * voxelSize[0];
		cloud.points.at(i).data[1] = pfloat[2 * i * 3 + 1] * voxelSize[1];
		cloud.points.at(i).data[2] = pfloat[2 * i * 3 + 2] * voxelSize[2];
		cloud.points.at(i).data[3] = 1.0f;
	}

	delete[] pfloat;

	boost::shared_ptr<pcl::PolygonMesh> mesh_ptr(new pcl::PolygonMesh());
	pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);

	mesh_ptr->polygons.resize(cloud.points.size() / 3);
	for (size_t i = 0; i < mesh_ptr->polygons.size(); ++i)
	{
		pcl::Vertices v;
		v.vertices.push_back(i * 3 + 0);
		v.vertices.push_back(i * 3 + 1);
		v.vertices.push_back(i * 3 + 2);
		mesh_ptr->polygons[i] = v;
	}
	return mesh_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
	CurrentFrameCloudView() : cloud_device_(opencl_utils::get()->m_context, 480, 640), cloud_viewer_("Frame Cloud Viewer")
	{
		cloud_ptr_ = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);

		cloud_viewer_.setBackgroundColor(0, 0, 0.15);
		cloud_viewer_.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1);
		cloud_viewer_.addCoordinateSystem(1.0, "global");
		cloud_viewer_.initCameraParameters();
		cloud_viewer_.setPosition(640, 500);
		cloud_viewer_.setSize(640, 480);
		cloud_viewer_.setCameraClipDistances(0.01, 10.01);
	}

	void
		show(const KinfuTracker& kinfu)
	{
		kinfu.getLastFrameCloud(cloud_device_);

		int c;
		cloud_device_.download(opencl_utils::get()->m_command_queue, cloud_ptr_->points, c);
		cloud_ptr_->width = cloud_device_.cols();
		cloud_ptr_->height = cloud_device_.rows();
		cloud_ptr_->is_dense = false;

		cloud_viewer_.removeAllPointClouds();
		cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
		cloud_viewer_.spinOnce();
	}

	void
		setViewerPose(const Eigen::Affine3f& viewer_pose) {
		::setViewerPose(cloud_viewer_, viewer_pose);
	}

	PointCloud<PointXYZ>::Ptr cloud_ptr_;
	CLDeviceArray2D<PointXYZ> cloud_device_;
	visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
	ImageView(int viz) : view_device_(opencl_utils::get()->m_context), colors_device_(opencl_utils::get()->m_context), generated_depth_(opencl_utils::get()->m_context), viz_(viz), paint_image_(false), accumulate_views_(false)
	{
		if (viz_)
		{
			viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
			viewerScene_->setWindowTitle("View3D from ray tracing");

			viewerScene_->setPosition(0, 0);

			viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
			viewerDepth_->setWindowTitle("Kinect Depth stream");
			viewerDepth_->setPosition(960, 0);
			//viewerColor_.setWindowTitle ("Kinect RGB stream");
		}
	}

	void
		showScene(KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr = 0)
	{
		if (pose_ptr)
		{
			raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
			raycaster_ptr_->generateSceneView(view_device_);
		}
		else
			kinfu.getImage(view_device_);

		if (paint_image_ && registration && !pose_ptr)
		{
			colors_device_.upload(opencl_utils::get()->m_command_queue, rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
			paint3DView(colors_device_, view_device_);
		}


		int cols;
		view_device_.download(opencl_utils::get()->m_command_queue, view_host_, cols);
		if (viz_)
			viewerScene_->showRGBImage(reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols(), view_device_.rows());

		//viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);

#ifdef HAVE_OPENCV
		if (accumulate_views_)
		{
			views_.push_back(cv::Mat());
			cv::cvtColor(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back(), CV_RGB2GRAY);
			//cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
		}
#endif
	}

	void
		showDepth(const PtrStepSz<const unsigned short>& depth)
	{
		if (viz_)
			viewerDepth_->showShortImage(depth.data, depth.cols, depth.rows, 0, 5000, true);
	}


	void
		showGeneratedDepth(KinfuTracker& kinfu, const Eigen::Affine3f& pose)
	{
		raycaster_ptr_->run(kinfu.volume(), pose);
		raycaster_ptr_->generateDepthImage(generated_depth_);

		int c;
		vector<unsigned short> data;
		generated_depth_.download(opencl_utils::get()->m_command_queue, data, c);

		if (viz_)
			viewerDepth_->showShortImage(&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
	}

	void
		toggleImagePaint()
	{
		paint_image_ = !paint_image_;
		cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
	}

	int viz_;
	bool paint_image_;
	bool accumulate_views_;

	visualization::ImageViewer::Ptr viewerScene_;
	visualization::ImageViewer::Ptr viewerDepth_;
	//visualization::ImageViewer viewerColor_;

	KinfuTracker::View view_device_;
	KinfuTracker::View colors_device_;
	vector<KinfuTracker::PixelRGB> view_host_;

	RayCaster::Ptr raycaster_ptr_;

	KinfuTracker::DepthMap generated_depth_;

#ifdef HAVE_OPENCV
	vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
	enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };


	SceneCloudView(int viz) : cloud_buffer_device_(opencl_utils::get()->m_context), normals_device_(opencl_utils::get()->m_context), combined_device_(opencl_utils::get()->m_context), point_colors_device_(opencl_utils::get()->m_context), triangles_buffer_device_(opencl_utils::get()->m_context), triangles_device_copy_(opencl_utils::get()->m_context), viz_(viz), extraction_mode_(GPU_Connected6), compute_normals_(false), valid_combined_(false), cube_added_(false),
		NumOfPose_(0)
	{
		cloud_ptr_ = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);
		normals_ptr_ = PointCloud<Normal>::Ptr(new PointCloud<Normal>);
		combined_ptr_ = PointCloud<PointNormal>::Ptr(new PointCloud<PointNormal>);
		point_colors_ptr_ = PointCloud<RGB>::Ptr(new PointCloud<RGB>);

		if (viz_)
		{
			cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Scene Cloud Viewer"));

			cloud_viewer_->setBackgroundColor(0, 0, 0);
			cloud_viewer_->addCoordinateSystem(1.0, "global");
			cloud_viewer_->initCameraParameters();
			cloud_viewer_->setPosition(320, 500);
			cloud_viewer_->setSize(640, 480);
			cloud_viewer_->setCameraClipDistances(0.01, 10.01);

			cloud_viewer_->addText("H: print help", 2, 15, 20, 34, 135, 246);
			std::stringstream ss;
			ss.str("");
			ss << "length";
			cloud_viewer_->addText("Length :  0.00 cm", 440, 10, 20, 34, 135, 246, ss.str());
		}
	}
	inline void
		drawCameraCoordinateSystem(Eigen::Affine3f& pose, const string& name)
	{

		size_t NumOfPose = NumOfPose_;
		std::stringstream ss;
		ss.str("");
		ss << name << "_" << NumOfPose;
		cloud_viewer_->addCoordinateSystem(0.1f, pose, ss.str(), 0);
		cloud_viewer_->spinOnce();
		NumOfPose_++;
	}
	inline void
		removeCameraCoordinateSystem(const string& name)
	{
		std::stringstream ss;
		for (size_t NumOfPose = 0; NumOfPose < NumOfPose_; NumOfPose++)
		{
			ss.str("");
			ss << name << "_" << NumOfPose;
			cloud_viewer_->removeCoordinateSystem(ss.str(), 0);
		}
		NumOfPose_ = 0;
	}
	inline void
		drawCamera(Eigen::Affine3f& pose, const string& name, double r, double g, double b)
	{
		double focal = focalLen_rgb_x_;
		double height = 480;
		double width = 640;

		// create a 5-point visual for each camera
		pcl::PointXYZ p1, p2, p3, p4, p5;
		p1.x = 0; p1.y = 0; p1.z = 0;
		double angleX = RAD2DEG(2.0 * atan(width / (2.0*focal)));
		double angleY = RAD2DEG(2.0 * atan(height / (2.0*focal)));
		double dist = 0.1;
		double minX, minY, maxX, maxY;
		maxX = dist*tan(atan(width / (2.0*focal)));
		minX = -maxX;
		maxY = dist*tan(atan(height / (2.0*focal)));
		minY = -maxY;
		p2.x = minX; p2.y = minY; p2.z = dist;
		p3.x = maxX; p3.y = minY; p3.z = dist;
		p4.x = maxX; p4.y = maxY; p4.z = dist;
		p5.x = minX; p5.y = maxY; p5.z = dist;
		p1 = pcl::transformPoint(p1, pose);
		p2 = pcl::transformPoint(p2, pose);
		p3 = pcl::transformPoint(p3, pose);
		p4 = pcl::transformPoint(p4, pose);
		p5 = pcl::transformPoint(p5, pose);
		std::stringstream ss;
		ss.str("");
		ss << name << "_line1";
		cloud_viewer_->addLine(p1, p2, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line2";
		cloud_viewer_->addLine(p1, p3, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line3";
		cloud_viewer_->addLine(p1, p4, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line4";
		cloud_viewer_->addLine(p1, p5, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line5";
		cloud_viewer_->addLine(p2, p5, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line6";
		cloud_viewer_->addLine(p5, p4, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line7";
		cloud_viewer_->addLine(p4, p3, r, g, b, ss.str());
		ss.str("");
		ss << name << "_line8";
		cloud_viewer_->addLine(p3, p2, r, g, b, ss.str());
	}
	inline void
		removeCamera(const string& name)
	{
		cloud_viewer_->removeShape(name);
		std::stringstream ss;
		ss.str("");
		ss << name << "_line1";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line2";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line3";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line4";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line5";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line6";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line7";
		cloud_viewer_->removeShape(ss.str());
		ss.str("");
		ss << name << "_line8";
		cloud_viewer_->removeShape(ss.str());
	}
	void
		show(KinfuTracker& kinfu, bool integrate_colors)
	{
		viewer_pose_ = kinfu.getCameraPose();

		ScopeTimeT time("PointCloud Extraction");
		cout << "\nGetting cloud... " << flush;

		valid_combined_ = false;

		if (extraction_mode_ != GPU_Connected6)     // So use CPU
		{
			kinfu.volume().fetchCloudHost(*cloud_ptr_, extraction_mode_ == CPU_Connected26);
		}
		else
		{
			CLDeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud(cloud_buffer_device_);

			if (compute_normals_)
			{
				kinfu.volume().fetchNormals(extracted, normals_device_); //ok
				pcl::gpu::mergePointNormal(extracted, normals_device_, combined_device_);// ok
				combined_device_.download(opencl_utils::get()->m_command_queue, combined_ptr_->points);
				combined_ptr_->width = (int)combined_ptr_->points.size();
				combined_ptr_->height = 1;

				valid_combined_ = true;
			}
			else
			{
				extracted.download(opencl_utils::get()->m_command_queue, cloud_ptr_->points);
				cloud_ptr_->width = (int)cloud_ptr_->points.size();
				cloud_ptr_->height = 1;
			}

			if (integrate_colors)
			{
				kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
				point_colors_device_.download(opencl_utils::get()->m_command_queue, point_colors_ptr_->points);
				point_colors_ptr_->width = (int)point_colors_ptr_->points.size();
				point_colors_ptr_->height = 1;
			}
			else
				point_colors_ptr_->points.clear();
		}
		size_t points_size = valid_combined_ ? combined_ptr_->points.size() : cloud_ptr_->points.size();
		cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

		if (viz_)
		{
			cloud_viewer_->removeAllPointClouds();
			if (valid_combined_)
			{
				visualization::PointCloudColorHandlerRGBCloud<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
				cloud_viewer_->addPointCloud<PointNormal>(combined_ptr_, rgb, "Cloud");
				cloud_viewer_->addPointCloudNormals<PointNormal>(combined_ptr_, 50);
			}
			else
			{
				visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
				cloud_viewer_->addPointCloud<PointXYZ>(cloud_ptr_, rgb);
			}
		}
	}

	void
		toggleCube(const Eigen::Vector3f& size)
	{
		if (!viz_)
			return;

		if (cube_added_)
			cloud_viewer_->removeShape("cube");
		else
			//cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0)*0.25, size(1)*0.25, size(2)*0.25);
			cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

		cube_added_ = !cube_added_;
	}

	void
		toggleExtractionMode()
	{
		extraction_mode_ = (extraction_mode_ + 1) % 3;

		switch (extraction_mode_)
		{
		case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
		case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
		case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
		}
		;
	}

	void
		toggleNormals()
	{
		compute_normals_ = !compute_normals_;
		cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
	}

	void
		clearClouds(bool print_message = false)
	{
		if (!viz_)
			return;

		cloud_viewer_->removeAllPointClouds();
		cloud_ptr_->points.clear();
		normals_ptr_->points.clear();
		if (print_message)
			cout << "Clouds/Meshes were cleared" << endl;
	}

	void
		showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
	{
		if (!viz_)
			return;

		ScopeTimeT time("Mesh Extraction");
		cout << "\nGetting mesh... " << flush;

		if (!marching_cubes_)
			marching_cubes_ = MarchingCubes::Ptr(new MarchingCubes());

		CLDeviceArray<float> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
		triangles_device_copy_ = triangles_device; // for measure convex size

		mesh_ptr_ = convertToMesh(triangles_device, kinfu.volume().getVoxelSize());

		cloud_viewer_->removeAllPointClouds();
		if (mesh_ptr_)
			cloud_viewer_->addPolygonMesh(*mesh_ptr_);

		cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
	}

	void
		showMeshSize(KinfuTracker& kinfu, int pickPoint)
	{
		std::vector<Eigen::Vector2f> vPoints;
		float length;

		vPoints = convertToMeshSlice(triangles_device_copy_, kinfu.volume().getVoxelSize(), pickPoint);
		length = ConvexHull::convexHull(vPoints);

		printf("Length: %.2f cm\n", length);

		setLength(length);
		vPoints.clear();
	}

	void setFocalLengthRGB(float x, float y){ focalLen_rgb_x_ = x; focalLen_rgb_y_ = y; }

	void setLength(float d) { length = d; }
	float getLength() { return length; }

	float length;
	int viz_;
	int extraction_mode_;
	bool compute_normals_;
	bool valid_combined_;
	bool cube_added_;
	unsigned long NumOfPose_;
	Eigen::Affine3f viewer_pose_;

	float focalLen_rgb_x_;
	float focalLen_rgb_y_;

	visualization::PCLVisualizer::Ptr cloud_viewer_;

	PointCloud<PointXYZ>::Ptr cloud_ptr_;
	PointCloud<Normal>::Ptr normals_ptr_;

	CLDeviceArray<PointXYZ> cloud_buffer_device_;
	CLDeviceArray<Normal> normals_device_;

	PointCloud<PointNormal>::Ptr combined_ptr_;
	CLDeviceArray<PointNormal> combined_device_;

	CLDeviceArray<RGB> point_colors_device_;
	PointCloud<RGB>::Ptr point_colors_ptr_;

	MarchingCubes::Ptr marching_cubes_;
	CLDeviceArray<PointXYZ> triangles_buffer_device_;

	boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
	CLDeviceArray<float> triangles_device_copy_;
};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
	Config config_;

	enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };

	KinFuApp(pcl::Grabber& source,
		float vsz,
		int icp,
		int viz,
		Config config,
		boost::shared_ptr<CameraPoseProcessor> pose_processor = boost::shared_ptr<CameraPoseProcessor>())
		: depth_device_(opencl_utils::get()->m_context), exit_(false),
		scan_(false),
		scan_mesh_(false),
		scan_volume_(false),
		independent_camera_(false),
		doreset_(false),
		doTextureMapping_(false),
		registration_(false),
		integrate_colors_(false),
		texture_mapping_(false),
		pcd_source_(false),
		start_(false),
		focal_length_(-1.f),
		capture_(source),
		scene_cloud_view_(viz),
		image_view_(viz),
		time_ms_(0),
		icp_(icp),
		viz_(viz),
		pose_processor_(pose_processor),
		name_("camPose"),
		config_(config),
		kinfu_(config)
	{
		//Init Kinfu Tracker
		Eigen::Vector3f volume_size = Vector3f::Constant(vsz/*meters*/);
		//Eigen::Vector3f volume_size = Vector3f::Constant(0.5f/*meters*/);
		//Eigen::Vector3f volume_size = Vector3f(0.7f,2.0f,0.7f);
		kinfu_.volume().setSize(volume_size);

		Eigen::Matrix3f R = Eigen::Matrix3f::Identity();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
		Eigen::Vector3f t = volume_size * 0.5f - Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

		Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);

		kinfu_.setInitalCameraPose(pose);
		kinfu_.volume().setTsdfTruncDist(0.030f/*meters*/);
		kinfu_.setIcpCorespFilteringParams(0.1f/*meters*/, sin(pcl::deg2rad(20.f)));
		kinfu_.setDepthTruncationForICP(1000.f/*meters*/);
		kinfu_.setCameraMovementThreshold(0.001f);




		if (!icp)
			kinfu_.disableIcp();

		//Init KinfuApp            
		tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
		image_view_.raycaster_ptr_ = RayCaster::Ptr(new RayCaster(kinfu_.rows(), kinfu_.cols()));

		if (viz_)
		{
			FocalLen f = config_.getFocalLength();
			scene_cloud_view_.setFocalLengthRGB(f.rgbX, f.rgbY);
			scene_cloud_view_.cloud_viewer_->registerKeyboardCallback(keyboard_callback, (void*)this);
			//scene_cloud_view_.cloud_viewer_->registerMouseCallback(mouse_callback, (void*)this);
			scene_cloud_view_.cloud_viewer_->registerPointPickingCallback(pp_callback, (void*)this);
			image_view_.viewerScene_->registerKeyboardCallback(keyboard_callback, (void*)this);
			image_view_.viewerDepth_->registerKeyboardCallback(keyboard_callback, (void*)this);
			scene_cloud_view_.toggleCube(volume_size);

		}
	}

	~KinFuApp()
	{
		if (evaluation_ptr_)
			evaluation_ptr_->saveAllPoses(kinfu_);
	}

	void
		initCurrentFrameView()
	{
		current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView());
		current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback(keyboard_callback, (void*)this);
		//current_frame_cloud_view_->cloud_viewer_.registerMouseCallback(mouse_callback, (void*)this);
		current_frame_cloud_view_->cloud_viewer_.registerPointPickingCallback(pp_callback, (void*)this);
		current_frame_cloud_view_->setViewerPose(kinfu_.getCameraPose());
	}

	void
		initRegistration()
	{

		registration_ = true;
		kinfu_.setDepthIntrinsics(kinfu_.getFocalLengthRGBx(), kinfu_.getFocalLengthRGBy());
		
	}

	void
		setDepthIntrinsics(std::vector<float> depth_intrinsics)
	{
		float fx = depth_intrinsics[0];
		float fy = depth_intrinsics[1];

		if (depth_intrinsics.size() == 4)
		{
			float cx = depth_intrinsics[2];
			float cy = depth_intrinsics[3];
			kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
			cout << "Depth intrinsics changed to fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << endl;
		}
		else {
			kinfu_.setDepthIntrinsics(fx, fy);
			cout << "Depth intrinsics changed to fx=" << fx << " fy=" << fy << endl;
		}
	}

	void
		toggleColorIntegration()
	{
		if (registration_)
		{
			const int max_color_integration_weight = 2;
			kinfu_.initColorIntegration(max_color_integration_weight);
			integrate_colors_ = true;
		}
		cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration(-r) mode )") << endl;
	}

	void
		enableTextureMapping()
	{
		if (registration_ && integrate_colors_)
		{
			texture_mapping_ = true;
		}
		cout << "Texture mapping: " << (texture_mapping_ ? "On" : "Off ( requires registration(-r) &integrate colors(-ic) mode )") << endl;
	}
	void
		enableTruncationScaling()
	{
		kinfu_.volume().setTsdfTruncDist(kinfu_.volume().getSize()(0) / 100.0f);
	}

	void
		toggleIndependentCamera()
	{
		independent_camera_ = !independent_camera_;
		cout << "Camera mode: " << (independent_camera_ ? "Independent" : "Bound to Kinect pose") << endl;
	}

	void
		toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
	{
		evaluation_ptr_ = Evaluation::Ptr(new Evaluation(eval_folder));
		if (!match_file.empty())
			evaluation_ptr_->setMatchFile(match_file);

		kinfu_.setDepthIntrinsics(evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
		image_view_.raycaster_ptr_ = RayCaster::Ptr(new RayCaster(kinfu_.rows(), kinfu_.cols(),
			evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy));
	}


	void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data)
	{
		bool has_image = false;

		if (has_data)
		{
			depth_device_.upload(opencl_utils::get()->m_command_queue, depth.data, depth.step, depth.rows, depth.cols);
			if (integrate_colors_)
				image_view_.colors_device_.upload(opencl_utils::get()->m_command_queue, rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
			{
				SampledScopeTime fps(time_ms_);

				if (start_)
				{
					if (setinitCamPose == true)
					{
						/** Change volume position **/
						//Eigen::Matrix3f R = Eigen::Matrix3f::Identity();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());

						float roll = 0;
						float yaw = 0;
						float pitch = 0;

						Eigen::AngleAxisf rollAngle(deg2rad(roll), Eigen::Vector3f::UnitZ());
						Eigen::AngleAxisf yawAngle(deg2rad(yaw), Eigen::Vector3f::UnitY());
						Eigen::AngleAxisf pitchAngle(deg2rad(pitch), Eigen::Vector3f::UnitX());

						Eigen::Quaternionf q = rollAngle * yawAngle * pitchAngle;
						Eigen::Matrix3f R = q.toRotationMatrix();

						Eigen::Vector3f volume_size = kinfu_.volume().getSize();// Vector3f::Constant(vsz/*meters*/);

						Eigen::Vector3f t = volume_size * 0.5f - Vector3f(0, abs(sin(deg2rad(pitch))) * (depth.data[depth.cols * depth.cols / 2 + depth.cols / 2] / 1000.f), abs(cos(deg2rad(pitch))) * (depth.data[depth.cols * depth.cols / 2 + depth.cols / 2] / 1000.f/*for mm -> meter*/)/* + volume_size(2) * 0.25f*/);
						cout << "init center T : " << sin(deg2rad(pitch)) << "   " << cos((deg2rad(pitch))) << endl;
						cout << "init center T : " << t << endl;
						Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);
						kinfu_.setInitalCameraPose(pose);
						setinitCamPose = false;

					}
					//cout << "Start" << endl;
					//cout <<  kinfu_.getCameraPose(0).translation() << endl;
					//run kinfu algorithm
					if (integrate_colors_)
						has_image = kinfu_(depth_device_, image_view_.colors_device_);
					else
						has_image = kinfu_(depth_device_);
				}
				else
				{
					cout << "init center depth : " << depth.data[depth.cols * depth.rows / 2 + depth.cols / 2] / 1000.f << "(m)" << endl;
				}
			}

			// process camera pose
			if (pose_processor_)
			{
				pose_processor_->processPose(kinfu_.getCameraPose());
			}

			image_view_.showDepth(depth);
			//image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());

			if (texture_mapping_)
			{
				Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = kinfu_.getCameraPose().linear();
				Vector3f tcurr = kinfu_.getCameraPose().translation();

				float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
				float tnorm = (tcurr - tprev).norm();

				const float alpha = 1.f;
				bool save = (rnorm + alpha * tnorm) / 2 >= 0.001f * 250.0f;

				//Rcurr = Rcurr.transpose();
				//tcurr = -Rcurr * tcurr;

				if (save || kinfu_.getNumberOfPoses() == 1)
				{
					//saving rgb
					char filename[256];
					sprintf(filename, "texturing_sample/%d.png", kinfu_.getNumberOfPoses());

					std::vector<unsigned char> image;
					image.resize(rgb24.rows * rgb24.cols * 4);
					for (unsigned x = 0; x < rgb24.cols; x++)
					{
						for (unsigned y = 0; y < rgb24.rows; y++)
						{
							image[4 * rgb24.cols * y + 4 * x + 0] = rgb24(y, x).r;
							image[4 * rgb24.cols * y + 4 * x + 1] = rgb24(y, x).g;
							image[4 * rgb24.cols * y + 4 * x + 2] = rgb24(y, x).b;
							image[4 * rgb24.cols * y + 4 * x + 3] = 255;
						}
					}
					std::vector<unsigned char> png;
					unsigned error = lodepng::encode(png, image, rgb24.cols, rgb24.rows);
					if (!error) lodepng::save_file(png, filename);
					if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

					//saving poses
					Eigen::Matrix<float, 3, 3, Eigen::RowMajor> RcurrSave = Rcurr.transpose();
					Vector3f tcurrSave = -RcurrSave * tcurr;

					sprintf(filename, "texturing_sample/%d.cam", kinfu_.getNumberOfPoses());
					float fx, fy, cx, cy;
					kinfu_.getDepthIntrinsics(fx, fy, cx, cy);
					ofstream myfile(filename);
					if (myfile.is_open())
					{
						//tcurrSave[0] += 0.00;
						myfile << tcurrSave[0] << " " << tcurrSave[1] << " " << tcurrSave[2] << " ";
						myfile << RcurrSave(0, 0) << " " << RcurrSave(0, 1) << " " << RcurrSave(0, 2) << " ";
						myfile << RcurrSave(1, 0) << " " << RcurrSave(1, 1) << " " << RcurrSave(1, 2) << " ";
						myfile << RcurrSave(2, 0) << " " << RcurrSave(2, 1) << " " << RcurrSave(2, 2) << "\n";
						myfile << 0.8 << " " << 0 << " " << 0 << " " << 1 << " " << 0.5 << " " << 0.5 << "\n"; // this setting is for Kinect I
						myfile.close();
					}
					else cout << "Unable to open file";

					//saving current as previous
					Rprev = Rcurr;
					tprev = tcurr;
				}
			}

		}

		if (doreset_)
		{
			scene_cloud_view_.removeCameraCoordinateSystem(name_);
			setinitCamPose = true;
			doreset_ = false;
			start_ = false;
			kinfu_.setdoreset(true);
			if (texture_mapping_)
			{
				char filename[256];
				for (int i = 0; i<1000; i++)
				{
					sprintf(filename, "texturing_sample/%d.png", i);
					remove(filename);
					sprintf(filename, "texturing_sample/%d.cam", i);
					remove(filename);
				}
			}
		}

		if (texture_mapping_ && doTextureMapping_)
		{
			char sCurrentPath[1024] = { 0, };
			char sCurrentPath1[1024] = { 0, };
			char sCurrentPath2[1024] = { 0, };
			char sPath1[1024] = "\\texturing_sample";
			GetCurrentDirectory(1024, sCurrentPath);
			strcpy(sCurrentPath1, sCurrentPath);
			strcat(sCurrentPath1, sPath1);
			cout << sCurrentPath1 << endl;
			// run russian texturing and then run meshlab sequentially
			SHELLEXECUTEINFO rSEI = { 0 };
			rSEI.cbSize = sizeof(rSEI);
			rSEI.lpVerb = "open";
			//please change to your own path
			rSEI.lpFile = "run.bat";
			rSEI.lpParameters = 0;

			//getcwd(buff, 1024);
			rSEI.lpDirectory = sCurrentPath1;
			rSEI.nShow = SW_NORMAL;
			rSEI.fMask = SEE_MASK_NOCLOSEPROCESS;

			ShellExecuteEx(&rSEI);   // you should check for an error here

			while (TRUE) {
				DWORD nStatus = MsgWaitForMultipleObjects(
					1, &rSEI.hProcess, FALSE,
					INFINITE, QS_ALLINPUT   // drop through on user activity 
					);
				if (nStatus == WAIT_OBJECT_0) {  // done: the program has ended
					char sPath2[1024] = "\\texturing_sample\\texturing_sample.obj";
					strcpy(sCurrentPath2, sCurrentPath);
					strcat(sCurrentPath2, sPath2);
					cout << sCurrentPath2 << endl;
					ShellExecute(GetDesktopWindow(), "open", "meshlab.exe", sCurrentPath2, NULL, SW_SHOWNORMAL);
					break;
				}
				MSG msg;     // else process some messages while waiting...
				while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
					DispatchMessage(&msg);
				}
			}  // launched process has exited

			DWORD dwCode;
			GetExitCodeProcess(rSEI.hProcess, &dwCode);  // ERRORLEVEL value

			doTextureMapping_ = false;
		}

		if (scan_)
		{
			scan_ = false;
			scene_cloud_view_.show(kinfu_, integrate_colors_);

			if (scan_volume_)
			{
				cout << "Downloading TSDF volume from device ... " << flush;
				kinfu_.volume().downloadTsdfAndWeighs(tsdf_volume_.volumeWriteable(), tsdf_volume_.weightsWriteable());
				tsdf_volume_.setHeader(Eigen::Vector3i(pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_.volume().getSize());
				cout << "done [" << tsdf_volume_.size() << " voxels]" << endl << endl;

				cout << "Converting volume to TSDF cloud ... " << flush;
				tsdf_volume_.convertToTsdfCloud(tsdf_cloud_ptr_);
				cout << "done [" << tsdf_cloud_ptr_->size() << " points]" << endl << endl;
			}
			else
				cout << "[!] tsdf volume download is disabled" << endl << endl;

		}

		if (scan_mesh_)
		{
			scan_mesh_ = false;
			scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
		}

		if (has_image)
		{
			Eigen::Affine3f viewer_pose = getViewerPose(*scene_cloud_view_.cloud_viewer_);
			image_view_.showScene(kinfu_, rgb24, registration_, independent_camera_ ? &viewer_pose : 0);

			/*add coordinate */
			Eigen::Affine3f last_pose = kinfu_.getCameraPose();
			//string name = "cam_pose";
			//scene_cloud_view_.drawCameraCoordinateSystem(last_pose, name_);

		}

		if (current_frame_cloud_view_)
			current_frame_cloud_view_->show(kinfu_);

		if (!independent_camera_)
			setViewerPose(*scene_cloud_view_.cloud_viewer_, kinfu_.getCameraPose());
	}

	void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)
	{
		{
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			if (exit_ || !lock)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];
		}
		data_ready_cond_.notify_one();
	}

	void source_cb2_device(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
	{
		{
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			if (exit_ || !lock)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];

			rgb24_.cols = image_wrapper->getWidth();
			rgb24_.rows = image_wrapper->getHeight();
			rgb24_.step = rgb24_.cols * rgb24_.elemSize();

			source_image_data_.resize(rgb24_.cols * rgb24_.rows);
			image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
			rgb24_.data = &source_image_data_[0];
		}
		data_ready_cond_.notify_one();
	}


	void source_cb1_oni(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)
	{
		{
			boost::mutex::scoped_lock lock(data_ready_mutex_);
			if (exit_)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];
		}
		data_ready_cond_.notify_one();
	}

	void source_cb2_oni(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
	{
		{
			boost::mutex::scoped_lock lock(data_ready_mutex_);
			if (exit_)
				return;

			depth_.cols = depth_wrapper->getWidth();
			depth_.rows = depth_wrapper->getHeight();
			depth_.step = depth_.cols * depth_.elemSize();

			source_depth_data_.resize(depth_.cols * depth_.rows);
			depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
			depth_.data = &source_depth_data_[0];

			rgb24_.cols = image_wrapper->getWidth();
			rgb24_.rows = image_wrapper->getHeight();
			rgb24_.step = rgb24_.cols * rgb24_.elemSize();

			source_image_data_.resize(rgb24_.cols * rgb24_.rows);
			image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
			rgb24_.data = &source_image_data_[0];
		}
		data_ready_cond_.notify_one();
	}

	void
		source_cb3(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr & DC3)
	{
		{
			boost::mutex::scoped_try_lock lock(data_ready_mutex_);
			if (exit_ || !lock)
				return;
			int width = DC3->width;
			int height = DC3->height;
			depth_.cols = width;
			depth_.rows = height;
			depth_.step = depth_.cols * depth_.elemSize();
			source_depth_data_.resize(depth_.cols * depth_.rows);

			rgb24_.cols = width;
			rgb24_.rows = height;
			rgb24_.step = rgb24_.cols * rgb24_.elemSize();
			source_image_data_.resize(rgb24_.cols * rgb24_.rows);

			unsigned char *rgb = (unsigned char *)&source_image_data_[0];
			unsigned short *depth = (unsigned short *)&source_depth_data_[0];

			for (int i = 0; i < width*height; i++)
			{
				PointXYZRGBA pt = DC3->at(i);
				rgb[3 * i + 0] = pt.r;
				rgb[3 * i + 1] = pt.g;
				rgb[3 * i + 2] = pt.b;
				depth[i] = pt.z / 0.001;
			}
			rgb24_.data = &source_image_data_[0];
			depth_.data = &source_depth_data_[0];
		}
		data_ready_cond_.notify_one();
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// videoCapture 
	cv::VideoCapture GetvideoCapture(cv::VideoCapture& cap, const unsigned int driverNum, bool need_RGB = false, bool RGB_resigstration = false)
	{

		// select driver using configuration 
		unsigned int driverIdx = 0;

		if (KINECT == config_.getDevice())
		{
			driverIdx = CV_CAP_OPENNI;
		}
		else if (XTION_100 == config_.getDevice() || XTION_606 == config_.getDevice())
		{
			driverIdx = CV_CAP_OPENNI_ASUS;
		}
		else
		{
			driverIdx = CV_CAP_ANY;
		}

		// video capture open.
		cap.open(driverIdx + driverNum);

		if (!cap.isOpened())
		{
			printf("GetvideoCapture : input stream can't open!! driverNum %d\n", driverNum);
			exit(1);
		}
		else
		{
			printf("GetvideoCapture : open! \n");
		}
		if (need_RGB)
		{
			// resolution and hz setting.
			if (config_.getResolution() == VGA && config_.getHz() == _30HZ)
			{
				cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
			}
			else if (config_.getResolution() == SXGA && config_.getHz() == _15HZ)
			{
				cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ);
			}
			else if (config_.getResolution() == SXGA && config_.getHz() == _30HZ)
			{
				cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_30HZ);
			}
			else if (config_.getResolution() == QVGA && config_.getHz() == _30HZ)
			{
				if (config_.getDevice() != KINECT)
				{
					cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_QVGA_30HZ);
				}
				else
				{
					cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
				}
			}
			else if (config_.getResolution() == QVGA && config_.getHz() == _60HZ)
			{
				cap.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_QVGA_60HZ);
			}
		}

		if (RGB_resigstration)
		{
			cap.set(CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION, RGB_resigstration);
		}

		return cap;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	const double p1 = -0.000000007034;
	const double p2 = 0.000006123;
	const double p3 = -0.002088;
	const double p4 = 0.3508;
	const double p5 = -29.54;
	const double p6 = 1060;
	unsigned short *f = new unsigned short[640 * 480];

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// start main loop streo

	void
		startMainLoop_streo(bool triggered_capture)
	{
		cv::VideoCapture cap1;
		cap1.open(0);

		bool scene_view_not_stopped = viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped() : true;
		bool image_view_not_stopped = viz_ ? !image_view_.viewerScene_->wasStopped() : true;
		while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
		{

			cv::Mat img1;
			cv::Mat img2(480, 640, CV_8UC1);
			cap1 >> img1;
			if (!img1.empty())
				cv::cvtColor(img1, img2, CV_BGR2GRAY);

			usb_depth_.cols = 640;
			usb_depth_.rows = 480;
			usb_depth_.step = 1280;

			for (int i = 0; i < 640 * 480; i++)
			{
				f[i] = (unsigned short(10.0 * (p1*pow((double)img2.data[i], 5.0) + p2*pow((double)img2.data[i], 4.0) + p3*pow((double)img2.data[i], 3.0)
					+ p4*pow((double)img2.data[i], 2.0) + p5*(double)img2.data[i] + p6)));
			}

			usb_depth_.data = &f[0];


			bool has_data = true;

			try { this->execute(usb_depth_, rgb24_, has_data); }


			catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
			catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

			if (viz_)
				scene_cloud_view_.cloud_viewer_->spinOnce(3);
		}

		delete[] f;

	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// start main loop defualt
	void
		startMainLoop_openCV(bool triggered_capture)
	{

		cv::Mat depthImg, rgbImg;

		cv::VideoCapture cap;
		GetvideoCapture(cap, 0);		

		bool scene_view_not_stopped = viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped() : true;
		bool image_view_not_stopped = viz_ ? !image_view_.viewerScene_->wasStopped() : true;
		while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
		{			
			if (!cap.grab())
			{
				cout << "cap:: Can not grab images." << endl;				
				continue;
			}			
			
			cap.retrieve(depthImg, CV_CAP_OPENNI_DEPTH_MAP);
			cap.retrieve(rgbImg, CV_CAP_OPENNI_BGR_IMAGE);
			if (config_.getResolution() == SXGA)
			{
				cv::Mat croprgbImg;
				cv::Rect roi(0, 0, 1280, 960);
				croprgbImg = rgbImg(roi);
				rgbImg = croprgbImg;

			}

			if (config_.getResolution() == QVGA && 	config_.getDevice() == KINECT)
			{
				cv::Mat croprgbImg;
				cv::resize(rgbImg, croprgbImg, cv::Size(320, 240), 0, 0, CV_INTER_NN);
				rgbImg = croprgbImg;

				cv::resize(depthImg, croprgbImg, cv::Size(320, 240), 0, 0, CV_INTER_NN);
				depthImg = croprgbImg;

			}

			imshow("Kinect-RGB", rgbImg);
			cv::cvtColor(rgbImg, rgbImg, CV_BGR2RGB);

			depth_.cols = depthImg.cols;
			depth_.rows = depthImg.rows;
			depth_.step = depthImg.cols * depth_.elemSize();
			depth_.data = (unsigned short*)&depthImg.data[0];
			rgb24_.cols = rgbImg.cols;
			rgb24_.rows = rgbImg.rows;
			rgb24_.step = rgbImg.cols * rgb24_.elemSize();
			rgb24_.data = (KinfuTracker::PixelRGB*)&rgbImg.data[0];


			bool has_data = true;
			try { this->execute(depth_, rgb24_, has_data); }


			catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
			catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

			if (viz_)
				scene_cloud_view_.cloud_viewer_->spinOnce(3);
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		startMainLoop(bool triggered_capture)
	{
		using namespace openni_wrapper;
		typedef boost::shared_ptr<DepthImage> DepthImagePtr;
		typedef boost::shared_ptr<Image> ImagePtr;

		boost::function<void(const ImagePtr&, const DepthImagePtr&, float constant)> func1_dev = boost::bind(&KinFuApp::source_cb2_device, this, _1, _2, _3);
		boost::function<void(const DepthImagePtr&)> func2_dev = boost::bind(&KinFuApp::source_cb1_device, this, _1);

		boost::function<void(const ImagePtr&, const DepthImagePtr&, float constant)> func1_oni = boost::bind(&KinFuApp::source_cb2_oni, this, _1, _2, _3);
		boost::function<void(const DepthImagePtr&)> func2_oni = boost::bind(&KinFuApp::source_cb1_oni, this, _1);

		bool is_oni = dynamic_cast<pcl::ONIGrabber*>(&capture_) != 0;
		boost::function<void(const ImagePtr&, const DepthImagePtr&, float constant)> func1 = is_oni ? func1_oni : func1_dev;
		boost::function<void(const DepthImagePtr&)> func2 = is_oni ? func2_oni : func2_dev;

		boost::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&) > func3 = boost::bind(&KinFuApp::source_cb3, this, _1);

		bool need_colors = integrate_colors_ || registration_;
		if (pcd_source_ && !capture_.providesCallback<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)>())
		{
			std::cout << "grabber doesn't provide pcl::PointCloud<pcl::PointXYZRGBA> callback !\n";
		}
		boost::signals2::connection c = pcd_source_ ? capture_.registerCallback(func3) : need_colors ? capture_.registerCallback(func1) : capture_.registerCallback(func2);

		{
			boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

			if (!triggered_capture)
				capture_.start(); // Start stream

			bool scene_view_not_stopped = viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped() : true;
			bool image_view_not_stopped = viz_ ? !image_view_.viewerScene_->wasStopped() : true;
			while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
			{

				if (triggered_capture)
					capture_.start(); // Triggers new frame
				bool has_data = data_ready_cond_.timed_wait(lock, boost::posix_time::millisec(100));

				try { this->execute(depth_, rgb24_, has_data); }

				catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
				catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

				if (viz_)
					scene_cloud_view_.cloud_viewer_->spinOnce(3);
			}

			if (!triggered_capture)
				capture_.stop(); // Stop stream
		}
		c.disconnect();

	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		writeCloud(int format) const
	{
		const SceneCloudView& view = scene_cloud_view_;

		// Points to export are either in cloud_ptr_ or combined_ptr_.
		// If none have points, we have nothing to export.
		if (view.cloud_ptr_->points.empty() && view.combined_ptr_->points.empty())
		{
			cout << "Not writing cloud: Cloud is empty" << endl;
		}
		else
		{
			if (view.point_colors_ptr_->points.empty()) // no colors
			{
				if (view.valid_combined_)
					writeCloudFile(format, view.combined_ptr_);
				else
					writeCloudFile(format, view.cloud_ptr_);
			}
			else
			{
				if (view.valid_combined_)
					writeCloudFile(format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
				else
					writeCloudFile(format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
			}
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		writeMesh(int format) const
	{
		if (scene_cloud_view_.mesh_ptr_)
			writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	void
		printHelp()
	{
		cout << endl;
		cout << "KinFu app hotkeys" << endl;
		cout << "=================" << endl;
		cout << "    H    : print this help" << endl;
		cout << "    S    : Start" << endl;
		cout << "    F    : Finish(Stop)" << endl;
		cout << "    R    : Reset" << endl;
		cout << "   Esc   : exit" << endl;
		cout << "    T    : take cloud" << endl;
		cout << "    A    : take mesh" << endl;
		cout << "    M    : toggle cloud exctraction mode" << endl;
		cout << "    N    : toggle normals exctraction" << endl;
		cout << "    I    : toggle independent camera mode" << endl;
		cout << "    B    : toggle volume bounds" << endl;
		cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
		cout << "    C    : clear clouds" << endl;
		cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
		cout << "    7,8  : save mesh to PLY, VTK" << endl;
		cout << "   X, V  : TSDF volume utility" << endl;
		cout << endl;
	}

	int pickPoint;
	bool exit_;
	bool scan_;
	bool scan_mesh_;
	bool scan_volume_;
	bool doreset_;
	bool start_;
	bool independent_camera_;
	bool doTextureMapping_;
	bool setinitCamPose = true;
	bool registration_;
	bool integrate_colors_;
	bool texture_mapping_;
	bool pcd_source_;
	float focal_length_;
	string name_;
	pcl::Grabber& capture_;
	KinfuTracker kinfu_;

	SceneCloudView scene_cloud_view_;
	ImageView image_view_;
	boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;




	KinfuTracker::DepthMap depth_device_;

	pcl::TSDFVolume<float, short> tsdf_volume_;
	pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

	Evaluation::Ptr evaluation_ptr_;

	boost::mutex data_ready_mutex_;
	boost::condition_variable data_ready_cond_;

	std::vector<KinfuTracker::PixelRGB> source_image_data_;
	std::vector<unsigned short> source_depth_data_;
	std::vector<unsigned short> source_usb_depth_data_;
	PtrStepSz<const unsigned short> depth_;
	PtrStepSz<const unsigned short> usb_depth_;
	PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

	double time_ms_;
	int icp_, viz_;

	boost::shared_ptr<CameraPoseProcessor> pose_processor_;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev; // for saving rgb, pose
	Vector3f tprev;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	static void
		keyboard_callback(const visualization::KeyboardEvent &e, void *cookie)
	{
		KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

		int key = e.getKeyCode();

		if (e.keyUp())
			switch (key)
		{
			case 27: app->exit_ = true; break;
			case (int)'t': case (int)'T': app->scan_ = true; break;
			case (int)'s': case (int)'S': app->start_ = true; break;
			case (int)'f': case (int)'F': app->start_ = false; break;
			case (int)'r': case (int)'R': app->doreset_ = true; break;
			case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
			case (int)'h': case (int)'H': app->printHelp(); break;
			case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExtractionMode(); break;
			case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals(); break;
			case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds(true); break;
			case (int)'i': case (int)'I': app->toggleIndependentCamera(); break;
			case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_.volume().getSize()); break;
			case (int)'7': case (int)'8': app->writeMesh(key - (int)'0'); app->doTextureMapping_ = true; break;
			case (int)'1': case (int)'2': case (int)'3': app->writeCloud(key - (int)'0'); break;
			case '*': app->image_view_.toggleImagePaint(); break;

			case (int)'x': case (int)'X':
				app->scan_volume_ = !app->scan_volume_;
				cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
				break;
			case (int)'v': case (int)'V':
				cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
				app->tsdf_volume_.save("tsdf_volume.dat", true);
				cout << "done [" << app->tsdf_volume_.size() << " voxels]" << endl;
				cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
				pcl::io::savePCDFile<pcl::PointXYZI>("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
				cout << "done [" << app->tsdf_cloud_ptr_->size() << " points]" << endl;
				break;

			default:
				break;
		}
	}

	static void
		mouse_callback(const pcl::visualization::MouseEvent& event, void*)
	{
		if (event.getType() == pcl::visualization::MouseEvent::MouseButtonPress && event.getButton() == pcl::visualization::MouseEvent::LeftButton){
			std::cout << "Mouse : " << event.getX() << ", " << event.getY() << std::endl;
		}
	}

	static void
		pp_callback(const pcl::visualization::PointPickingEvent& event, void* cookie)
	{
		KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

		if (event.getPointIndex() != -1)
		{
			float x, y, z;
			char str[64];

			event.getPoint(x, y, z);
			printf("X: %f , Y: %f, Z: %f\n", x, y, z);
			if (x != 0.f || y != 0.f || z != 0.f)
			{
				app->scene_cloud_view_.showMeshSize(app->kinfu_, (int)(y * VOLUME_Y));
				sprintf(str, "Length : %.2f cm", app->scene_cloud_view_.getLength());
				std::stringstream ss;
				ss.str("");
				ss << "length";
				app->scene_cloud_view_.cloud_viewer_->updateText(str, 440, 10, 20, 34, 135, 246, ss.str());
			}
		}
	}

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
writeCloudFile(int format, const CloudPtr& cloud_prt)
{
	if (format == KinFuApp::PCD_BIN)
	{
		cout << "Saving point cloud to 'cloud_bin.pcd' (binary)... " << flush;
		pcl::io::savePCDFile("cloud_bin.pcd", *cloud_prt, true);
	}
	else
		if (format == KinFuApp::PCD_ASCII)
		{
			cout << "Saving point cloud to 'cloud.pcd' (ASCII)... " << flush;
			pcl::io::savePCDFile("cloud.pcd", *cloud_prt, false);
		}
		else   /* if (format == KinFuApp::PLY) */
		{
			cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
			pcl::io::savePLYFileASCII("cloud.ply", *cloud_prt);

		}
	cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile(int format, const pcl::PolygonMesh& mesh)
{
	if (format == KinFuApp::MESH_PLY)
	{
		cout << "Saving mesh to to 'model_file.ply'... " << flush;
		pcl::io::savePLYFile("texturing_sample\\model_file.ply", mesh);
	}
	else /* if (format == KinFuApp::MESH_VTK) */
	{
		cout << "Saving mesh to to 'model_file.vtk'... " << flush;
		pcl::io::saveVTKFile("model_file.vtk", mesh);
	}
	cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help()
{

	cout << "\nKinFu parameters:" << endl;
	cout << "    --help, -h                              : print this message" << endl;
	cout << "    --registration, -r                      : try to enable registration (source needs to support this)" << endl;
	cout << "    --current-cloud, -cc                    : show current frame cloud" << endl;
	cout << "    --save-views, -sv                       : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;
	cout << "    --integrate-colors, -ic                 : enable color integration mode (allows to get cloud with colors)" << endl;
	cout << "    --texture - mappings, -t                : enable color texture mapping mode" << endl;
	cout << "    --scale-truncation, -st                 : scale the truncation distance and raycaster based on the volume size" << endl;
	cout << "    -volume_size <size_in_meters>           : define integration volume size" << endl;
	cout << "    --depth-intrinsics <fx>,<fy>[,<cx>,<cy> : set the intrinsics of the depth camera" << endl;
	cout << "    -save_pose <pose_file.csv>              : write tracked camera positions to the specified file" << endl;
	cout << "Valid depth data sources:" << endl;
	cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
	cout << "";
	cout << " For RGBD benchmark (Requires OpenCV):" << endl;
	cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;

	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char* argv[])
{

	if (pc::find_switch(argc, argv, "--help") || pc::find_switch(argc, argv, "-h"))
		return print_cli_help();

	int device = 0;
	pc::parse_argument(argc, argv, "-gpu", device);
	pcl::gpu::setDevice(device);
	pcl::gpu::printShortCudaDeviceInfo(device);

	//  if (checkIfPreFermiGPU(device))
	//    return cout << endl << "Kinfu is supported only for Fermi and Kepler arhitectures. It is not even compiled for pre-Fermi by default. Exiting..." << endl, 1;

	boost::shared_ptr<pcl::Grabber> capture;

	bool triggered_capture = false;
	bool pcd_input = false;

	std::string eval_folder, match_file, openni_device, oni_file, pcd_dir;

	
	Config configuation("../../../../../gpu/kinfu_opencl/Config.txt");
	//Config configuation(XTION_100, VGA, _30HZ );		
	//Config configuation(XTION_100, QVGA, _30HZ );	
	//Config configuation(KINECT, VGA, _30HZ );
	//Config configuation(KINECT, SXGA, _15HZ);		
	//Config configuation(COMPACT_STREO, VGA, _30HZ );	

	if (!configuation.isVaildConfig())
	{
		cout << "[ERROR] Configuration is not valid" << endl;
		return 1;
	}

	if (configuation.getDevice() == XTION_100 || configuation.getDevice() == XTION_606)
	{
		try
		{
			if (configuation.getResolution() == VGA)
			{
				capture.reset(new pcl::OpenNIGrabber());
			}
			else if (configuation.getResolution() == QVGA)
			{
				if (configuation.getHz() == _60HZ)
					capture.reset(new pcl::OpenNIGrabber("", pcl::OpenNIGrabber::OpenNI_QVGA_60Hz, pcl::OpenNIGrabber::OpenNI_QVGA_60Hz));
				else if (configuation.getHz() == _30HZ)
					capture.reset(new pcl::OpenNIGrabber("", pcl::OpenNIGrabber::OpenNI_QVGA_30Hz, pcl::OpenNIGrabber::OpenNI_QVGA_30Hz));

			}
		}
		catch (const pcl::PCLException& /*e*/) { return cout << "Can't open depth source" << endl, -1; }
	}
	
	/*float volume_size = 0.7f;*/
	float volume_size = 0.4f;
	pc::parse_argument(argc, argv, "-volume_size", volume_size);

	int icp = 1, visualization = 1;
	std::vector<float> depth_intrinsics;
	pc::parse_argument(argc, argv, "--icp", icp);
	pc::parse_argument(argc, argv, "--viz", visualization);

	std::string camera_pose_file;
	boost::shared_ptr<CameraPoseProcessor> pose_processor;
	if (pc::parse_argument(argc, argv, "-save_pose", camera_pose_file) && camera_pose_file.size() > 0)
	{
		pose_processor.reset(new CameraPoseWriter(camera_pose_file));
	}

	KinFuApp app(*capture, volume_size, icp, visualization, configuation, pose_processor);

	if (pc::parse_argument(argc, argv, "-eval", eval_folder) > 0)
		app.toggleEvaluationMode(eval_folder, match_file);

	if (pc::find_switch(argc, argv, "--current-cloud") || pc::find_switch(argc, argv, "-cc"))
		app.initCurrentFrameView();

	if (pc::find_switch(argc, argv, "--save-views") || pc::find_switch(argc, argv, "-sv"))
		app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  

	if (pc::find_switch(argc, argv, "--registration") || pc::find_switch(argc, argv, "-r"))
	{
		if (pcd_input)
		{
			app.pcd_source_ = true;
			app.registration_ = true; // since pcd provides registered rgbd
		}
		else
		{
			app.initRegistration();
		}
	}
	if (pc::find_switch(argc, argv, "--integrate-colors") || pc::find_switch(argc, argv, "-ic"))
		app.toggleColorIntegration();

	if (pc::find_switch(argc, argv, "--scale-truncation") || pc::find_switch(argc, argv, "-st"))
		app.enableTruncationScaling();

	if (pc::find_switch(argc, argv, "--texture-mapping") || pc::find_switch(argc, argv, "-t"))
		app.enableTextureMapping();
	if (pc::parse_x_arguments(argc, argv, "--depth-intrinsics", depth_intrinsics) > 0)
	{
		if ((depth_intrinsics.size() == 4) || (depth_intrinsics.size() == 2))
		{
			app.setDepthIntrinsics(depth_intrinsics);
		}
		else
		{
			pc::print_error("Depth intrinsics must be given on the form fx,fy[,cx,cy].\n");
			return -1;
		}
	}

	void(KinFuApp::*pStartMainLoop)(bool triggered_capture);



	// executing
	if (configuation.getDevice() == COMPACT_STREO)
	{
		pStartMainLoop = &KinFuApp::startMainLoop_streo;
	}
	else if (configuation.getDevice() == XTION_100 || configuation.getDevice() == XTION_606)
	{
		pStartMainLoop = &KinFuApp::startMainLoop;
	}
	else
	{
		pStartMainLoop = &KinFuApp::startMainLoop_openCV;
	}

	try { (app.*pStartMainLoop)(triggered_capture); }
	catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
	catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
	catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

#ifdef HAVE_OPENCV
	for (size_t t = 0; t < app.image_view_.views_.size(); ++t)
	{
		if (t == 0)
		{
			cout << "Saving depth map of first view." << endl;
			cv::imwrite("./depthmap_1stview.png", app.image_view_.views_[0]);
			cout << "Saving sequence of (" << app.image_view_.views_.size() << ") views." << endl;
		}
		char buf[4096];
		sprintf(buf, "./%06d.png", (int)t);
		cv::imwrite(buf, app.image_view_.views_[t]);
		printf("writing: %s\n", buf);
	}
#endif

	return 0;
}
