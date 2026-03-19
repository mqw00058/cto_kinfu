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

#include <pcl/gpu/kinfu/color_volume.h>
#include <pcl/gpu/kinfu/tsdf_volume.h>
#include "internal.h"
#include <algorithm>
#include <Eigen/Core>

using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
using pcl::device::device_cast;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::ColorVolume::ColorVolume(const TsdfVolume& tsdf, cl_color_volume* clColorVolume, int max_weight) : color_volume_(opencl_utils::get()->m_context), resolution_(tsdf.getResolution()), volume_size_(tsdf.getSize()), max_weight_(1), clColorVolume_(clColorVolume)
{
  max_weight_ = max_weight < 0 ? max_weight_ : max_weight;
  max_weight_ = max_weight_ > 255 ? 255 : max_weight_;

  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  color_volume_.create (volume_y * volume_z, volume_x);
  reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::ColorVolume::~ColorVolume()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::ColorVolume::reset()
{
	int volume[3] = { resolution_(0), resolution_(1), resolution_(2) };
	clColorVolume_->initColorVolume(color_volume_, volume);//device::initColorVolume(color_volume_);// need implement
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
pcl::gpu::ColorVolume::getMaxWeight() const
{
  return max_weight_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CLDeviceArray2D<int>
pcl::gpu::ColorVolume::data() const
{
  return color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::ColorVolume::fetchColors(const CLDeviceArray<PointType>& cloud, CLDeviceArray<RGB>& colors) const
{  
  colors.create(cloud.size());
  CLDeviceArray<float4>& c = (CLDeviceArray<float4>&)cloud;
  CLDeviceArray<uchar4>& cr = (CLDeviceArray<uchar4>&)colors;
  clColorVolume_->extractColors(color_volume_, device_cast<const float3> (volume_size_), c, cr/*bgra*/);
  //device::exctractColors(color_volume_, device_cast<const float3> (volume_size_), cloud, (uchar4*)colors.ptr()/*bgra*/);// need implement
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////