/*
 * point_cloud.h - minimal PCL point cloud stub for standalone build.
 */
#ifndef PCL_POINT_CLOUD_H_
#define PCL_POINT_CLOUD_H_

#include <pcl/pcl_macros.h>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace pcl
{
  /** \brief PointCloud represents the base class in PCL for storing collections of 3D points. */
  template <typename PointT>
  class PCL_EXPORTS PointCloud
  {
  public:
    typedef boost::shared_ptr<PointCloud<PointT> > Ptr;
    typedef boost::shared_ptr<const PointCloud<PointT> > ConstPtr;

    typedef std::vector<PointT> VectorType;
    typedef PointT value_type;

    std::vector<PointT> points;
    unsigned int width;
    unsigned int height;
    bool is_dense;
    float sensor_origin_[4];
    float sensor_orientation_[4];

    PointCloud()
      : width(0), height(0), is_dense(true)
    {
      for (int i = 0; i < 4; ++i) { sensor_origin_[i] = 0.0f; sensor_orientation_[i] = 0.0f; }
    }

    inline std::size_t size() const { return points.size(); }
    inline bool empty() const { return points.empty(); }

    inline void resize(std::size_t n) { points.resize(n); width = (unsigned int)n; height = 1; }

    inline PointT& operator[](std::size_t n) { return points[n]; }
    inline const PointT& operator[](std::size_t n) const { return points[n]; }

    inline void push_back(const PointT& pt) { points.push_back(pt); width = (unsigned int)points.size(); height = 1; }
    inline void clear() { points.clear(); width = 0; height = 0; }

    typedef typename VectorType::iterator iterator;
    typedef typename VectorType::const_iterator const_iterator;
    inline iterator begin() { return points.begin(); }
    inline iterator end()   { return points.end(); }
    inline const_iterator begin() const { return points.begin(); }
    inline const_iterator end()   const { return points.end(); }
  };

} // namespace pcl

#endif // PCL_POINT_CLOUD_H_
