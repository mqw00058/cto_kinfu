/*
 * point_types.h - minimal PCL point types stub for standalone build.
 */
#ifndef PCL_POINT_TYPES_H_
#define PCL_POINT_TYPES_H_

#include <pcl/pcl_macros.h>

namespace pcl
{
  /** \brief A point structure representing Euclidean xyz coordinates. */
  struct PCL_EXPORTS PointXYZ
  {
    union {
      float data[4];
      struct { float x, y, z; };
    };
    PointXYZ() : x(0.0f), y(0.0f), z(0.0f) { data[3] = 0.0f; }
    PointXYZ(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { data[3] = 0.0f; }
  };

  /** \brief A point structure representing normal coordinates and the surface curvature estimate. */
  struct PCL_EXPORTS Normal
  {
    union {
      float data_n[4];
      struct { float normal_x, normal_y, normal_z; float curvature; };
    };
    Normal() : normal_x(0.0f), normal_y(0.0f), normal_z(0.0f), curvature(0.0f) {}
  };

  /** \brief A point structure representing Euclidean xyz coordinates, padded with an extra float. */
  typedef PointXYZ PointNormal;

  struct PCL_EXPORTS RGB
  {
    union {
      struct { unsigned char b, g, r, a; };
      float rgb;
      unsigned int rgba;
    };
    RGB() : r(0), g(0), b(0), a(0) {}
  };

} // namespace pcl

#endif // PCL_POINT_TYPES_H_
