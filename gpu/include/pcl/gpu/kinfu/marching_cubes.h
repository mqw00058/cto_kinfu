#ifndef PCL_KINFU_TSDF_MARCHING_CUBES_H_
#define PCL_KINFU_TSDF_MARCHING_CUBES_H_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array_cl.h>
#include <Eigen/Core>
#include <vector>

#define NO_POST_DENOISING
//#define CONNECTED_COMPONENTS_DENOISING_2D
//#define CONNECTED_COMPONENTS_DENOISING_3D


namespace pcl
{
	namespace gpu
	{
		class TsdfVolume;

		/** \brief MarchingCubes implements MarchingCubes functionality for TSDF volume on GPU
		  * \author Seung-Yong Woo, LG elentoronics Ltd, (sy.woo@lge.com)
		  */
		class PCL_EXPORTS MarchingCubes
		{
		public:

			/** \brief Default size for triangles buffer */
			enum
			{
				POINTS_PER_TRIANGLE = 3,
				DEFAULT_TRIANGLES_BUFFER_SIZE = 2 * 1000 * 1000 * POINTS_PER_TRIANGLE
			};

			/** \brief Point type. */
			typedef pcl::PointXYZ PointType;

			/** \brief Smart pointer. */
			typedef boost::shared_ptr<MarchingCubes> Ptr;

			/** \brief Default constructor */
			MarchingCubes();

			/** \brief Destructor */
			~MarchingCubes();

			/** \brief Runs marching cubes triangulation.
				* \param[in] tsdf
				* \param[in] triangles_buffer Buffer for triangles. Its size determines max extracted triangles. If empty, it will be allocated with default size will be used.
				* \return Array with triangles. Each 3 consequent poits belond to a single triangle. The returned array points to 'triangles_buffer' data.
				*/
			CLDeviceArray<cl_float>
				run(const TsdfVolume& tsdf_volume, CLDeviceArray<PointType>& triangles_buffer);

			void init();
		private:
			opencl_utils *clData;

			cl_program program;
			cl_kernel constructHPLevelKernel;
			cl_kernel classifyCubesKernel;
			cl_kernel traverseHPKernel;
			cl_kernel constructHPLevelCharCharKernel;
			cl_kernel constructHPLevelCharShortKernel;
			cl_kernel constructHPLevelShortShortKernel;
			cl_kernel constructHPLevelShortIntKernel;

			std::vector<cl_mem> buffers;
			cl_mem cubeIndexesBuffer;
			cl_mem cubeIndexesImage;
			cl_mem volumeDataImage;
			CLDeviceArray<cl_float> verticesBuffer;

			int isolevel;

			void convertVolume(const TsdfVolume& tsdf_volume);
			void histoPyramidConstruction();
			void updateScalarField();
			void histoPyramidTraversal(int sum);

#ifdef CONNECTED_COMPONENTS_DENOISING_2D
			int ConnectedComponentLabeling(const TsdfVolume& tsdf_volume, unsigned char** bitmap, int** labelmap, int** labelmapcounter);
			void Tracer(int *cy, int *cx, int *tracingdirection, unsigned char** bitmap, int** labelmap);
			void ContourTracing(int cy, int cx, int labelindex, int tracingdirection, unsigned char** bitmap, int** labelmap);
#endif
		};
	}
}

#endif /* PCL_KINFU_MARCHING_CUBES_H_ */
