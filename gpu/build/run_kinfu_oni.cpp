/*
 * run_kinfu_oni.cpp
 * Standalone KinectFusion runner using synthetic depth data.
 *
 * OpenNI2 is not required. Depth frames are generated synthetically
 * (a virtual sphere seen from a slowly panning camera) to exercise the
 * full KinFu pipeline and produce a reconstructed mesh.
 *
 * Usage:
 *   ./run_kinfu_oni              -- synthetic mode (default)
 *   ./run_kinfu_oni --frames N   -- override number of frames (default 200)
 *
 * Output:
 *   output.stl   -- ASCII STL mesh of the reconstructed scene
 */

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
// kinfu.h already includes opencl_utils.h and kinfu_config.h

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Synthetic depth generator
// Renders a hemisphere (sphere of radius 1.5 m, center at (0,0,2.5 m) in
// world frame) onto a VGA depth image.  camera_tx_m pans the camera
// along X, which creates inter-frame variation that ICP can track.
// ---------------------------------------------------------------------------
static void gen_sphere_depth(std::vector<unsigned short>& buf,
                              int rows, int cols,
                              float camera_tx_m)
{
    const float fx = 585.f, fy = 585.f;
    const float cx_i = cols * 0.5f;
    const float cy_i = rows * 0.5f;

    // Sphere center in camera coordinates (camera panning right → sphere moves left)
    const float scx = -camera_tx_m;
    const float scy =  0.f;
    const float scz =  2.5f;   // 2.5 m forward
    const float R   =  1.5f;   // 1.5 m radius

    buf.assign((size_t)rows * cols, 0);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float dx = (c - cx_i) / fx;
            float dy = (r - cy_i) / fy;
            // Ray: origin=(0,0,0), dir=(dx, dy, 1.0)

            // |t*(dx,dy,1) - (scx,scy,scz)|^2 = R^2
            float a    = dx*dx + dy*dy + 1.f;
            float b    = -2.f * (dx*scx + dy*scy + scz);
            float cval = scx*scx + scy*scy + scz*scz - R*R;
            float disc = b*b - 4.f*a*cval;

            if (disc >= 0.f) {
                float t = (-b - sqrtf(disc)) / (2.f * a);
                if (t > 0.1f) {
                    float depth_mm = t * 1000.f;
                    if (depth_mm > 0.f && depth_mm < 65535.f)
                        buf[r * cols + c] = (unsigned short)(depth_mm + 0.5f);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int num_frames = 200;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            num_frames = atoi(argv[++i]);
    }

    printf("============================================================\n");
    printf(" KinectFusion OpenCL – synthetic depth test\n");
    printf(" Frames : %d\n", num_frames);
    printf("============================================================\n");

    // ------------------------------------------------------------------
    // 1. Create KinfuTracker  (this also initialises the OpenCL singleton)
    // ------------------------------------------------------------------
    Config config(KINECT, VGA, _30HZ, /*levels=*/3, /*iters=*/NULL,
                  /*min_delta=*/1e-4, /*cl_device=*/NVIDIA);

    pcl::gpu::KinfuTracker tracker(config);

    // ------------------------------------------------------------------
    // 2. Obtain the OpenCL context created by the tracker
    // ------------------------------------------------------------------
    opencl_utils* cl = opencl_utils::get();
    if (!cl || !cl->m_context) {
        fprintf(stderr, "ERROR: OpenCL not initialised.\n");
        return 1;
    }

    const int rows = config.getRows(); // 480
    const int cols = config.getCols(); // 640

    printf("Depth resolution : %d x %d\n", cols, rows);

    // ------------------------------------------------------------------
    // 3. Allocate the DepthMap on the device
    // ------------------------------------------------------------------
    pcl::gpu::KinfuTracker::DepthMap depth_map(cl->m_context);
    depth_map.create(rows, cols);

    // ------------------------------------------------------------------
    // 4. Main loop – process synthetic frames
    // ------------------------------------------------------------------
    std::vector<unsigned short> host_depth;
    int tracked = 0, lost = 0;

    // Camera pans 10 cm over all frames (slow enough for ICP to track)
    const float pan_total_m = 0.10f;

    for (int i = 0; i < num_frames; i++) {
        float tx = pan_total_m * i / (float)(num_frames - 1);
        gen_sphere_depth(host_depth, rows, cols, tx);

        // Upload to device (stride=0 means tight/contiguous)
        depth_map.upload(cl->m_command_queue,
                         host_depth.data(),
                         /*stride=*/0, cols, rows);

        bool ok = tracker(depth_map);
        if (ok) ++tracked; else ++lost;

        if (i % 20 == 0 || i == num_frames - 1)
            printf("Frame %4d / %d  tracking=%s\n",
                   i, num_frames, ok ? "OK" : "LOST");
    }

    printf("\nTracking summary: %d OK / %d lost\n", tracked, lost);

    // ------------------------------------------------------------------
    // 5. Extract mesh via MarchingCubes and write PLY directly
    // ------------------------------------------------------------------
    printf("Extracting mesh (marching cubes)...\n");

    pcl::gpu::MarchingCubes mc;
    pcl::gpu::CLDeviceArray<pcl::PointXYZ> tri_buf;
    pcl::gpu::CLDeviceArray<cl_float> tri_verts = mc.run(tracker.volume(), tri_buf);

    size_t n_floats = tri_verts.size();   // totalTriangles * 18 floats
    printf("  float count : %zu  triangles : %zu\n", n_floats, n_floats / 18);

    if (n_floats > 0) {
        std::vector<float> host(n_floats);
        tri_verts.download(cl->m_command_queue, host.data());
        clFinish(cl->m_command_queue);

        // voxel size in metres
        Eigen::Vector3f vsz = tracker.volume().getVoxelSize();

        const char* out_path = "output.stl";
        FILE* fp = fopen(out_path, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }

        // MarchingCubes buffer layout (marchingcube.cl):
        //   18 floats per triangle = 3 vertices × [vx,vy,vz, nx,ny,nz]
        //   positions are voxel coords → multiply by voxel size for metres
        //   normals are TSDF gradient vectors → normalize before writing
        int n_triangles = (int)(n_floats / 18);
        int n_vertices  = n_triangles * 3;
        fprintf(fp, "ply\nformat ascii 1.0\n");
        fprintf(fp, "element vertex %d\n", n_vertices);
        fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
        fprintf(fp, "property float nx\nproperty float ny\nproperty float nz\n");
        fprintf(fp, "element face %d\n", n_triangles);
        fprintf(fp, "property list uchar int vertex_index\n");
        fprintf(fp, "end_header\n");

        for (int t = 0; t < n_triangles; ++t) {
            for (int v = 0; v < 3; ++v) {
                int base = (t * 3 + v) * 6;  // stride 6 floats per vertex
                float x  = host[base + 0] * vsz[0];
                float y  = host[base + 1] * vsz[1];
                float z  = host[base + 2] * vsz[2];
                float nx = host[base + 3];
                float ny = host[base + 4];
                float nz = host[base + 5];
                float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
                if (nlen > 0.f) { nx /= nlen; ny /= nlen; nz /= nlen; }
                fprintf(fp, "%f %f %f %f %f %f\n", x, y, z, nx, ny, nz);
            }
        }
        for (int t = 0; t < n_triangles; ++t)
            fprintf(fp, "3 %d %d %d\n", t*3, t*3+1, t*3+2);

        fclose(fp);
        printf("Saved: %s  (%d triangles)\n", out_path, n_triangles);
    } else {
        printf("No triangles extracted.\n");
    }

    return 0;
}
