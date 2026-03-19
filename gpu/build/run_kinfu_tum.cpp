/*
 * run_kinfu_tum.cpp
 * KinectFusion runner for TUM RGB-D benchmark sequences.
 *
 * Reads depth PNG images listed in depth.txt (TUM format):
 *   # timestamp filename
 *   1305032180.405639 depth/1305032180.405639.png
 *   ...
 *
 * Depth PNG encoding: uint16, value = depth_meters * 5000
 *   → divide by 5 to get millimetres for KinFu.
 *
 * Usage:
 *   ./run_kinfu_tum <dataset_dir> [--config FILE] [--frames N] [--skip K] [--no-gt]
 *
 *   dataset_dir  path to extracted TUM sequence folder
 *                (must contain depth.txt and depth/ subfolder)
 *   --config     path to kinfu config file
 *   --depth-txt  path to depth list file (default: <dataset_dir>/depth.txt)
 *                use this to pass depth_masked.txt from mask_teddy_depth.py
 *   --frames N   process only first N depth frames (default: all)
 *   --skip K     skip every K-1 frames, process every K-th (default: 1)
 *   --no-gt      disable ground-truth pose hint (pure ICP, may lose tracking)
 *
 * Ground truth (groundtruth.txt in dataset_dir):
 *   If present and --no-gt not set, the nearest GT pose is used as an
 *   orientation hint for ICP each frame, fixing fast-rotation tracking loss.
 *
 * Output:
 *   output_tum.ply  ASCII PLY mesh of the reconstructed scene
 *
 * TUM freiburg1 camera intrinsics (used here):
 *   fx = 517.3,  fy = 516.5,  cx = 318.6,  cy = 255.3
 *   depth scale: 5000 (raw / 5000 = metres)
 */

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>

#include <png.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Load a 16-bit grayscale PNG into buf (row-major, width*height elements).
// Returns true on success.
// ---------------------------------------------------------------------------
static bool load_depth_png(const char* path,
                            std::vector<unsigned short>& buf,
                            int& out_width, int& out_height)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return false;
    }

    unsigned char sig[8];
    if (fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8)) {
        fprintf(stderr, "Not a PNG: %s\n", path);
        fclose(fp);
        return false;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                             NULL, NULL, NULL);
    if (!png) { fclose(fp); return false; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int width      = (int)png_get_image_width(png, info);
    int height     = (int)png_get_image_height(png, info);
    int color_type = png_get_color_type(png, info);
    int bit_depth  = png_get_bit_depth(png, info);

    if (bit_depth != 16 || color_type != PNG_COLOR_TYPE_GRAY) {
        fprintf(stderr, "Expected 16-bit grayscale PNG: %s (got depth=%d type=%d)\n",
                path, bit_depth, color_type);
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return false;
    }

    // On little-endian systems we need big-endian swap (PNG is big-endian)
    png_set_swap(png);

    png_read_update_info(png, info);

    buf.resize((size_t)width * height);
    std::vector<png_bytep> rows(height);
    for (int r = 0; r < height; ++r)
        rows[r] = (png_bytep)(&buf[r * width]);

    png_read_image(png, rows.data());
    png_read_end(png, NULL);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    out_width  = width;
    out_height = height;
    return true;
}

// ---------------------------------------------------------------------------
// Depth entry: timestamp + relative file path
// ---------------------------------------------------------------------------
struct DepthEntry {
    double      timestamp;
    std::string filename;
};

// Parse depth.txt — return list sorted by timestamp.
// Lines starting with '#' are skipped.
static std::vector<DepthEntry> parse_depth_txt(const char* path)
{
    std::vector<DepthEntry> entries;
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open depth.txt: %s\n", path);
        return entries;
    }
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double ts;
        char fname[256];
        if (sscanf(line, "%lf %255s", &ts, fname) == 2) {
            DepthEntry e;
            e.timestamp = ts;
            e.filename  = std::string(fname);
            entries.push_back(e);
        }
    }
    fclose(fp);
    std::sort(entries.begin(), entries.end(),
              [](const DepthEntry& a, const DepthEntry& b){ return a.timestamp < b.timestamp; });
    return entries;
}

// ---------------------------------------------------------------------------
// Ground truth entry: timestamp + pose (translation + quaternion)
// ---------------------------------------------------------------------------
struct GTEntry {
    double timestamp;
    float  tx, ty, tz;
    float  qx, qy, qz, qw;
};

// Parse groundtruth.txt — format: timestamp tx ty tz qx qy qz qw
static std::vector<GTEntry> parse_groundtruth(const char* path)
{
    std::vector<GTEntry> entries;
    FILE* fp = fopen(path, "r");
    if (!fp) return entries;   // not fatal — GT is optional

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        GTEntry e;
        if (sscanf(line, "%lf %f %f %f %f %f %f %f",
                   &e.timestamp,
                   &e.tx, &e.ty, &e.tz,
                   &e.qx, &e.qy, &e.qz, &e.qw) == 8) {
            entries.push_back(e);
        }
    }
    fclose(fp);
    std::sort(entries.begin(), entries.end(),
              [](const GTEntry& a, const GTEntry& b){ return a.timestamp < b.timestamp; });
    return entries;
}

// Find the ground truth entry with the nearest timestamp.
// Returns false if entries is empty or the nearest match is further than max_dt.
static bool find_nearest_gt(const std::vector<GTEntry>& gt, double ts,
                             GTEntry& out, double max_dt = 0.1)
{
    if (gt.empty()) return false;

    // Binary search for the first entry >= ts
    size_t lo = 0, hi = gt.size();
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (gt[mid].timestamp < ts) lo = mid + 1;
        else                        hi = mid;
    }

    // lo is the first index with timestamp >= ts; check lo and lo-1
    double best_dt = 1e18;
    size_t best_idx = 0;
    if (lo < gt.size()) {
        double dt = std::abs(gt[lo].timestamp - ts);
        if (dt < best_dt) { best_dt = dt; best_idx = lo; }
    }
    if (lo > 0) {
        double dt = std::abs(gt[lo-1].timestamp - ts);
        if (dt < best_dt) { best_dt = dt; best_idx = lo-1; }
    }

    if (best_dt > max_dt) return false;
    out = gt[best_idx];
    return true;
}

// Convert a GT entry (translation + quaternion) to Eigen::Affine3f
static Eigen::Affine3f gt_to_affine(const GTEntry& g)
{
    Eigen::Quaternionf q(g.qw, g.qx, g.qy, g.qz);
    q.normalize();
    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    pose.rotate(q.toRotationMatrix());
    pose.translation() << g.tx, g.ty, g.tz;
    return pose;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // --- Parse arguments ---
    const char* dataset_dir = NULL;
    const char* config_file = NULL;
    const char* depth_txt_override = NULL;
    int max_frames = -1;   // -1 = all
    int skip       =  1;   // process every skip-th frame
    bool use_gt    = true; // use groundtruth.txt pose hints

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            max_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip") == 0 && i + 1 < argc)
            skip = atoi(argv[++i]);
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc)
            config_file = argv[++i];
        else if (strcmp(argv[i], "--depth-txt") == 0 && i + 1 < argc)
            depth_txt_override = argv[++i];
        else if (strcmp(argv[i], "--no-gt") == 0)
            use_gt = false;
        else if (argv[i][0] != '-')
            dataset_dir = argv[i];
    }

    if (!dataset_dir) {
        fprintf(stderr, "Usage: %s <dataset_dir> [--config FILE] [--depth-txt FILE] [--frames N] [--skip K] [--no-gt]\n", argv[0]);
        fprintf(stderr, "  dataset_dir : path to TUM sequence (contains depth.txt)\n");
        fprintf(stderr, "  --config    : path to kinfu config file\n");
        fprintf(stderr, "  --depth-txt : override depth list file (e.g. depth_masked.txt)\n");
        fprintf(stderr, "  --no-gt     : disable ground-truth pose hints (pure ICP)\n");
        return 1;
    }
    if (skip < 1) skip = 1;

    // --- Locate depth.txt (or override) ---
    std::string depth_txt = depth_txt_override
        ? std::string(depth_txt_override)
        : std::string(dataset_dir) + "/depth.txt";
    std::vector<DepthEntry> depth_entries = parse_depth_txt(depth_txt.c_str());

    if (depth_entries.empty()) {
        fprintf(stderr, "No depth frames found in %s\n", depth_txt.c_str());
        return 1;
    }

    int total = (int)depth_entries.size();
    if (max_frames > 0 && max_frames < total) total = max_frames;

    // --- Load ground truth (optional) ---
    std::string gt_txt = std::string(dataset_dir) + "/groundtruth.txt";
    std::vector<GTEntry> gt_entries = parse_groundtruth(gt_txt.c_str());
    if (!use_gt || gt_entries.empty()) {
        use_gt = false;
        printf("Ground truth : not used%s\n",
               gt_entries.empty() ? " (groundtruth.txt not found)" : " (--no-gt)");
    } else {
        printf("Ground truth : %zu poses loaded from %s\n", gt_entries.size(), gt_txt.c_str());
    }

    printf("============================================================\n");
    printf(" KinectFusion OpenCL – TUM RGB-D runner\n");
    printf(" Dataset : %s\n", dataset_dir);
    printf(" Frames  : %d  (skip=%d, effective=%.0f fps)\n",
           total, skip, 30.0 / skip);
    printf(" GT hint : %s\n", use_gt ? "enabled" : "disabled");
    printf("============================================================\n");

    // --- Create KinfuTracker ---
    Config config;
    if (config_file) {
        printf("Loading config: %s\n", config_file);
        Config tmp(const_cast<char*>(config_file));
        config = tmp;
    } else {
        config.setConfig(KINECT, VGA, _30HZ, /*levels=*/3, /*iters=*/NULL,
                         /*min_delta=*/1e-4, /*cl_device=*/NVIDIA);
    }

    // Always override focal lengths for TUM freiburg1 camera
    // (fx=517.3, fy=516.5 regardless of config DEVICE setting)
    config.focalLength_.depthX = 517.3f;
    config.focalLength_.depthY = 516.5f;
    config.focalLength_.rgbX   = 517.3f;
    config.focalLength_.rgbY   = 516.5f;

    pcl::gpu::KinfuTracker tracker(config);

    // When ground truth is available, disable ICP and inject GT poses directly.
    // The hint mechanism's rotation clamp (0.15 rad ≈ 8.6°) is too tight for
    // fast-motion sequences (freiburg1_teddy rotates ~15-20°/frame), so we
    // bypass ICP entirely and use GT translation+rotation for TSDF integration.
    if (use_gt)
        tracker.disableIcp();

    // KinFu's initial camera pose in its own world frame: {R=I, t=init_tcam}.
    // For VOLUME_SIZE=3m, init_tcam ≈ (1.5, 1.5, -0.3).
    // We need to map GT poses (in the TUM recording world frame) into KinFu's
    // world frame by computing:
    //   T_kinfu_i = T_kinfu_0 * T_gt_0^{-1} * T_gt_i
    // This ensures frame-0 TSDF (always integrated with T_kinfu_0) and all
    // subsequent frames are consistent in KinFu's world frame.
    Eigen::Affine3f T_kinfu_0 = tracker.getCameraPose(0); // {I, init_tcam}
    Eigen::Affine3f T_gt_0     = Eigen::Affine3f::Identity();
    Eigen::Affine3f T_gt_0_inv = Eigen::Affine3f::Identity();
    bool gt_origin_set = false;
    if (use_gt && !depth_entries.empty()) {
        GTEntry gt0;
        if (find_nearest_gt(gt_entries, depth_entries[0].timestamp, gt0)) {
            T_gt_0     = gt_to_affine(gt0);
            T_gt_0_inv = T_gt_0.inverse();
            gt_origin_set = true;
            printf("GT origin   : ts=%.6f  t=(%.3f, %.3f, %.3f)\n",
                   gt0.timestamp, gt0.tx, gt0.ty, gt0.tz);
        } else {
            printf("WARNING: no GT pose found near first depth frame ts=%.6f\n",
                   depth_entries[0].timestamp);
        }
    }

    // --- Get OpenCL context ---
    opencl_utils* cl = opencl_utils::get();
    if (!cl || !cl->m_context) {
        fprintf(stderr, "ERROR: OpenCL not initialised.\n");
        return 1;
    }

    const int rows = config.getRows(); // 480
    const int cols = config.getCols(); // 640

    printf("Depth resolution : %d x %d\n", cols, rows);
    printf("Focal length     : fx=%.1f  fy=%.1f\n",
           config.focalLength_.depthX, config.focalLength_.depthY);

    // --- Allocate depth map on device ---
    pcl::gpu::KinfuTracker::DepthMap depth_map(cl->m_context);
    depth_map.create(rows, cols);

    // --- Main loop ---
    std::vector<unsigned short> host_depth;
    int tracked = 0, lost = 0, loaded = 0;

    for (int i = 0; i < total; i += skip) {
        const DepthEntry& de = depth_entries[i];
        std::string fpath = std::string(dataset_dir) + "/" + de.filename;

        int w = 0, h = 0;
        if (!load_depth_png(fpath.c_str(), host_depth, w, h)) {
            fprintf(stderr, "Skipping unreadable frame: %s\n", fpath.c_str());
            continue;
        }

        if (w != cols || h != rows) {
            fprintf(stderr, "Frame size mismatch (%dx%d vs expected %dx%d): %s\n",
                    w, h, cols, rows, fpath.c_str());
            continue;
        }

        // TUM depth: raw / 5000 = metres → raw / 5 = millimetres
        for (auto& v : host_depth)
            v = (unsigned short)(v / 5);

        depth_map.upload(cl->m_command_queue,
                         host_depth.data(),
                         /*stride=*/0, cols, rows);

        // Build GT hint pose if available.
        // Pose injected into KinFu's world frame:
        //   T_hint = T_kinfu_0 * T_gt_0^{-1} * T_gt_i
        bool ok;
        if (use_gt && gt_origin_set) {
            GTEntry gt;
            if (find_nearest_gt(gt_entries, de.timestamp, gt)) {
                Eigen::Affine3f T_gt_i    = gt_to_affine(gt);
                Eigen::Affine3f hint_pose = T_kinfu_0 * T_gt_0_inv * T_gt_i;
                ok = tracker(depth_map, &hint_pose);
            } else {
                // No close GT entry — fall back to pure ICP for this frame
                ok = tracker(depth_map);
            }
        } else {
            ok = tracker(depth_map);
        }

        if (ok) ++tracked; else ++lost;
        ++loaded;

        printf("Frame %4d / %d  [%s]  tracking=%s\n",
               loaded, (total + skip - 1) / skip,
               de.filename.c_str(),
               ok ? "OK" : "LOST");
    }

    printf("\nTracking summary: %d OK / %d lost  (%d frames processed)\n",
           tracked, lost, loaded);

    // --- Extract mesh ---
    printf("Extracting mesh (marching cubes)...\n");

    pcl::gpu::MarchingCubes mc;
    pcl::gpu::CLDeviceArray<pcl::PointXYZ> tri_buf;
    pcl::gpu::CLDeviceArray<cl_float> tri_verts = mc.run(tracker.volume(), tri_buf);

    size_t n_floats = tri_verts.size();
    printf("  triangles : %zu\n", n_floats / 18);

    if (n_floats > 0) {
        std::vector<float> host(n_floats);
        tri_verts.download(cl->m_command_queue, host.data());
        clFinish(cl->m_command_queue);

        Eigen::Vector3f vsz = tracker.volume().getVoxelSize();

        const char* out_path = "output_tum.ply";
        FILE* fp = fopen(out_path, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); return 1; }

        // MarchingCubes output layout (from marchingcube.cl vstore3 calls):
        //   per triangle: 18 floats = 3 vertices × [vx,vy,vz, nx,ny,nz]
        //   vertex positions are in voxel coordinates → multiply by voxel size
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
