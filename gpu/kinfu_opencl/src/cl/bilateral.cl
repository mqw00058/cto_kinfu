
#define LOCAL_SIZE 16
//
//__constant const float sigma_color2_inv_half = 5.5555555 * 10 - 4;
//__constant const float sigma_space2_inv_half = 2.46913582 * 10 - 2;
__constant const float sigma_color2_inv_half = 5.5555555e-2;
__constant const float sigma_space2_inv_half = 2.46913582e-1;

__kernel __attribute__((work_group_size_hint(LOCAL_SIZE, LOCAL_SIZE, 1))) void bilateralF(__global const ushort* src,
	__global ushort* dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int cols = get_global_size(0);
	int rows = get_global_size(1);
	if (x >= cols || y >= rows) 
		return;

	const int R = 6;       //static_cast<int>(sigma_space * 1.5);
	const int D = R * 2 + 1;

	const int x_local = get_local_id(0);
	const int y_local = get_local_id(1);
	const int cols_local = get_local_size(0);
	const int rows_local = get_local_size(1);
	const int cols2_local = cols_local * 2;
	const int rows2_local = cols_local * 2;
	const int cols_half_local = cols_local / 2;
	const int rows_half_local = cols_local / 2;
	const int x_group = get_group_id(0);
	const int y_group = get_group_id(1);

	__local ushort src_local[LOCAL_SIZE * 2 * LOCAL_SIZE * 2];
	int x_global = x - (cols_half_local)+x_local;
	int y_global = y - (rows_half_local)+y_local;
	if (x_global >= 0 && y_global >= 0 && x_global < cols && y_global < rows)
	{
		int idx_global = cols * y_global + x_global;
		int idx_local = (cols2_local)* (y_local * 2) + (x_local * 2);
#ifdef ANDROID
		__local ushort2* psrc_local = &src_local[idx_local];
		__global ushort2* psrc = &src[idx_global];
		*psrc_local = *psrc;
		psrc_local = &src_local[idx_local + cols2_local];
		psrc = &src[idx_global + cols];
		*psrc_local = *psrc;
#else
		src_local[idx_local] = src[idx_global];
		src_local[idx_local + 1] = src[idx_global + 1];
		src_local[idx_local + cols2_local] = src[idx_global + cols];
		src_local[idx_local + cols2_local + 1] = src[idx_global + cols + 1];
#endif
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int idx = mad24(cols, y, x);
	int x_local_src = x_local + cols_half_local;
	int y_local_src = y_local + rows_half_local;
	int idx_local_src = mad24(cols2_local, y_local_src, x_local_src);
	int src_depth = (int)src_local[idx_local_src];

	int tx = min(x - D / 2 + D, cols - 1);
	int ty = min(y - D / 2 + D, rows - 1);

	float sum1 = 0.0f;
	float sum2 = 0.0f;

	float s2 = sigma_space2_inv_half;
	float c2 = sigma_color2_inv_half;

	for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
	{
		int cy_local = cy - (y_group*rows_local) + rows_half_local;
		for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
		{
			int cx_local = cx - (x_group*cols_local) + cols_half_local;
			int idx2 = mad24(cols2_local, cy_local, cx_local);
			int tmp = (int)src_local[idx2];

			float space2 = (float)(x - cx) * (x - cx) + (float)(y - cy) * (y - cy);
			float color2 = (float)(src_depth - tmp) * (src_depth - tmp);

			float weight = native_exp(-(space2 * s2 + color2 * c2));

			sum1 += (float)tmp * weight;
			sum2 += weight;
		}
	}

	int res = convert_int_rte(sum1 / sum2);

	dst[idx] = max(0, min(res, SHRT_MAX));
}
__kernel __attribute__((work_group_size_hint(32, 8, 1))) void pyrDown(__global const ushort* src,
	__global ushort* dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int dst_cols = get_global_size(0);
	const int dst_rows = get_global_size(1);
	const int src_cols = get_global_size(0) * 2;
	const int src_rows = get_global_size(1) * 2;
	const float sigma_color = 30.0f;
	if (x >= dst_cols || y >= dst_rows)
		return;

	const int D = 5;

	int idx = mad24(src_cols, (2 * y), (2 * x));
	int center = src[idx];

	int tx = min(2 * x - D / 2 + D, src_cols - 1);
	int ty = min(2 * y - D / 2 + D, src_rows - 1);
	int cy = max(0, 2 * y - D / 2);

	int sum = 0;
	int count = 0;

	for (; cy < ty; ++cy)
	for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx)
	{
		int idx2 = mad24(src_cols, cy, cx);
		int val = src[idx2];

		if (abs(val - center) < 3 * sigma_color)
		{
			sum += val;
			++count;
		}
	}

	int idx3 = mad24(dst_cols, y, x);
	dst[idx3] = sum / count;
}

__kernel __attribute__((work_group_size_hint(32, 8, 1))) void truncateDepth(__global ushort* depth,
	__global float* g_max_distance
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int cols = get_global_size(0);
	const int rows = get_global_size(1);

	int idx = mad24(cols, y, x);

	if (x < cols && y < rows)
	if (depth[idx] > *g_max_distance) {
		depth[idx] = 0;
	}
}