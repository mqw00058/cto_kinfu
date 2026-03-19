#define SHORT_NAN 32767
#define SHORT_MAX 32766
#define MAX_WEIGHT 128
typedef struct {
	float fx;
	float fy;
	float cx;
	float cy;
} Intr;
typedef struct
{
	float fx_inv;
	float fy_inv;
	float cx;
	float cy;
} Intr_inv;
typedef struct {
	float x;
	float y;
	float z;
} Float3;

typedef struct {
	Float3 data[3];
} Mat33;
typedef struct {
	float resolution[3];
	float volume[3];
	float tranc_dist;//0.03f
	int width;
	int height;
	Intr intr;
	Mat33 Rcurr_inv;
	Float3 tcurr;
} TsdfParams;
void pack_tsdf(float val, short weight, __global short2* tsdf, int idx)
{
	int fixedp = max(-SHORT_NAN, min(SHORT_NAN, convert_int_rtz(val *  convert_float(SHORT_NAN))));
	short2 res = { convert_short(fixedp), (weight) };
	tsdf[idx] = res;
}
float2 unpack_tsdf(__global short2* tsdf, int idx)
{
	float2 res = { native_divide(convert_float(tsdf[idx].x), convert_float(SHORT_NAN)), convert_float(tsdf[idx].y) };
	return res;
}
float3 getGlobalVoxel(int x, int y, float3 cell_size, float3 Tcurr)
{
	float3 pos = (float3)(x, y, 0.f);
	float3 g_v = (pos + 0.5f) * cell_size - Tcurr;
	//float3 g_v = { (x + 0.5f) * cell_size.x - Tcurr.x, (y + 0.5f) * cell_size.y - Tcurr.y, (0 + 0.5f) * cell_size.z - Tcurr.z };
	return g_v;
}
__kernel void TSDF(__global short2* Volume_data,
	__global float* scaled,
	__constant TsdfParams* params
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	TsdfParams l_params = *params;
	Mat33 R_curr_inv = l_params.Rcurr_inv;

	if (x >= convert_int(l_params.resolution[0]) || y >= convert_int(l_params.resolution[1]))
	{
		return;
	}

	int w = l_params.width;
	int h = l_params.height;

	float3 Tcurr = { l_params.tcurr.x, l_params.tcurr.y, l_params.tcurr.z };

	Intr intr = l_params.intr;

	float tranc_dist = l_params.tranc_dist;
	float3 cell_size = { native_divide(l_params.volume[0], l_params.resolution[0]), native_divide(l_params.volume[1], l_params.resolution[1]), native_divide(l_params.volume[2], l_params.resolution[2]) };
	float3 v_g = getGlobalVoxel(x, y, cell_size, Tcurr);
	float v_g_part_norm = v_g.x * v_g.x + v_g.y * v_g.y;
	float3 v;
	v.x = (R_curr_inv.data[0].x * v_g.x + R_curr_inv.data[0].y * v_g.y + R_curr_inv.data[0].z * v_g.z) * intr.fx;
	v.y = (R_curr_inv.data[1].x * v_g.x + R_curr_inv.data[1].y * v_g.y + R_curr_inv.data[1].z * v_g.z) * intr.fy;
	v.z = (R_curr_inv.data[2].x * v_g.x + R_curr_inv.data[2].y * v_g.y + R_curr_inv.data[2].z * v_g.z);

	float z_scaled = 0;
	float Rcurr_inv_0_z_scaled = R_curr_inv.data[0].z * cell_size.z * intr.fx;
	float Rcurr_inv_1_z_scaled = R_curr_inv.data[1].z * cell_size.z * intr.fy;
	float tranc_dist_inv = native_divide(1.0f, tranc_dist);

	int pos = mad24(y, l_params.resolution[0], x);
	int elem_step = l_params.resolution[1] * l_params.resolution[0];

	for (int z = 0; z < convert_int(l_params.resolution[2]);
		++z,
		v_g.z += cell_size.z,
		z_scaled += cell_size.z,
		v.x += Rcurr_inv_0_z_scaled,
		v.y += Rcurr_inv_1_z_scaled,
		pos += elem_step)
	{
		float inv_z = native_divide(1.0f, (v.z + R_curr_inv.data[2].z * z_scaled));
		if (inv_z < 0)
		{
			continue;
		}
		int2 coo;

		coo.x = convert_int_rte(v.x * inv_z + intr.cx);
		coo.y = convert_int_rte(v.y * inv_z + intr.cy);


		if (coo.x >= 0 && coo.y >= 0 && coo.x < w && coo.y < h)
		{
			float Dp_scaled = scaled[coo.y * w + coo.x];
			float sdf = Dp_scaled - sqrt(v_g.z * v_g.z + v_g_part_norm);

			if (Dp_scaled != 0.f && sdf >= -tranc_dist) //meters
			{
				float tsdf = min(1.0f, sdf * tranc_dist_inv);
				float2 tsdf_prev = unpack_tsdf(Volume_data, pos);

				const int Wrk = 1;
				float tsdf_new = native_divide((tsdf_prev.x * tsdf_prev.y + Wrk * tsdf), (tsdf_prev.y + Wrk));
				tsdf_new =  max(-1.0, min(tsdf_new, 1.0)); //inyeop
				int weight_new = min(convert_int(tsdf_prev.y) + Wrk, MAX_WEIGHT);
				
				pack_tsdf(tsdf_new, weight_new, Volume_data, pos);
			}	
		}
	}
}
__kernel void scaleDepth(__global ushort* dmap,
						__global float* scaled,
						__global Intr_inv *g_intr_inv
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= w || y >= h)
		return;
	int idx = mad24(w , y , x);
	Intr_inv intr_inv = *g_intr_inv;
	float xl = (convert_float(x) - intr_inv.cx) * intr_inv.fx_inv;
	float yl = (convert_float(y) - intr_inv.cy) * intr_inv.fy_inv;
	float lambda = sqrt(xl * xl + yl * yl + 1.0f);
	scaled[idx] = convert_float(dmap[idx]) * native_divide(lambda , 1000.f);

}
__kernel void initializeVolume(__global short2* volume, __global int* volume_size)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if (x < volume_size[0] && y < volume_size[1])
	{
		int pos = y * volume_size[0] + x;
		int elem_step = volume_size[1] * volume_size[0];

		for (int z = 0; z < volume_size[2]; z++, pos += elem_step)
		{
			pack_tsdf(1.f, 0, volume, pos);
		}
	}
}
