typedef struct {
	float fx;
	float fy;
	float cx;
	float cy;
} Intr;

typedef struct {
	float x;
	float y;
	float z;
} Float3;
typedef struct {
	unsigned char x;
	unsigned char y;
	unsigned char z;
} Uchar3;
typedef struct {
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
} Uchar4;
typedef struct
{
	Float3 data[3];
}  Mat33;
typedef struct {
	float resolution[3];
	float volume[3];
	float tranc_dist;
	int weight;
	int width;
	int height;
	Intr intr;
} colorParams;
typedef struct {
	float VOLUME[3];
	float cell_size[3];
} extractColorParams;

enum
{
	ONE_VOXEL = 0
};

int3 getVoxel(float3 point, float3 cell_size) {
	int3 v;
	v.x = convert_int_rtn(point.x / cell_size.x);                // round to negative infinity
	v.y = convert_int_rtn(point.y / cell_size.y);
	v.z = convert_int_rtn(point.z / cell_size.z);
	return v;
}
float3 getVoxelGCoo(int x, int y, int z, float3 cell_size) {
	float3 coo = { convert_float(x), convert_float(y), convert_float(z) };
	coo += 0.5f;                 //shift to cell center;
	coo.x *= cell_size.x;
	coo.y *= cell_size.y;
	coo.z *= cell_size.z;
	return coo;
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void updateColorVolumeKernel(__global uchar4* color_volume,
	__global Uchar3* colors,
	__read_only image2d_t vmap_src,
	__constant Mat33* cam_pos_inv,
	__global float* cam_pos_t,
	__constant colorParams* params
	)
{
	//const int x = get_global_id(0);
	//const int y = get_global_id(1);
	//colorParams l_params = *params;
	//Mat33 R_curr_inv = *cam_pos_inv;

	//if (x >= convert_int(l_params.resolution[0]) || y >= convert_int(l_params.resolution[1]))
	//{
	//	return;
	//}

	//int w = l_params.width; //640
	//int h = l_params.height;  //480

	//float3 Tcurr = *(__global float3*)&cam_pos_t[0];
	//Intr intr = l_params.intr;

	//float tranc_dist = l_params.tranc_dist;
	//int max_weight = l_params.weight;
	//float3 cell_size = { l_params.volume[0] / l_params.resolution[0], l_params.volume[1] / l_params.resolution[1], l_params.volume[2] / l_params.resolution[2] };


	//for (int z = 0; z < l_params.resolution[2]/*VOLUME_Z*/; ++z)
	//{
	//	float3 v_g = getVoxelGCoo(x, y, z, cell_size);
	//	float3 v_g_temp = v_g - Tcurr;
	//	float3 v;
	//	v.x = (R_curr_inv.data[0].x * v_g_temp.x + R_curr_inv.data[0].y * v_g_temp.y + R_curr_inv.data[0].z * v_g_temp.z);
	//	v.y = (R_curr_inv.data[1].x * v_g_temp.x + R_curr_inv.data[1].y * v_g_temp.y + R_curr_inv.data[1].z * v_g_temp.z);
	//	v.z = (R_curr_inv.data[2].x * v_g_temp.x + R_curr_inv.data[2].y * v_g_temp.y + R_curr_inv.data[2].z * v_g_temp.z);
	//	
	//	if (v.z <= 0)
	//		continue;

	//	int2 coo;                   //project to current cam
	//	coo.x = convert_int_rte(v.x * intr.fx / v.z + intr.cx);
	//	coo.y = convert_int_rte(v.y * intr.fy / v.z + intr.cy);
	//	//if (x == 100 && y == 100)printf("%d %d %f \n", coo.x, coo.y, 0.f);
	//	if (coo.x >= 0 && coo.y >= 0 && coo.x < w && coo.y < h)
	//	{
	//		float3 p;
	//		int2 coord = (int2)(coo.x, coo.y);
	//		p = read_imagef(vmap_src, sampler, coord).xyz;
	//		
	//		if (isnan(p.x))
	//			continue;


	//		bool update = false;
	//		if (ONE_VOXEL)
	//		{
	//			int3 vp = getVoxel(p, cell_size);
	//			update = vp.x == x && vp.y == y && vp.z == z;
	//		}
	//		else
	//		{
	//			float dist = length(p - v_g);
	//			update = dist < tranc_dist;
	//		}
	//		if (update)
	//		{
	//			int pos = (l_params.resolution[1] * z + y)* l_params.resolution[0] + x;
	//			__global uchar4 *ptr = (color_volume + pos);
	//			Uchar3 rgb = colors[coo.y* w + coo.x];
	//			//if (rgb.x != 0.0f)printf("%d %d %d \n", rgb.x, rgb.y, rgb.z);
	//			uchar4 volume_rgbw = *ptr;
	//			int weight_prev = convert_int(volume_rgbw.w);
	//			const float Wrk = 1.f;
	//			float new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
	//			float new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
	//			float new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);
	//			int weight_new = weight_prev + 1;
	//			uchar4 volume_rgbw_new;
	//			volume_rgbw_new.x = min(255, max(0, convert_int_rte(new_x)));
	//			volume_rgbw_new.y = min(255, max(0, convert_int_rte(new_y)));
	//			volume_rgbw_new.z = min(255, max(0, convert_int_rte(new_z)));
	//			volume_rgbw_new.w = min(max_weight, weight_new);
	//			*ptr = volume_rgbw_new;
	//		}
	//	}
	//}
}
__kernel void initColorVolumeKernel(__global Uchar4* color_volume, __global int* volume_size)
{
	//const int x = get_global_id(0);
	//const int y = get_global_id(1);

	//if (x < volume_size[0] && y < volume_size[1])
	//{
	//	int pos = y * volume_size[0] + x;
	//	int elem_step = volume_size[1] * volume_size[0];

	//	for (int z = 0; z < volume_size[2]; z++, pos += elem_step)
	//	{
	//		Uchar4 res = { 0, 0, 0, 0 };
	//		color_volume[pos] = res;
	//	}
	//}
}

__kernel void extractColorsKernel(__global Uchar4* color_volume,
	__global float* points,
	__global unsigned char* colors,
	__constant extractColorParams* params)
{
	//const int idx = get_global_id(0);
	//if (idx >= get_global_size(0))
	//	return;

	//extractColorParams l_params = *params;

	//float3 cell_size = *(float3*)&l_params.cell_size[0];
	//float3 VOLUME_ = *(float3*)&l_params.VOLUME[0];

	//float4 p_temp = *(__global float4*)&points[idx * 4];
	//float3 p = p_temp.xyz;
	//int3 v = convert_int3_rtn(p / cell_size);
	//int idx2 = (VOLUME_.y * v.z + v.y) * VOLUME_.x + v.x;
	//Uchar4 rgbw = color_volume[idx2];

	//Uchar4 result = { rgbw.z, rgbw.y, rgbw.x, 0 };
	//*(__global Uchar4*)&colors[idx * 4] = result; //bgra


}
