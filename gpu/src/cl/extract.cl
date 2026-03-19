#define SHORT_NAN 32767
#define SHORT_MAX 32766
#define MAX_WEIGHT 128
enum
{
	CTA_SIZE_X = 32,
	CTA_SIZE_Y = 8,
	CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

	MAX_LOCAL_POINTS = 3,
	////warp enum
	LOG_WARP_SIZE = 5,
	WARP_SIZE = 1 << LOG_WARP_SIZE,
	STRIDE = WARP_SIZE,
	DEFAULT_CLOUD_BUFFER_SIZE = 100 * 1000 * 1000
};

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

typedef float4 PointType;
typedef struct { float x, y, z, w, c1, c2, c3, c4; } NormalType;
//typedef float8 NormalType;
//typedef struct {
//	float resolution[3];
//	float volume[3];
//	float tranc_dist;//0.03f
//	int width;
//	int height;
//	Intr intr;
//	Mat33 Rcurr_inv;
//	Float3 tcurr;
//} TsdfParams;
typedef struct {
	float VOLUME[3];
	float cell_size[3];
} ExtractParams;
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
int3 getVoxel(float3 point, float3 cell_size)
{
	int3 v = convert_int3_rtn(native_divide(point, cell_size));
	return v;
}

float3 fetchPoint(int idx, __global float* point)
{
	PointType p = *(__global PointType*)&point[idx * 4];
	return (float3)(p.x, p.y, p.z);
}

int flattenedThreadId()
{
	return (get_local_id(2) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0));
}

unsigned int laneId()
{
	unsigned int ret;
	//asm("mov.u32 %0, %%laneid;" : "=r"(ret) );
	ret = get_local_id(0);
	return ret;
}


int scan_warp(__local int *ptr, const unsigned int idx,int total_warp)
{
	const unsigned int lane = idx & 31;       // index of thread in warp (0..31) 
	barrier(CLK_LOCAL_MEM_FENCE);
	if(total_warp>0)if (lane >= 1) ptr[idx] = ptr[idx - 1] + ptr[idx]; barrier(CLK_LOCAL_MEM_FENCE);
	if (total_warp>0)if (lane >= 2) ptr[idx] = ptr[idx - 2] + ptr[idx]; barrier(CLK_LOCAL_MEM_FENCE);
	if (total_warp>0)if (lane >= 4) ptr[idx] = ptr[idx - 4] + ptr[idx]; barrier(CLK_LOCAL_MEM_FENCE);
	if (total_warp>0)if (lane >= 8) ptr[idx] = ptr[idx - 8] + ptr[idx]; barrier(CLK_LOCAL_MEM_FENCE);
	if (total_warp>0)if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx]; barrier(CLK_LOCAL_MEM_FENCE);
	// if (Kind == inclusive)
	////  return ptr[idx];
	// else
	//barrier(CLK_LOCAL_MEM_FENCE);
	return (lane > 0) ? ptr[idx - 1] : 0;
}
//
//int warp_reduce2(__local int *ptr, const unsigned int tid)
//{
//	const unsigned int lane = tid & 31;       
//	
//	if (lane < 16)
//	{
//		int partial = ptr[tid];
//		
//		ptr[tid] = partial = partial + ptr[tid + 16];
//		ptr[tid] = partial = partial + ptr[tid + 8];
//		ptr[tid] = partial = partial + ptr[tid + 4];
//		ptr[tid] = partial = partial + ptr[tid + 2];
//		ptr[tid] = partial = partial + ptr[tid + 1];
//	}
//	//printf("2 %d %d %d %d\n", ptr[tid - lane], tid, lane, ptr[tid]);
//	return ptr[tid - lane];
//}

int warp_reduce(__local int *ptr, const unsigned int tid)
{
	const unsigned int lane = tid & 31; // index of thread in warp (0..31)        
	barrier(CLK_LOCAL_MEM_FENCE); 
	if (lane < 16)	{
		int partial = ptr[tid];ptr[tid] = partial = partial + ptr[tid + 16];	}barrier(CLK_LOCAL_MEM_FENCE);	//barrier(CLK_LOCAL_MEM_FENCE);//printf("warp_reduce2 1 %d %d %d %d\n", ptr[tid - lane], tid, ptr[tid], lane);
	if (lane < 16)	{
		int partial = ptr[tid];	ptr[tid] = partial = partial + ptr[tid + 8];	}barrier(CLK_LOCAL_MEM_FENCE);
	//printf("warp_reduce2 2 %d %d %d %d\n", ptr[tid - lane], tid, ptr[tid], lane);
	if (lane < 16)	{
		int partial = ptr[tid]; ptr[tid] = partial = partial + ptr[tid + 4];	}barrier(CLK_LOCAL_MEM_FENCE);
		//printf("warp_reduce2 3 %d %d %d %d\n", ptr[tid - lane], tid, ptr[tid], lane);
	if (lane < 16)	{
		int partial = ptr[tid]; ptr[tid] = partial = partial + ptr[tid + 2];	}barrier(CLK_LOCAL_MEM_FENCE);
		//printf("warp_reduce2 4 %d %d %d %d\n", ptr[tid - lane], tid, ptr[tid], lane);
	if (lane < 16)	{
		int partial = ptr[tid]; ptr[tid] = partial = partial + ptr[tid + 1];	}barrier(CLK_LOCAL_MEM_FENCE);
		//printf("warp_reduce2 5 %d %d %d %d\n", ptr[tid - lane], tid, ptr[tid], lane);

	
	////barrier(CLK_LOCAL_MEM_FENCE);
	
	return ptr[tid - lane];
}

bool All(int predicate, __local int* cta_buffer)
{
	int tid = flattenedThreadId();
	cta_buffer[tid] = predicate ? 1 : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	return (warp_reduce(cta_buffer, tid) == 32);

}
float2 fetch(__global short2* volume_data, int idx) {
	float2 tsdf = unpack_tsdf(volume_data, idx);
	return tsdf;
}
float readTsdf(__global short2* volume_data, int x, int y, int z, float3 VOLUME_) {
	int idx = (VOLUME_.y * z + y) * VOLUME_.x + x;
	float2 tsdf = unpack_tsdf(volume_data, idx);
	 return tsdf.x;
}
void  store_point_type(float x, float y, float z, __global PointType* ptr) {
	//printf("%f %f %f\n", x, y, z);
	*ptr = (PointType)(x, y, z, 0.f);
}
float interpolateTrilineary(__global short2* volume_data, const float3 point, ExtractParams l_params) 
{

	float3 cell_size = *(float3*)&l_params.cell_size[0];
	float3 VOLUME_ = *(float3*)&l_params.VOLUME[0];

	int3 g = getVoxel(point, cell_size);

	float vx = (g.x + 0.5f) * cell_size.x;
	float vy = (g.y + 0.5f) * cell_size.y;
	float vz = (g.z + 0.5f) * cell_size.z;

	g.x = (point.x < vx) ? (g.x - 1) : g.x;
	g.y = (point.y < vy) ? (g.y - 1) : g.y;
	g.z = (point.z < vz) ? (g.z - 1) : g.z;

	float a = (point.x - (g.x + 0.5f) * cell_size.x) / cell_size.x;
	float b = (point.y - (g.y + 0.5f) * cell_size.y) / cell_size.y;
	float c = (point.z - (g.z + 0.5f) * cell_size.z) / cell_size.z;

	float res = readTsdf(volume_data, g.x + 0, g.y + 0, g.z + 0, VOLUME_) * (1 - a) * (1 - b) * (1 - c) +
		readTsdf(volume_data, g.x + 0, g.y + 0, g.z + 1, VOLUME_) * (1 - a) * (1 - b) * c +
		readTsdf(volume_data, g.x + 0, g.y + 1, g.z + 0, VOLUME_) * (1 - a) * b * (1 - c) +
		readTsdf(volume_data, g.x + 0, g.y + 1, g.z + 1, VOLUME_) * (1 - a) * b * c +
		readTsdf(volume_data, g.x + 1, g.y + 0, g.z + 0, VOLUME_) * a * (1 - b) * (1 - c) +
		readTsdf(volume_data, g.x + 1, g.y + 0, g.z + 1, VOLUME_) * a * (1 - b) * c +
		readTsdf(volume_data, g.x + 1, g.y + 1, g.z + 0, VOLUME_) * a * b * (1 - c) +
		readTsdf(volume_data, g.x + 1, g.y + 1, g.z + 1, VOLUME_) * a * b * c;
	return res;
}


__kernel void extractCloud(__global short2* volume,
	__global float* output,//float4
	__global int* global_count,
	__global int* output_count,
	__global unsigned int* blocks_done,
	__local int* cta_buffer,
	__local float* storage_X,
	__local float* storage_Y,
	__local float* storage_Z,
	__constant ExtractParams* params)
{


	//ExtractParams l_params = *params;
	//int x = get_global_id(0);//get_local_id(0) + get_group_id(0) * CTA_SIZE_X;
	//int y = get_global_id(1);// get_local_id(1) + get_group_id(1) * CTA_SIZE_Y;


	//float3 cell_size = { l_params.cell_size[0], l_params.cell_size[1], l_params.cell_size[2] };
	//float3 VOLUME_ = { l_params.VOLUME[0], l_params.VOLUME[1], l_params.VOLUME[2] };

	//if (All(x >= VOLUME_.x, cta_buffer) || All(y >= VOLUME_.y, cta_buffer)){
	//	return;
	//}
	////
	//float3 V;
	//V.x = (x + 0.5f) * cell_size.x;
	//V.y = (y + 0.5f) * cell_size.y;
	//////////////////////////
	//int ftid = flattenedThreadId();
	///*if (ftid == 0){
	//	for (int i = 0; i < CTA_SIZE * MAX_LOCAL_POINTS; i++)
	//	{
	//		storage_W[i] = 0.f;
	//	}
	//}*/
	//

	//barrier(CLK_LOCAL_MEM_FENCE);
	//for (int z = 0; z < VOLUME_.z - 1; ++z)
	//{
	//	float3 points[MAX_LOCAL_POINTS];
	//	int local_count = 0;

	//	if (x < VOLUME_.x && y < VOLUME_.y)
	//	{
	//		int idx = (VOLUME_.y * z + y) * VOLUME_.x + x;
	//		float2 tsdf1 = fetch(volume, idx);
	//		float F = tsdf1.x;
	//		int W = convert_int(tsdf1.y);
	//		if (W != 0 && F != 1.f)
	//		{
	//			V.z = (z + 0.5f) * cell_size.z;
	//			//process dx
	//			if (x + 1 < VOLUME_.x)
	//			{
	//				int idx2 = (VOLUME_.y * z + y) * VOLUME_.x + (x + 1);
	//				float2 tsdf2 = fetch(volume, idx2);
	//				float Fn = tsdf2.x;
	//				int Wn = convert_int(tsdf2.y);
	//				if (Wn != 0 && Fn != 1.f)
	//				if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
	//				{
	//					float3 p;
	//					p.y = V.y;
	//					p.z = V.z;

	//					float Vnx = V.x + cell_size.x;

	//					float d_inv = native_divide(1.f , (fabs(F) + fabs(Fn)));
	//					p.x = (V.x * fabs(Fn) + Vnx * fabs(F)) * d_inv;
	//				
	//					points[local_count++] = p;
	//				}
	//			}               /* if (x + 1 < VOLUME_X) */

	//			//process dy
	//			if (y + 1 < VOLUME_.y)
	//			{
	//				int idx3 = (VOLUME_.y * z + (y + 1)) * VOLUME_.x + x;
	//				float2 tsdf3 = fetch(volume, idx3);
	//				float Fn = tsdf3.x;
	//				int Wn = convert_int(tsdf3.y);

	//				if (Wn != 0 && Fn != 1.f)
	//				if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
	//				{
	//					float3 p;
	//					p.x = V.x;
	//					p.z = V.z;

	//					float Vny = V.y + cell_size.y;

	//					float d_inv = native_divide(1.f , (fabs(F) + fabs(Fn)));
	//					p.y = (V.y * fabs(Fn) + Vny * fabs(F)) * d_inv;
	//					
	//					points[local_count++] = p;
	//				}
	//			}                /*  if (y + 1 < VOLUME_Y) */

	//			//process dz
	//			//if (z + 1 < VOLUME_.z) // guaranteed by loop
	//			{
	//				int idx4 = (VOLUME_.y * (z + 1) + y) * VOLUME_.x + x;
	//				float2 tsdf4 = fetch(volume, idx4);
	//				float Fn = tsdf4.x;
	//				int Wn = convert_int(tsdf4.y);

	//				if (Wn != 0 && Fn != 1.f)
	//				if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
	//				{
	//					float3 p;
	//					p.x = V.x;
	//					p.y = V.y;

	//					float Vnz = V.z + cell_size.z;

	//					float d_inv = native_divide(1.f , (fabs(F) + fabs(Fn)));
	//					p.z = (V.z * fabs(Fn) + Vnz * fabs(F)) * d_inv;

	//					points[local_count++] = p;
	//				}
	//			}               /* if (z + 1 < VOLUME_Z) */
	//		}              /* if (W != 0 && F != 1.f) */
	//	}            /* if (x < VOLUME_X && y < VOLUME_Y) */



	//	int tid = flattenedThreadId();

	//	cta_buffer[tid] = local_count;
	//	int total_warp = warp_reduce(cta_buffer, tid);
	//	
	//	int lane;
	//	int old_global_count;
	//	int storage_index;
	//	int offset = 0;
	//	if (total_warp > 0)
	//	{
	//		lane = laneId();
	//		storage_index = (ftid >> LOG_WARP_SIZE/*5*/) * WARP_SIZE /*32*/ * MAX_LOCAL_POINTS/*3*/;

	//	
	//		volatile __local int* cta_buffer1 = (__local int*)(storage_X + storage_index);

	//		cta_buffer1[lane] = local_count;
	//		offset = scan_warp(cta_buffer1, lane,total_warp);
	//	
	//		if (lane == 0)
	//		{
	//			int old_global_count = atomic_add(global_count, total_warp);
	//			cta_buffer1[0] = old_global_count;
	//		}

	//		 old_global_count = cta_buffer1[0];
	//		for (int l = 0; l < local_count; ++l)
	//		{
	//			storage_X[storage_index + offset + l] = points[l].x;
	//			storage_Y[storage_index + offset + l] = points[l].y;
	//			storage_Z[storage_index + offset + l] = points[l].z;
	//		}
	//	
	//		__global PointType *pos = (__global PointType*)(output + (4 * (old_global_count + lane)));
	//		for (int i = lane; i < total_warp; i += STRIDE, pos += (STRIDE * 4))
	//		{
	//			float x = storage_X[storage_index + i];
	//			float y = storage_Y[storage_index + i];
	//			float z = storage_Z[storage_index + i];
	//			store_point_type(x, y, z, pos);
	//		}

	//		bool full = (old_global_count + total_warp) >= DEFAULT_CLOUD_BUFFER_SIZE;

	//		if (full)
	//			break;

	//	}

	//}         /* for(int z = 0; z < VOLUME_Z - 1; ++z) */

	//if (ftid == 0)
	//{
	//	unsigned int total_blocks = get_num_groups(0) *  get_num_groups(1) *  get_num_groups(2);
	//	//*blocks_done = (blocks_done >= total_blocks) ? 0 : value;
	//	unsigned int value = atomic_inc(blocks_done);
	//	value = (value >= total_blocks) ? 0 : value;

	//	//last block
	//	if (value >= total_blocks - 1)
	//	{
	//		*output_count = min(DEFAULT_CLOUD_BUFFER_SIZE, *global_count);
	//		*blocks_done = 0;
	//		*global_count = 0;
	//	}
	//}
}

__kernel void extractNormals(__global short2* volume,
	__global float* input,
	__global float* output,//float4
	__constant ExtractParams* params)
{
	//const int idx = get_global_id(0);
	//if (idx >= get_global_size(0))
	//	return;

	//ExtractParams l_params = *params;

	//float3 cell_size = *(float3*)&l_params.cell_size[0];
	//float3 VOLUME_ = *(float3*)&l_params.VOLUME[0];

	//float3 n = { NAN, NAN, NAN };
	//float3 point = fetchPoint(idx, input);
	//int3 g = getVoxel(point, cell_size);
	//if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < VOLUME_.x - 2 && g.y < VOLUME_.y - 2 && g.z < VOLUME_.z - 2)
	//{
	//	float3 t;

	//	t = point;
	//	t.x += cell_size.x;
	//	float Fx1 = interpolateTrilineary(volume, t, l_params);

	//	t = point;
	//	t.x -= cell_size.x;
	//	float Fx2 = interpolateTrilineary(volume, t, l_params);

	//	n.x = (Fx1 - Fx2);

	//	t = point;
	//	t.y += cell_size.y;
	//	float Fy1 = interpolateTrilineary(volume, t, l_params);

	//	t = point;
	//	t.y -= cell_size.y;
	//	float Fy2 = interpolateTrilineary(volume, t, l_params);

	//	n.y = (Fy1 - Fy2);

	//	t = point;
	//	t.z += cell_size.z;
	//	float Fz1 = interpolateTrilineary(volume, t, l_params);

	//	t = point;
	//	t.z -= cell_size.z;
	//	float Fz2 = interpolateTrilineary(volume, t, l_params);

	//	n.z = (Fz1 - Fz2);

	//	n = normalize(n);
	//}
	////storeNormal(idx, n);
	//(*(__global PointType*)&output[idx * 4]).s012 = n;
}

void storeNormal(int idx, float3 normal,__global float* output) 
{
	/*NormalType n;
	n.x = normal.x; n.y = normal.y; n.z = normal.z;
	*(__global NormalType*)&output[idx * sizeof(NormalType)] = n;*/
}

__kernel void extractNormals2(__global short2* volume,
	__global float* input,
	__global float* output,//float8
	__constant ExtractParams* params)
{
	const int idx = get_global_id(0);
	if (idx >= get_global_size(0))
		return;

	ExtractParams l_params = *params;

	float3 cell_size = *(float3*)&l_params.cell_size[0];
	float3 VOLUME_ = *(float3*)&l_params.VOLUME[0];

	float3 n = { NAN, NAN, NAN };
	float3 point = fetchPoint(idx,input);
	int3 g = getVoxel(point, cell_size);
	//if(idx ==0) 	printf("%f %f %f\n", cell_sz.x, cell_sz.y, cell_sz.z);
	if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < VOLUME_.x - 2 && g.y < VOLUME_.y - 2 && g.z < VOLUME_.z - 2)
	{
		float3 t;

		t = point;
		t.x += cell_size.x;
		float Fx1 = interpolateTrilineary(volume, t, l_params);

		t = point;
		t.x -= cell_size.x;
		float Fx2 = interpolateTrilineary(volume, t, l_params);

		n.x = (Fx1 - Fx2);

		t = point;
		t.y += cell_size.y;
		float Fy1 = interpolateTrilineary(volume, t, l_params);

		t = point;
		t.y -= cell_size.y;
		float Fy2 = interpolateTrilineary(volume, t, l_params);

		n.y = (Fy1 - Fy2);

		t = point;
		t.z += cell_size.z;
		float Fz1 = interpolateTrilineary(volume, t, l_params);

		t = point;
		t.z -= cell_size.z;
		float Fz2 = interpolateTrilineary(volume, t, l_params);

		n.z = (Fz1 - Fz2);

		n = normalize(n);
	}
	storeNormal(idx, n, output);
	//(*(NormalType*)&output[idx * 8]).s012 = n;
}