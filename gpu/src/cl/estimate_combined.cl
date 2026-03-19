//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#define IMAGE_OPT
typedef float float_type;
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
	Float3 data[3];
} Mat33;
typedef struct
{
	float distThres;
	float angleThres;
	Intr intr;
	Mat33 Rcurr;
	Float3 tcurr;
	Mat33 Rprev_inv;
	Float3 tprev;
} CombinedParams;

typedef struct
{
	const int step;
	const int length;
} TranformReductionParams;

inline int flattenedThreadId()
{
	return get_local_id(2) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);
	//return threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

inline void reduce(__local float_type* buffer, const int CTA_SIZE_)
{
	int tid = flattenedThreadId();
	float_type val = buffer[tid];

	if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 128) { if (tid <  64) buffer[tid] = val = val + buffer[tid + 64]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 64) { if (tid <  32) buffer[tid] = val = val + buffer[tid + 32]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 32) { if (tid <  16) buffer[tid] = val = val + buffer[tid + 16]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 16) { if (tid <  8) buffer[tid] = val = val + buffer[tid + 8]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 8) { if (tid <  4) buffer[tid] = val = val + buffer[tid + 4]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 4) { if (tid <  2) buffer[tid] = val = val + buffer[tid + 2]; barrier(CLK_LOCAL_MEM_FENCE); }
	if (CTA_SIZE_ >= 2) { if (tid <  1) buffer[tid] = val = val + buffer[tid + 1]; barrier(CLK_LOCAL_MEM_FENCE); }
}

inline float3 packTofloat3(__constant float* v) {
	return (*((__constant float3*)&v[0]));
}

inline float3 Mat9xfloat3(__constant float* m, float3* v) {
	return (float3)( dot( (*((__constant float3*)&m[0])), *v), dot( (*((__constant float3*)&m[3])), *v), dot( (*((__constant float3*)&m[6])), *v) );
}
inline float3 Mat33xfloat3(Mat33 R, float3 v) {
	float3 temp;
	temp.x = (R.data[0].x * v.x + R.data[0].y * v.y + R.data[0].z * v.z);
	temp.y = (R.data[1].x * v.x + R.data[1].y * v.y + R.data[1].z * v.z);
	temp.z = (R.data[2].x * v.x + R.data[2].y * v.y + R.data[2].z * v.z);
	return temp;
	//return (float3)(dot((*((__constant float3*)&m[0])), *v), dot((*((__constant float3*)&m[3])), *v), dot((*((__constant float3*)&m[6])), *v));
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

bool search(int x,
	int y,
	float3* n,
	float3* d,
	float3* s,
	__read_only image2d_t  nmap_curr_data_,
	__read_only image2d_t  vmap_curr_data_,
	__read_only image2d_t  nmap_g_prev_data_,
	__read_only image2d_t  vmap_g_prev_data_,
	__constant CombinedParams *g_params
	)
{
	const int cols = get_global_size(0);
	const int rows = get_global_size(1);
	CombinedParams params = *g_params;
	Intr intr = params.intr;
	Mat33 Rcurr = params.Rcurr;
	float3 tcurr = { params.tcurr.x, params.tcurr.y, params.tcurr.z };
	Mat33 Rprev_inv = params.Rprev_inv;
	float3 tprev = { params.tprev.x, params.tprev.y, params.tprev.z };
	int2 coord = (int2)(x, y);
	float3 ncurr = read_imagef(nmap_curr_data_, sampler, coord).xyz;

	if (isnan(ncurr.x))
		return (false);

	float3 vcurr = read_imagef(vmap_curr_data_, sampler, coord).xyz;

	float3 vcurr_g = Mat33xfloat3(Rcurr, vcurr) + tcurr;
	float3 temp = vcurr_g - tprev;
	float3 vcurr_cp = Mat33xfloat3(Rprev_inv, temp);
	float vcurr_cp_z_inv = native_divide(1, vcurr_cp.z);
	int2 ukr;//projection
	ukr.x = convert_int(vcurr_cp.x * intr.fx * vcurr_cp_z_inv + intr.cx);
	ukr.y = convert_int(vcurr_cp.y * intr.fy * vcurr_cp_z_inv + intr.cy);
	if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
	return (false);

	float3 nprev_g = read_imagef(nmap_g_prev_data_, sampler, ukr).xyz;
	
	if (isnan(nprev_g.x))
		return (false);

	float3 vprev_g = read_imagef(vmap_g_prev_data_, sampler, ukr).xyz;

	float dist = fast_length(vprev_g - vcurr_g);
	if (dist > params.distThres)
		return (false);

	float3 ncurr_g = Mat33xfloat3(Rcurr, ncurr);

	float sine = fast_length(cross(ncurr_g, nprev_g.xyz)); //if(x==129&&y==108) printf("cl_search_body x %d, y %d, cols %d, rows %d, distThres %f angleThres %f ncurr.x %f vcurr.x %f vcurr.y %f vcurr.z %f dist %f sine %f \n", x, y, cols, rows, distThres, angleThres, ncurr.x, vcurr.x, vcurr.y, vcurr.z, dist, sine);

	if (sine >= params.angleThres)
		return (false);
	*n = nprev_g;
	*d = vprev_g;
	*s = vcurr_g;
	return (true);
}

__kernel void combinedKernel(__global float_type* gbuf_data_,
	__read_only image2d_t  nmap_curr_data_,
	__read_only image2d_t  vmap_curr_data_,
	__read_only image2d_t  nmap_g_prev_data_,
	__read_only image2d_t  vmap_g_prev_data_,
	__constant CombinedParams* params
	) {
	const int cols = get_global_size(0);
	const int rows = get_global_size(1);

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const size_t gbuf_step = get_num_groups(0) * get_num_groups(1);
	float3 n; float3 d; float3 s;
	bool found_coresp = false;

	const int CTA_SIZE_X = 32;
	const int CTA_SIZE_Y = 8 / 2;

	if (x < cols && y < rows)
		found_coresp = search(x, y, &n, &d, &s,
		nmap_curr_data_,
		vmap_curr_data_,
		nmap_g_prev_data_,
		vmap_g_prev_data_,
		params);

	float row[7];
	if (found_coresp) {
		float3 temp2 = cross(s, n);
		row[0] = temp2.x; row[1] = temp2.y; row[2] = temp2.z;
		row[3] = n.x; row[4] = n.y; row[5] = n.z;
		row[6] = dot(n, d - s);
	}
	else {
		row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
	}

	//const int CTA_SIZE = CTA_SIZE_X(32) * CTA_SIZE_Y(8);
	__local float_type smem[32 * 8 / 2];

	int tid = flattenedThreadId();
	int shift = 0;
	for (int i = 0; i < 6; ++i)//rows
	{
		//#pragma unroll
		for (int j = i; j < 7; ++j)//cols + b
		{
			smem[tid] = row[i] * row[j];
			barrier(CLK_LOCAL_MEM_FENCE);

			reduce(smem, 32 * 8 / 2);

			if (tid == 0) {
				(gbuf_data_ + shift++ * gbuf_step)[get_group_id(0) + get_num_groups(0) * get_group_id(1)] = smem[0];
			}
		}
	}
}

__kernel void TransformEstimatorKernel2(__global const float_type* data,
	__global float_type* output,
	__global TranformReductionParams* params
	)
{
	const int x = get_group_id(0);
	const int step = params->step;
	const int length = params->length;

	int idx = step * x;
	__global float_type* beg = (data + idx);
	__global float_type* end = beg + length;

	int tid = get_local_id(0);

	float_type sum = 0.f;
	for (__global float_type *t = beg + tid; t < end; t += 512 / 4)
	{
		sum += *t;
	}
	__local float_type smem[512 / 4];

	smem[tid] = sum;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	reduce(smem, 512 / 4);

	if (tid == 0) {
		output[x] = smem[0];
	}
}
