//#define IMAGE_OPT
struct Intr
{
	float fx;
	float fy;
	float cx;
	float cy;
};

#define DIVISOR 32767

struct RayCasterParams
{
	float Rcurr[9];
	float tcurr[3];

	float time_step;
	float volume_size[3];
	float cell_size[3];

	struct Intr intr;

	int VOLUME_X;
	int VOLUME_Y;
	int VOLUME_Z;
};

float3 RayCaster_get_ray_next(struct RayCasterParams* params, int x, int y)
{
	float3 ray_next;
	ray_next.x = native_divide((x - params->intr.cx), params->intr.fx);
	ray_next.y = native_divide((y - params->intr.cy), params->intr.fy);
	ray_next.z = 1;
	return ray_next;
}

bool checkInds(struct RayCasterParams* params, const int3* g)
{
	return (g->x >= 0 && g->y >= 0 && g->z >= 0 && g->x < params->VOLUME_X && g->y < params->VOLUME_Y && g->z < params->VOLUME_Z);
}

float unpack_tsdf(short2 value)
{
	return (float)value.x / DIVISOR;
}

float RayCaster_readTsdf(__global const short2* volume, struct RayCasterParams* params, int x, int y, int z)
{
	return unpack_tsdf(volume[mad24(mad24(params->VOLUME_Y, z, y), params->VOLUME_X, x)]);
}

void RayCaster_getVoxel(struct RayCasterParams* params, const float3* point, int3* v)
{
	//int3 v;
	v->x = convert_int_rtn(native_divide(point->x, params->cell_size[0]));
	v->y = convert_int_rtn(native_divide(point->y, params->cell_size[1]));
	v->z = convert_int_rtn(native_divide(point->z, params->cell_size[2]));
	//return v;
}

float RayCaster_interpolateTrilinearyPoint(__global const short2* volume, struct RayCasterParams* params, float3* point)
{
	int3 g;
	RayCaster_getVoxel(params, point, &g);

	if (g.x <= 0 || g.x >= params->VOLUME_X - 1)
		return NAN;

	if (g.y <= 0 || g.y >= params->VOLUME_Y - 1)
		return NAN;

	if (g.z <= 0 || g.z >= params->VOLUME_Z - 1)
		return NAN;

	float vx = (g.x + 0.5f) * params->cell_size[0];
	float vy = (g.y + 0.5f) * params->cell_size[1];
	float vz = (g.z + 0.5f) * params->cell_size[2];

	g.x = (point->x < vx) ? (g.x - 1) : g.x;
	g.y = (point->y < vy) ? (g.y - 1) : g.y;
	g.z = (point->z < vz) ? (g.z - 1) : g.z;

	float a = native_divide((point->x - (g.x + 0.5f) * params->cell_size[0]), params->cell_size[0]);
	float b = native_divide((point->y - (g.y + 0.5f) * params->cell_size[1]), params->cell_size[1]);
	float c = native_divide((point->z - (g.z + 0.5f) * params->cell_size[2]), params->cell_size[2]);

	float res = RayCaster_readTsdf(volume, params, g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
		RayCaster_readTsdf(volume, params, g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
		RayCaster_readTsdf(volume, params, g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
		RayCaster_readTsdf(volume, params, g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
		RayCaster_readTsdf(volume, params, g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
		RayCaster_readTsdf(volume, params, g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
		RayCaster_readTsdf(volume, params, g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
		RayCaster_readTsdf(volume, params, g.x + 1, g.y + 1, g.z + 1) * a * b * c;
	return res;
}

float RayCaster_interpolateTrilineary(__global const short2* volume, struct RayCasterParams* params, const float3* origin, const float3* dir, float time)
{
	float3 result;
	result.x = dir->x * time + origin->x;
	result.y = dir->y * time + origin->y;
	result.z = dir->z * time + origin->z;

	return RayCaster_interpolateTrilinearyPoint(volume, params, &result);
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void rayCastKernel(
	__global const short2* volume,
	__write_only image2d_t vmap_dst,
	__write_only image2d_t nmap_dst,
	__constant struct RayCasterParams* g_params
	)
{

	struct RayCasterParams params = *g_params;

	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int2 coord = (int2)(x, y);
#ifdef USE_MULTIQUEUE
	if (x >= get_global_size(0)*2 || y >= get_global_size(1)*2)
#else
	if (x >= get_global_size(0) || y >= get_global_size(1))
#endif
		return;

	float4 vmap = { NAN, NAN, NAN, 1.0f };
	float4 nmap = { NAN, NAN, NAN, 1.0f };
	write_imagef(vmap_dst, coord, vmap);
	write_imagef(nmap_dst, coord, nmap);

	float3 ray_start = (float3)(params.tcurr[0], params.tcurr[1], params.tcurr[2]);

	float3 ray_next_temp = RayCaster_get_ray_next(&params, x, y);
	float3 ray_next;
	ray_next.x = (params.Rcurr[0] * ray_next_temp.x + params.Rcurr[1] * ray_next_temp.y + params.Rcurr[2] * ray_next_temp.z) + params.tcurr[0];
	ray_next.y = (params.Rcurr[3] * ray_next_temp.x + params.Rcurr[4] * ray_next_temp.y + params.Rcurr[5] * ray_next_temp.z) + params.tcurr[1];
	ray_next.z = (params.Rcurr[6] * ray_next_temp.x + params.Rcurr[7] * ray_next_temp.y + params.Rcurr[8] * ray_next_temp.z) + params.tcurr[2];


	float3 ray_dir = fast_normalize(ray_next - ray_start);


	float txmin = native_divide(((ray_dir.x > 0.f ? 0.f : params.volume_size[0]) - ray_start.x), ray_dir.x);
	float tymin = native_divide(((ray_dir.y > 0.f ? 0.f : params.volume_size[1]) - ray_start.y), ray_dir.y);
	float tzmin = native_divide(((ray_dir.z > 0.f ? 0.f : params.volume_size[2]) - ray_start.z), ray_dir.z);
	float time_start_volume_temp = fmax(fmax(txmin, tymin), tzmin);
	//getMinTime END
	
	float txmax = native_divide(((ray_dir.x > 0.f ? params.volume_size[0] : 0.f) - ray_start.x), ray_dir.x);
	float tymax = native_divide(((ray_dir.y > 0.f ? params.volume_size[1] : 0.f) - ray_start.y), ray_dir.y);
	float tzmax = native_divide(((ray_dir.z > 0.f ? params.volume_size[2] : 0.f) - ray_start.z), ray_dir.z);

	float time_exit_volume = fmin(fmin(txmax, tymax), tzmax);
	//getMaxTime END

	float min_dist = 0.f;         //in meters
	float time_start_volume = fmax(time_start_volume_temp, min_dist);
	if (time_start_volume >= time_exit_volume)   // bottleneck.. but i dont know
		return;

	float time_curr = time_start_volume;
	
	float3 ray_next2 = ray_start + ray_dir * time_curr;
	int3 g;
	RayCaster_getVoxel(&params, &ray_next2, &g);
	g.x = max(0, min(g.x, params.VOLUME_X - 1));
	g.y = max(0, min(g.y, params.VOLUME_Y - 1));
	g.z = max(0, min(g.z, params.VOLUME_Z - 1));

	float tsdf = RayCaster_readTsdf(volume, &params, g.x, g.y, g.z);
	///////////////////////////////////////////////////////////////////////
	//infinite loop guard
	const float max_time = 3 * (params.volume_size[0] + params.volume_size[1] + params.volume_size[2]);


	for (; time_curr < max_time; time_curr += params.time_step)
	{
		float tsdf_prev = tsdf;

		ray_next2 = ray_start + ray_dir * (time_curr + params.time_step);
		int3 g;
		RayCaster_getVoxel(&params, &ray_next2, &g);
		if (!checkInds(&params, &g))
			break;

		tsdf = RayCaster_readTsdf(volume, &params, g.x, g.y, g.z);

		if (tsdf_prev < 0.f && tsdf > 0.f)
			break;

		if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
		{
			float Ftdt = RayCaster_interpolateTrilineary(volume, &params, &ray_start, &ray_dir, time_curr + params.time_step);
			if (isnan(Ftdt))
				break;

			float Ft = RayCaster_interpolateTrilineary(volume, &params, &ray_start, &ray_dir, time_curr);
			if (isnan(Ft))
				break;

			float Ts = time_curr - native_divide(params.time_step * Ft, (Ftdt - Ft));

			float3 vetex_found = ray_start + ray_dir * Ts;

			vmap.xyz = vetex_found;
			write_imagef(vmap_dst, coord, vmap);

			ray_next2 = ray_start + ray_dir * time_curr;

			int3 g;
			RayCaster_getVoxel(&params, &ray_next2,&g);
			if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < params.VOLUME_X - 2 && g.y < params.VOLUME_Y - 2 && g.z < params.VOLUME_Z - 2)
			{
				float3 t;
				float3 n;

				t = vetex_found;
				t.x += params.cell_size[0];
				float Fx1 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				t = vetex_found;
				t.x -= params.cell_size[0];
				float Fx2 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				n.x = (Fx1 - Fx2);

				t = vetex_found;
				t.y += params.cell_size[1];
				float Fy1 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				t = vetex_found;
				t.y -= params.cell_size[1];
				float Fy2 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				n.y = (Fy1 - Fy2);

				t = vetex_found;
				t.z += params.cell_size[2];
				float Fz1 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				t = vetex_found;
				t.z -= params.cell_size[2];
				float Fz2 = RayCaster_interpolateTrilinearyPoint(volume, &params, &t);

				n.z = (Fz1 - Fz2);

				n = normalize(n);

				nmap.xyz = n;

				write_imagef(nmap_dst, coord, nmap);
			}
			break;
		}
	}
}