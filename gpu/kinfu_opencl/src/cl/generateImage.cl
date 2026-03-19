struct lightParams																																					 
{																																									 
	float pos[3];
	int number;					 																														 
};		
typedef struct {
	unsigned char x;
	unsigned char y;
	unsigned char z;
} Uchar3;

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void GI(__read_only image2d_t vmap_src,	
__read_only image2d_t nmap_src,	
__constant struct lightParams* params,
__global Uchar3* dst
)																																								 
{																																									 
	const int x = get_global_id(0);																																	 
	const int y = get_global_id(1);	
	struct lightParams l_params = *params;		
	int w = get_global_size(0);
	int h = get_global_size(1);
	float3 pos = { l_params.pos[0], l_params.pos[1], l_params.pos[2] };
	int number = l_params.number;

	if (x >= w || y >= h)								
	{																				
		return;																		
	}																																	 

	// float3 pos = l_params.pos[0];
	//  int number = l_params.number;
	int2 coord = (int2)(x, y);
	int dstidx = (w * y) + x;
	float3 v = read_imagef(vmap_src, sampler, coord).xyz;
	float3 n = read_imagef(nmap_src, sampler, coord).xyz;		

	uchar3 color = {0,0,0};		
	if (!isnan (v.x) && !isnan (n.x))
	{
		float weight = 1.f;
		//for (int i = 0; i < number; ++i)  //number is 1 so have to remove
		{
		float3 vec = normalize (pos - v);

		weight *= fabs (dot (vec, n));
		}
		int br = (int)(205 * weight) + 50;
		br = max (0, min (255, br));
		color = (br, br, br);
	}
	//possible?
	//dst[(y * w) + x] = (Uchar3)(color.x, color.y, color.z);
	dst[dstidx].x = color.x;
	dst[dstidx].y = color.y;
	dst[dstidx].z = color.z;
}


__kernel void GI2(__read_only image2d_t vmap_src,	
__read_only image2d_t nmap_src,	
__constant struct lightParams* params,
__write_only image2d_t dst
)																																								 
{																																									 
	const int x = get_global_id(0);																																	 
	const int y = get_global_id(1);	
	struct lightParams l_params = *params;		
	int w = get_global_size(0);
	int h = get_global_size(1);
	float3 pos = { l_params.pos[0], l_params.pos[1], l_params.pos[2] };
	int number = l_params.number;

	if (x >= w || y >= h)								
	{																				
		return;																		
	}																																	 

	// float3 pos = l_params.pos[0];
	//  int number = l_params.number;
	int2 coord = (int2)(x, y);
	int dstidx = (w * y) + x;
	float3 v = read_imagef(vmap_src, sampler, coord).xyz;
	float3 n = read_imagef(nmap_src, sampler, coord).xyz;		

	uchar3 color = {0,0,0};		
	if (!isnan (v.x) && !isnan (n.x))
	{
		float weight = 1.f;
		//for (int i = 0; i < number; ++i)  //number is 1 so have to remove
		{
		float3 vec = normalize (pos - v);

		weight *= fabs (dot (vec, n));
		}
		int br = (int)(205 * weight) + 50;
		br = max (0, min (255, br));
		color = (br, br, br);
	}
	write_imagef(dst, coord, (float4)(convert_float3(color)/(float3)255.0, 1) );
}

__kernel void paint3DView(__global Uchar3* colors,
__global Uchar3* dst
//__constant struct paintParams* params
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	const float colors_weight = 0.5f;

	if (x >= w || y >= h)
	{
		return;
	}
	int idx = (w * y) + x;
	uchar3 value;
	value.x = dst[idx].x;
	value.y = dst[idx].y;
	value.z = dst[idx].z;

	uchar3 color;
	color.x = colors[idx].x;
	color.y = colors[idx].y;
	color.z = colors[idx].z;

	if (value.x != 0 || value.y != 0 || value.z != 0)
	{
		float3 c = convert_float3(value) * (float3)(1.f - colors_weight) + convert_float3(color) * colors_weight;
		//float cx = value.x * (1.f - colors_weight) + color.x * colors_weight;
		//float cy = value.y * (1.f - colors_weight) + color.y * colors_weight;
		//float cz = value.z * (1.f - colors_weight) + color.z * colors_weight;
		value = min(255, max(0, convert_uchar3_rte(c)));
		//value.x = min(255, max(0, convert_int_rte(cx)));
		//value.y = min(255, max(0, convert_int_rte(cy)));
		//value.z = min(255, max(0, convert_int_rte(cz)));
	}
	dst[idx].x = value.x;
	dst[idx].y = value.y;
	dst[idx].z = value.z;
}