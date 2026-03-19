typedef struct {
	float x;
	float y;
	float z;
} Float3;

typedef struct {
	Float3 data[3];
} Mat33;
struct Intr_inv
{
	float fx_inv;
	float fy_inv;
	float cx;
	float cy;
};

struct TranformMapParams
{
	Mat33 Rmat;
	Float3 tvec;
};
typedef struct { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; } Float12;
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void computeVmap(__global const ushort* depth,
	__write_only image2d_t  vmap_image,
	__global struct Intr_inv* intr_inv
	)
{
	const int u = get_global_id(0);
	const int v = get_global_id(1);
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	struct Intr_inv l_intr_inv = *intr_inv;
	int w = get_global_size(0);
	int h = get_global_size(1);

	int depthidx = w * v + u;
	float4 vmap = { NAN, NAN, NAN, 1.0f };

	if (u < w && v < h)
	{
		float z = native_divide(depth[depthidx] , 1000.f); // load and convert: mm -> meters

		if (z != 0.0)
		{
			vmap.x = z * (u - l_intr_inv.cx) * l_intr_inv.fx_inv;
			vmap.y = z * (v - l_intr_inv.cy) * l_intr_inv.fy_inv;
			vmap.z = z;
		}
		write_imagef(vmap_image, coord, vmap);
	}
}

__kernel void computeNmap(__read_only image2d_t vmap_image,
	__write_only image2d_t nmap_image
	)
{
	const int u = get_global_id(0);
	const int v = get_global_id(1);
	int2 coord00 = (int2)(get_global_id(0), get_global_id(1));
	int2 coord01 = (int2)(get_global_id(0), get_global_id(1) + 1);
	int2 coord10 = (int2)(get_global_id(0) + 1, get_global_id(1));
	int w = get_global_size(0);
	int h = get_global_size(1);
	float4 nmap = { NAN, NAN, NAN, 1.0f };

	if (u >= w || v >= h)
		return;

	if (u == w - 1 || v == h - 1)
	{
		write_imagef(nmap_image, coord00, nmap);
		return;
	}

	float4 v00 = read_imagef(vmap_image, sampler, coord00);
	float4 v01 = read_imagef(vmap_image, sampler, coord01);
	float4 v10 = read_imagef(vmap_image, sampler, coord10);

	if (!isnan(v00.x) && !isnan(v01.x) && !isnan(v10.x))
	{

		float4 r = fast_normalize(cross(v01 - v00, v10 - v00));
		nmap = r;
	}
	write_imagef(nmap_image, coord00, nmap);
}

__kernel void transformVMap(
	__read_only image2d_t vmap_src,
	__write_only image2d_t vmap_dst,
	__constant struct TranformMapParams* params
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	struct TranformMapParams l_params = *params;
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= w || y >= h)
	{
		return;
	}
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	///Vertex MAP																	
	Mat33 R = l_params.Rmat;
	float3 T = { l_params.tvec.x, l_params.tvec.y, l_params.tvec.z };
	float4 vsrc = { NAN, NAN, NAN, 1.0f };
	float4 vdst = { NAN, NAN, NAN, 1.0f };
	vsrc = read_imagef(vmap_src, sampler, coord);
	if (!isnan(vsrc.x))
	{

		float3 vtemp = { (R.data[0].x * vsrc.x + R.data[0].y * vsrc.y + R.data[0].z * vsrc.z),
			(R.data[1].x * vsrc.x + R.data[1].y * vsrc.y + R.data[1].z * vsrc.z),
			(R.data[2].x * vsrc.x + R.data[2].y * vsrc.y + R.data[2].z * vsrc.z) };
		vdst.xyz = vtemp + T;
	}
	write_imagef(vmap_dst, coord, vdst);
}

__kernel void transformNMap(
	__read_only image2d_t nmap_src,
	__write_only image2d_t nmap_dst,
	__constant struct TranformMapParams* params
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	struct TranformMapParams l_params = *params;
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= w || y >= h)
	{
		return;
	}
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	///Vertex MAP																	
	Mat33 R = l_params.Rmat;
	float3 T = { l_params.tvec.x, l_params.tvec.y, l_params.tvec.z };
	//////////////////////////////nmap transform
	float4 nsrc = { NAN, NAN, NAN, 1.0f };
	float4 ndst = { NAN, NAN, NAN, 1.0f };
	nsrc = read_imagef(nmap_src, sampler, coord);

	if (!isnan(nsrc.x))
	{
		float3 ntemp = { (R.data[0].x * nsrc.x + R.data[0].y * nsrc.y + R.data[0].z * nsrc.z),
			(R.data[1].x * nsrc.x + R.data[1].y * nsrc.y + R.data[1].z * nsrc.z),
			(R.data[2].x * nsrc.x + R.data[2].y * nsrc.y + R.data[2].z * nsrc.z) };
		ndst.xyz = ntemp;
	}
	write_imagef(nmap_dst, coord, ndst);
}

//resize map
const sampler_t sampler2 = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;
__kernel void resizeVMaps(
	__read_only image2d_t src,
	__write_only image2d_t dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	int sw = w * 2;
	int sh = h * 2;

	int sx = x * 2;
	int sy = y * 2;
	float2 coordTemp = (float2)(sx, sy) + (float2)0.5;
	float2 srcCoord = coordTemp / (float2)(sw, sh);
	int2 dstCoord = (int2)(x, y);

	float4 v = read_imagef(src, sampler2, srcCoord);
	write_imagef(dst, dstCoord, v);
}

//resize map

__kernel void resizeNMaps(
	__read_only image2d_t src,
	__write_only image2d_t dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	int sw = w * 2;
	int sh = h * 2;

	int sx = x * 2;
	int sy = y * 2;
	float2 coordTemp = (float2)(sx, sy) + (float2)0.5;
	float2 srcCoord = coordTemp / (float2)(sw, sh);

	int2 dstCoord = (int2)(x, y);
	float4 v = read_imagef(src, sampler2, srcCoord);

	float4 n = fast_normalize(v);

	write_imagef(dst, dstCoord, n);
}

__constant const float p1 = 0.000000000000008;
__constant const float p2 = 0.000000000068739;
__constant const float p3 = 0.000000219991933;
__constant const float p4 = 0.000320891650199;
__constant const float p5 = 1.228570861853840;
__constant const float p6 = 58.361653233543200; 

//convert to real distance
//y = 0.000000000000008 x5 - 0.000000000068739 x4 + 0.000000219991933 x3 - 0.000320891650199 x2 + 1.228570861853840 x - 58.361653233543200 

ushort depth_convert(const ushort x)
{
	return convert_ushort(p1 * convert_float(pow(convert_float(x), 5.f)) - p2 * convert_float(pow(convert_float(x), 4.f)) + p3 * convert_float(pow(convert_float(x), 3.f)) - p4 * convert_float(pow(convert_float(x), 2.f)) + p5 * convert_float(x) - p6); 
}

__kernel void resizeDepthMaps(
	__global ushort* src,
	__global ushort* dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	int sw = w * 2;
	int sh = h * 2;

	int sx = x * 2;
	int sy = y * 2;

	
	if( src[sy*sw+sx] > 500 )
		dst[y*w+x] = depth_convert(src[sy*sw+sx]);
	else
	    dst[y*w+x] = 0;
	
	//dst[y*w+x] = src[sy*sw+sx];
}

__kernel void resizeDepthMaps2(
	__read_only image2d_t src,
	__write_only __global ushort* dst
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	int sw = w * 2;
	int sh = h * 2;

	int sx = x * 2;
	int sy = y * 2;

	int2 coord = (int2)(get_global_id(0)*2, get_global_id(1)*2);
	float depth = read_imagef(src, sampler, coord).x;
	dst[y*w+x] = (ushort)(depth);
}

__kernel void convertMap(
	__read_only image2d_t vmap_src,
	__global float* dst//float4
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	float4 qnan = { NAN, NAN, NAN, NAN };
	int2 coord = (int2)(x, y);
	int dw = 4 * w;
	int dx = 4 * x;
	int dst_idx = mad24(dw, y, dx);
	float4 t = read_imagef(vmap_src, sampler, coord);
	if(isnan(t.x))
	{
		t = qnan;
	}
	*(__global float4*)&dst[dst_idx] = t;
	//dst[y * w + x] = t;
}
__kernel void convertMap2(
	__read_only image2d_t vmap_src,
	__global float* dst //float8
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	int w = get_global_size(0);
	int h = get_global_size(1);
	if (x >= get_global_size(0) || y >= get_global_size(1))
	{
		return;
	}

	float4 qnan = { NAN, NAN, NAN, NAN };
	int2 coord = (int2)(x, y);
	int dw = 8 * w;
	int dx = 8 * x;
	int dst_idx = mad24(dw, y, dx);
	float4 t = read_imagef(vmap_src, sampler, coord);
	if (isnan(t.x))
	{
		t = qnan;
	}
	////

	(*(__global float8*)&dst[dst_idx]).s0123 = t;
	//dst[y * w + x].s0123 = t;
}



__kernel void mergePointNormal(
	__global float* cloud, //float4
	__global float* normals, //float8
	__global float* output //float12
	)
{
	const int x = get_global_id(0);

	
	if (x >= get_global_size(0))
	{
		return;
	}

	float4 p = *(__global float4*)&cloud[x * sizeof(float4)];
	float8 n = *(__global float8*)&normals[x * sizeof(float8)];
	Float12 o;
	o.x = p.x;
	o.y = p.y;
	o.z = p.z;
	
	o.normal_x = n.x;
	o.normal_y = n.y;
	o.normal_z = n.z;
	*(__global Float12*)&output[x * sizeof(Float12)] = o;
}