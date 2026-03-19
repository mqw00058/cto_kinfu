/*
 *  opencl_utils.h
 *  Simple OpenCL configuration
 *
 *  AUTHOR : Jeongyoun Yi <jeongyoun.yi@lge.com>
 *  UPDATE : 2015-02-06
 *
 *  Copyright (c) 2015 Advanced Imaging Technology, Creative Innovation Center. All rights reserved.
 */

#include <pcl/gpu/kinfu/opencl_utils.h>
#include <assert.h>
#include <algorithm>


opencl_utils* opencl_utils::g_opencl_utils = NULL;
CL_Device g_select_device;
void exit_break()
{
	exit(EXIT_FAILURE);
}

DLL_EXPORT opencl_utils* opencl_utils::get(){
	if (g_opencl_utils == NULL)
	{
		g_opencl_utils = new opencl_utils();
	}
	return g_opencl_utils;
}

DLL_EXPORT opencl_utils* opencl_utils::get(CL_Device d){
	g_select_device = d;
	if (g_opencl_utils == NULL)
	{
		g_opencl_utils = new opencl_utils();
	}
	return g_opencl_utils;
}

opencl_utils::opencl_utils() :m_event_wait(false)
{
	if (g_select_device == NVIDIA)
		openclInit_NVIDIA();
	else // if (g_select_device == INTEL)
		openclInit_INTEL();
}

opencl_utils::~opencl_utils()
{
	clReleaseCommandQueue(m_command_queue);
	clReleaseCommandQueue(m_command_queue_sub[0]);
	clReleaseCommandQueue(m_command_queue_sub[1]);
	clReleaseCommandQueue(m_command_queue_sub[2]);
	clReleaseContext(m_context);
};

void opencl_utils::openclInit_NVIDIA()
{
    cl_int err = CL_SUCCESS;

    m_context = createContextFromType(CL_DEVICE_TYPE_GPU, &err, 0, 0, -1, -1, &m_platform_id);
    
    // debugging
    printPlatformInfo(m_platform_id);
    
    m_device = getDevice(m_context);
    
    // debugging
    printDeviceInfo(m_device);
    
    m_command_queue = clCreateCommandQueue(m_context, m_device, (0 | CL_QUEUE_PROFILING_ENABLE /*| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/), &err);
    
    // using multiple queues?
	m_command_queue_sub[0] = clCreateCommandQueue(m_context, m_device, (0 | CL_QUEUE_PROFILING_ENABLE /*| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/), &err);
	m_command_queue_sub[1] = clCreateCommandQueue(m_context, m_device, (0 | CL_QUEUE_PROFILING_ENABLE /*| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/), &err);
	m_command_queue_sub[2] = clCreateCommandQueue(m_context, m_device, (0 | CL_QUEUE_PROFILING_ENABLE /*| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/), &err);
}

void opencl_utils::openclInit_INTEL() {
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	//bilateral_dst = NULL;
	m_device = NULL;
	m_context = NULL;
	m_command_queue = NULL;

	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	printf("ret_num_platforms: %d\n", ret_num_platforms);

	cl_platform_id *platform_id;
	platform_id = new cl_platform_id[ret_num_platforms];

	ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);

	char Buffer[512];              
	size_t Size;                    

	m_platform_id = platform_id[0];

	ret = clGetDeviceIDs(m_platform_id, CL_DEVICE_TYPE_GPU, 1, &m_device, &ret_num_devices);
	clGetDeviceInfo(m_device, CL_DEVICE_NAME, 512, Buffer, &Size);
	printf("CL_DEVICE_NAME : %s\n", Buffer);
#ifdef __ANDROID__
	LOGV("CL_DEVICE_NAME : %s\n", Buffer);
#endif
	cl_uint temp;
	clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &temp, &Size);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS : %d\n", temp);
#ifdef __ANDROID__
	LOGV("CL_DEVICE_MAX_COMPUTE_UNITS : %d\n", temp);
#endif
	clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &temp, &Size);
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %d\n", temp);
#ifdef __ANDROID__
	LOGV("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %d\n", temp);
#endif
	size_t temp2[3];
	clGetDeviceInfo(m_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*temp, temp2, NULL);
	printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %d %d %d\n", temp2[0], temp2[1], temp2[2]);
#ifdef __ANDROID__
	LOGV("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : %d %d %d\n", temp2[0], temp2[1], temp2[2]);
#endif
	size_t extensionSize;
	clGetDeviceInfo(m_device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
	char* extensions = (char*)malloc(extensionSize);
	clGetDeviceInfo(m_device, CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
	printf("CL_DEVICE_EXTENSIONS : %s\n",extensions);
#ifdef __ANDROID__
	LOGV("CL_DEVICE_EXTENSIONS : %s\n", extensions);
#endif
	free(extensions);

	cl_context_properties contextProperties[] =
	{
#ifdef __ANDROID__
		CL_CONTEXT_PLATFORM, (cl_context_properties)m_platform_id,
		CL_GL_CONTEXT_KHR, (cl_context_properties)eglGetCurrentContext(),
		CL_EGL_DISPLAY_KHR, (cl_context_properties)eglGetCurrentDisplay(),
		0
#else
		NULL
#endif
	};

	/* Create OpenCL context */
	m_context = clCreateContext(contextProperties, 1, &m_device, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
		printf("clCreateContext %d\n", ret);

	/* Create Command Queue */
	m_command_queue = clCreateCommandQueue(m_context, m_device, 0, &ret);
	if (ret != CL_SUCCESS)
		printf("clCreateCommandQueue %d\n", ret);
	m_command_queue_sub[0] = clCreateCommandQueue(m_context, m_device, 0, &ret);
	if (ret != CL_SUCCESS)
		printf("clCreateCommandQueue %d\n", ret);
	m_command_queue_sub[1] = clCreateCommandQueue(m_context, m_device, 0, &ret);
	if (ret != CL_SUCCESS)
		printf("clCreateCommandQueue %d\n", ret);
	m_command_queue_sub[2] = clCreateCommandQueue(m_context, m_device, 0, &ret);
	if (ret != CL_SUCCESS)
		printf("clCreateCommandQueue %d\n", ret);

	delete(platform_id);

	return;
}

cl_context opencl_utils::createContextFromType(cl_device_type device_type, cl_int* pErr, void* gl_context, void* gl_dc, int preferred_device_idx, int preferred_platform_idx, cl_platform_id* platform_id)
{
	cl_uint num_platforms;
	cl_context context = 0;
	unsigned int i;

	cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
	if(err != CL_SUCCESS)
	{
		goto RETURN_ERROR;
	}

	if(num_platforms > 0)     
	{
		cl_platform_id* platforms = (cl_platform_id*) malloc (sizeof(cl_platform_id)*num_platforms);
		err = clGetPlatformIDs(num_platforms, platforms, NULL);
		if(err != CL_SUCCESS)
		{
			free(platforms);
			goto RETURN_ERROR;
		}
		
		// check platforms
        for(i = 0; i < num_platforms; ++i)
        {
            char pbuf[128];
            
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
            if(err != CL_SUCCESS)
            {
                goto RETURN_ERROR;
            }
			printf("%s\n", pbuf);
            
            if(preferred_platform_idx >= 0 && i == preferred_platform_idx)
            {
                cl_platform_id temp = platforms[0];
                platforms[0] = platforms[i];
                platforms[i] = temp;
                break;
            }
            //else if(!strcmp(pbuf, __support_platform_vendor))
            else if(isSupportedVendor(pbuf))
            {
                cl_platform_id temp = platforms[0];
                platforms[0] = platforms[i];
                platforms[i] = temp;
                break;
            }
        }

		// assign context from supported platforms
		for(i = 0; i < num_platforms; ++i)
        {
            cl_platform_id platform = platforms[i];
            
            context = createContextFromPlatform(platform, device_type, pErr, gl_context, gl_dc, preferred_device_idx, preferred_platform_idx);
    
            if(*pErr == CL_SUCCESS)
            {
                *platform_id = platform;
                break;
            }
        }

		free (platforms);    
	}

	return context;

RETURN_ERROR:
	if(pErr != NULL) *pErr = err;
		return NULL;
}


cl_context opencl_utils::createContextFromPlatform(cl_platform_id platform, cl_device_type device_type, cl_int* pErr, void* gl_context, void* gl_dc, int preferred_device_idx, int preferred_platform_idx)
{
    cl_context context = 0;
    
    cl_uint num_devices;
    cl_device_id devices[HC_MAX_CL_DEVICES];
    
    cl_int err;
    
    err =  clGetDeviceIDs(platform, device_type, HC_MAX_CL_DEVICES, devices, &num_devices);
    
    if(preferred_device_idx >= 0 && preferred_device_idx < num_devices)
    {
        context = clCreateContext(NULL, 1, &devices[preferred_device_idx], NULL, NULL, &err);
    }
    else
    {
        context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
    }
    
    return context;
}

cl_device_id opencl_utils::getDevice(cl_context context)
{
    size_t sz;
    cl_int err = CL_SUCCESS;
    
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &sz);
    
    cl_device_id* devices = (cl_device_id*)malloc(sz);
    
    clGetContextInfo(context, CL_CONTEXT_DEVICES, sz, devices, NULL);
    
    for(int i = 0; i < sz/sizeof(cl_device_id); ++i)
    {
        cl_device_id dev = devices[i];
        
        if(isPreferredDevice(dev))
        {
            free(devices);
            return dev;
        }
    }
    
    cl_device_id device = devices[0];
    free(devices);
    
    return device;
}

bool opencl_utils::isPreferredDevice(cl_device_id device)
{
    cl_int err;
    char str[HC_MAX_STRING_BUF];
    
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, HC_MAX_STRING_BUF, str, NULL);
    
    return (strstr(str, __preferred_device_name) != NULL);
}


bool opencl_utils::isSupportedVendor(const char* platform_vendor_name)
{    
    int sz_vendor = PLATFORM_VENDOR_SIZE;
    
    for(int i = 0; i < sz_vendor; ++i)
        if(!strcmp(platform_vendor_name, __support_platform_vendor[i]))
        {
            return true;
        }
    
    return false;
}


// For debugging purpose
void opencl_utils::printPlatformInfo(cl_platform_id platform)
{
    cl_int err;
    char str[HC_MAX_STRING_BUF];
    
    printf("\n##Platform info##\n");    
    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_PLATFORM_VENDOR:    %s\n", str);
    
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_PLATFORM_NAME:      %s\n", str);
    
    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_PLATFORM_VERSION:   %s\n", str);
}

void opencl_utils::printDeviceInfo(cl_device_id device)
{
    cl_int err;
    char str[HC_MAX_STRING_BUF];
    size_t param[3];
    cl_uint size_uint;
    cl_ulong size_ulong;
    
    printf("\n##Device info##\n");   
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_DEVICE_NAME:        %s\n", str);
    
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_DEVICE_VERSION:     %s\n", str);
    
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, HC_MAX_STRING_BUF, str, NULL);
    printf("\tCL_DEVICE_EXTENSIONS:  %s\n", str);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(param[0]), &param[0], NULL);
    printf("\tCL_DEVICE_MAX_WORK_GROUP_SIZE:        %20d\n", param[0]);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(param), param, NULL);
    printf("\tCL_DEVICE_MAX_WORK_ITEM_SIZES:        %20d, %20d, %20d\n", param[0], param[1], param[2]);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(param[0]), &param[0], NULL);
    printf("\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:   %20d\n", param[0]);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(size_uint), &size_uint, NULL);
    printf("\tCL_DEVICE_MAX_READ_IMAGE_ARGS:        %20u\n", size_uint);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(size_uint), &size_uint, NULL);
    printf("\tCL_DEVICE_MAX_WRITE_IMAGE_ARGS:       %20u\n", size_uint);
    
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(param[0]), &param[0], NULL);
    printf("\tCL_DEVICE_IMAGE2D_MAX_WIDTH:          %20d\n", param[0]);
    
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(param[0]), &param[0], NULL);
    printf("\tCL_DEVICE_IMAGE2D_MAX_HEIGHT:         %20d\n", param[0]);
    
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(size_uint), &size_uint, NULL);
    printf("\tCL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:  %20u\n", size_uint);
    
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(size_ulong), &size_ulong, NULL);
    printf("\tCL_DEVICE_GLOBAL_MEM_CACHE_SIZE:      %20llu\n", size_ulong);
    
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_ulong), &size_ulong, NULL);
    printf("\tCL_DEVICE_GLOBAL_MEM_SIZE:            %20llu\n", size_ulong);
    
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_ulong), &size_ulong, NULL);
    printf("\tCL_DEVICE_LOCAL_MEM_SIZE:             %20llu\n", size_ulong);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_uint), &size_uint, NULL);
    printf("\tCL_DEVICE_MAX_COMPUTE_UNITS:          %20u\n", size_uint);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_ulong), &size_ulong, NULL);
    printf("\tCL_DEVICE_MAX_MEM_ALLOC_SIZE:         %20llu\n", size_ulong);

	printf("\n");
}


const char* opencl_utils::getErrorString(cl_int error)
{
    switch(error){
            // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            
            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
            
            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

cl_program opencl_utils::buildProgramFromFile(const char* path, const char* option)
{
	long length;
	char* source = NULL;

#ifdef KINFU_CL_DIR
	// Use the absolute kernel directory set at compile time.
	// Extract just the filename from the (possibly relative) path.
	const char* slash = strrchr(path, '/');
	const char* fname = slash ? slash + 1 : path;
	std::string abs_path = std::string(KINFU_CL_DIR) + "/" + fname;
	source = readText(abs_path.c_str(), &length);
#else
	source = readText(path, &length);
#endif

	std::cout << path << std::endl;
	cl_program program = buildProgram(source, length, option);

	free(source);

	return program;
}

cl_program opencl_utils::buildProgram(const char* source, size_t size, const char* option)
{
	cl_int err;

	cl_program program = clCreateProgramWithSource(m_context, 1, (const char**)&source, &size, &err); 
	assert(err == CL_SUCCESS);
	cl_build_status build_status;

	//err = clBuildProgram(program, 1, &m_device, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);

	err = clBuildProgram(program, 1, &m_device, option, NULL, NULL);
	err = clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL); 

	size_t sz_log;
	err = clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz_log);
	char* log = (char*)malloc(sz_log + 1);
	err = clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, sz_log, log, NULL); 

	log[sz_log] = '\0';

	printf("## Build log ##\n %d", build_status);
	printf("%s\n", log);

	if (build_status != CL_SUCCESS)
	{
#ifdef __ANDROID__
		char filename[256];
		sprintf(filename, "/sdcard/Download/clError.txt\0");
		FILE *file_out = fopen(filename, "w");
		fprintf(file_out, "%s\n", log);
		fclose(file_out);
		sprintf(filename, "/sdcard/Download/clErrorSource.txt\0");
		file_out = fopen(filename, "w");
		fprintf(file_out, "%s\n", source);
		fclose(file_out);
#endif
		exit(0);
	}

	free((void*)log);

	return program;
}

cl_kernel opencl_utils::compileKernelFromFile(cl_program program, const char* name, cl_int* err)
{
	cl_kernel kernel;

	kernel = clCreateKernel(program, name, err);

	return kernel;
}

char* opencl_utils::readText(const char* path, long *cnt)
{
	FILE* fp = NULL;
	char* content = NULL;

	long count = 0;

	if (path)
	{
		fp = fopen(path, "rb");
		if (!fp)
		{
			printf("file open error!! %s\n", path);
			exit(0);
		}

		if (fp)
		{
			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);

			if (count > 0)
			{
				content = (char*)malloc(sizeof(char)*(count + 1));
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}

	*cnt = count;

	return content;
}

double opencl_utils::getElapsedTimeInMS(cl_event* event)
{
	cl_int err;
	cl_ulong begin = 0;
	cl_ulong end = 0;
	size_t sz_ret;

	err = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin, &sz_ret);
	err |= clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &sz_ret);

	checkErr(__FUNCTION__, err);

	return ((double)end - (double)begin)*(1.0e-6);
}
