

#ifndef PCL_KINFU_KINFUTRACKER_CONFIG_HPP_
#define PCL_KINFU_KINFUTRACKER_CONFIG_HPP_

#ifndef __ANDROID__
#include <direct.h>
#endif

//#define DEBUG 0
#define __use_android__ 0


#ifdef __ANDROID__
#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "kinfu", __VA_ARGS__) 
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , "kinfu", __VA_ARGS__) 
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , "kinfu", __VA_ARGS__) 
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , "kinfu", __VA_ARGS__) 
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "kinfu", __VA_ARGS__) 
#else
#define LOGV(...)
#define LOGD(...) 
#define LOGI(...) 
#define LOGW(...) 
#define LOGE(...) 
#endif

typedef enum 
{
	QVGA,		// RGB:320 X 240	DEPTH :320 X 240 
	VGA ,		// RGB:640 X 480	DEPTH :640 X 480
	SXGA		// RGB:1280 X 960	DEPTH :640 X 480 
	
	
}Resolution;

typedef enum
{
	_15HZ,
	_30HZ, 
	_60HZ

}Hz;

enum CameraDevice
{
	KINECT,
	COMPACT_STREO,
	XTION_606,
	XTION_100,
	XTION_600,
	TOF
};

enum CL_Device
{
	NVIDIA,
	INTEL
};


struct FocalLen
{
	float rgbX;
	float rgbY;
	float depthX;
	float depthY;

	FocalLen() {}
	FocalLen(float rgbX_, float rgbY_, float depthX_, float depthY_) : rgbX(rgbX_), rgbY(rgbY_), depthX(depthX_), depthY(depthY_) {}

	void setValue(float rgbX_, float rgbY_, float depthX_, float depthY_)
	{
		this->rgbX = rgbX_;
		this->rgbY = rgbY_;
		this->depthX = depthX_;
		this->depthY = depthY_;
		
	}

	void operator()(CameraDevice device)
	{
		switch (device)
		{
		case KINECT:
			this->setValue(525.f, 525.f, 585.f, 585.f );
			break;
		case COMPACT_STREO:
			//this->setValue(497.824f, 495.16f, 497.824f, 495.16f);
			//this->setValue(517.666f, 517.255f, 517.666f, 517.255f);
			//this->setValue(650.55f, 650.648f, 650.55f, 650.648f); //6-2-13
			this->setValue(477.670254621091, 477.9867856946673, 493.8957189198446, 493.6558846924319); //6-2-10
			break;
		case XTION_606:
			this->setValue(525.f, 525.f, 586.23f, 586.93f );
			break;
		case XTION_600:
			this->setValue(525.f, 525.f, 586.23f, 586.93f);
			break;
		case XTION_100:
			this->setValue(525.f, 525.f, 562.23f, 562.93f );
			break;	
		case TOF:
			this->setValue(525.f, 525.f, 559.9944f, 559.9944f );
			break;
		default: break;

		}		

	}
	void operator/=(float div)
	{
		depthX /= div;
		depthY /= div;
		rgbX /= div;
		rgbY /= div;
	}		
};

struct Config
{
	Resolution resolution_;
	CameraDevice device_;
	Hz hz_;
	FocalLen focalLength_;
	CL_Device clDevice_;

	int rows_;
	int cols_;
	unsigned int levels_;
	int iters_[3];
	double min_delta_;

	Config(){}

	~Config(){}

	Config(char* filename)
	{
		char buf[512];
		FILE* fp;
		fp = fopen(filename, "r");

		if (fp == NULL)
		{
			char curPath[1024] = "";
#ifndef __ANDROID__			
			getcwd(curPath, 1024);			
#endif
	
			printf("[Warnning] Can'not read %s!! \nPlease place the file in this path : %s \n", filename, curPath);
			printf(">> Start program with the default configuration \n");

			setConfig(KINECT, VGA, _30HZ);
		}
		else
		{
			char* div = "\n\t ";
			char* params[7] = { "DEVICE", "RESOLUTION", "HZ", "ITERATION", "MIN_DELTA", "CL_DEVICE", "LEVELS" };

			bool(Config::*parsArgu[7])(char*) = { &Config::parseDevice, &Config::parseResolution, &Config::parseHZ,
				&Config::parseITERATION, &Config::parseMIN_DELTA, &Config::parseCL_DEVICE, &Config::parseLEVELS };

			int paramIndex;
			bool vaildParam = false;
			char* param;
			char* paramVal;
			while (fgets(buf, 1024, fp) != NULL)
			{
				param = strtok(buf, div);
				for (paramIndex = 0; paramIndex < 7; paramIndex++)
				{
					if (strcmp(param, params[paramIndex]) == 0)
					{
						paramVal = strtok(NULL, div);
						bool t = (this->*parsArgu[paramIndex])(paramVal);

						if (!t)
						{
							printf("[ERROR : kinfu_config.h] configuration parseError!!");
							return;
						}
					}
				}
			}
		}
		fclose(fp);
		printConfig();
	}	

	void setConfig(char* String)
	{

		char* div = "\n\t ";
		char* params[7] = { "DEVICE", "RESOLUTION", "HZ", "ITERATION", "MIN_DELTA", "CL_DEVICE", "LEVELS" };

		bool(Config::*parsArgu[7])(char*) = { &Config::parseDevice, &Config::parseResolution, &Config::parseHZ,
			&Config::parseITERATION, &Config::parseMIN_DELTA, &Config::parseCL_DEVICE, &Config::parseLEVELS };

		char* param;
		char* paramVal;
		char buf[512];
		char* pString[10] = { NULL, };
		int nLine = 0; 
		int stringLen = strlen(String);
		pString[nLine++] = String;

		for (int idx = 1; idx < stringLen; idx++)
		{
			if (String[idx] == '\n')
			{
				String[idx++] = NULL;
				pString[nLine++] = String + idx;					
			}
		}
		
		nLine = 0;
		while (pString[nLine])
		{
			strcpy(buf, pString[nLine]);
			nLine++;
			param = strtok(buf, div);
			for (int paramIndex = 0; paramIndex < 7; paramIndex++)
			{
				if (strcmp(param, params[paramIndex]) == 0)
				{
					paramVal = strtok(NULL, div);
					bool vaildParam = (this->*parsArgu[paramIndex])(paramVal);

					if (!vaildParam)
					{
						printf("[ERROR : kinfu_config.h] configuration parseError!!");
						return;
					}
					break;
				}
			}
		} 
		printConfig();
	}

	Config(CameraDevice device, Resolution resolution, Hz hz, int levels = 3, int * iters = NULL, double min_delta = 1e-4, CL_Device clDevice = NVIDIA)
	{
		resolution_ = resolution;
		device_ = device;
		hz_ = hz;
		min_delta_ = min_delta;
		setFocalLen(device);

		clDevice_ = clDevice;

		iters ? setIters(iters[0], iters[1], iters[2]) : setIters();
		levels_ = levels;

		setResolution(resolution);
		if (resolution == QVGA)
		{
			levels_ = 2;
			focalLength_ /= 2.0f;
		}
	}

	void setConfig(CameraDevice device, Resolution resolution, Hz hz, int levels = 3, int * iters = NULL, double min_delta = 1e-4, CL_Device clDevice = NVIDIA)
	{
		resolution_ = resolution;
		device_ = device;
		hz_ = hz;
		min_delta_ = min_delta;
		setFocalLen(device);

		clDevice_ = clDevice;	

		iters ? setIters(iters[0], iters[1], iters[2]) : setIters();
		levels_ = levels;

		setResolution(resolution);
		if (resolution == QVGA)
		{
			levels_ = 2;
			focalLength_ /= 2.0f;
		}
	}



	void setResolution(Resolution resolution)
	{
		switch (resolution_)
		{
		case QVGA:
			rows_ = 240; cols_ = 320;
			break;

		case VGA:
			rows_ = 480; cols_ = 640;
			break;

		case SXGA:
			rows_ = 480; cols_ = 640; //depth 640x480, RGB 1280 x 960			
			break;
		}
	}

	void setIters(int level0 = 20, int level1 = 16, int level2 = 8)
	{
		//const int iters[] = { 20, 16, 8 };
		//const int iters[] = { 100, 50, 4 };

		iters_[0] = level0;
		iters_[1] = level1;
		iters_[2] = level2;
	}

	void setFocalLen(CameraDevice device)
	{

		switch (device)
		{
		case KINECT:
			focalLength_(KINECT); break;
		case COMPACT_STREO:
			focalLength_(COMPACT_STREO); break;
		case XTION_606:
			focalLength_(XTION_606); break;
		case XTION_600:
			focalLength_(XTION_600); break;
		case XTION_100:
			focalLength_(XTION_100); break;
		case TOF:
			focalLength_(TOF); break;

		default: break;
		}

	}

	FocalLen getFocalLength() { return focalLength_; }
	CameraDevice getDevice() { return device_; }
	Resolution getResolution() { return resolution_; }
	Hz getHz() { return hz_; }
	CL_Device getCLDevice() { return clDevice_; }
	int* getIters() { return iters_; }
	double getMinDelta() { return min_delta_; }
	int getRows() { return rows_;  }
	int getCols() { return cols_; } 
	unsigned int getLevels() { return levels_; }


	bool parseDevice(char* paramValue)
	{
		char* devices[6] = { "KINECT", "COMPACT_STREO", "XTION_606","XTION_600", "XTION_100", "TOF" };

		for (int j = 0; j < 5; j++)
		{
			if (strcmp(paramValue, devices[j]) == 0)
			{
				device_ = (CameraDevice)j;
				setFocalLen(device_);
				return true;
			}
		}
		return false;
	}

	bool parseResolution(char* paramValue)
	{		
		char* resolution[3] = { "QVGA", "VGA", "SXGA" };
		for (int j = 0; j < 3; j++)
		{
			if (strcmp(paramValue, resolution[j]) == 0)
			{
				resolution_ = (Resolution)j;
				setResolution(resolution_);
				
				if (resolution_ == QVGA)
				{					
					focalLength_ /= 2.0f;
				}
				return true;
			}
		}
		return false;
	}

	bool parseHZ(char* paramValue)
	{
		int hz = atoi(paramValue);
		if (hz == 15)
		{
			hz_ = _15HZ;
			return true;
		}
		else if (hz == 30)
		{
			hz_ = _30HZ;
			return true;
		}
		else if (hz == 60)
		{
			hz_ = _60HZ;
			
			return true;
		}
		else
		{
			return false;
		}		
	}

	bool parseITERATION(char* paramValue)
	{	
		iters_[0] = atoi(strtok(paramValue, "- \t\n"));
		iters_[1] = atoi(strtok(NULL, "- \t\n"));
		iters_[2] = atoi(strtok(NULL, "- \t\n"));
		return true;
	}


	bool parseMIN_DELTA(char* paramValue)
	{
		min_delta_ = (double)atof(paramValue);
		return true;
	}

	bool parseCL_DEVICE(char* paramValue)
	{
		if (strcmp(paramValue, "NVIDIA") == 0)
		{
			clDevice_ = NVIDIA;
			return true;
		}
		else if (strcmp(paramValue, "INTEL") == 0)
		{
			clDevice_ = INTEL;
			return true;
		}
		else
		{
			return false;
		}
	}

	bool parseLEVELS(char* paramValue)
	{
		levels_ = atoi(paramValue);
		return true;
	}
	
	bool isVaildConfig()
	{
		if (device_ == COMPACT_STREO )
		{
			if (resolution_ == VGA && hz_ == _30HZ)
				return true;
			else
				return false;
		}

		if (device_ == KINECT)
		{
			if (resolution_ == VGA && hz_ == _30HZ)
				return true;
			else if (resolution_ == SXGA && hz_ == _15HZ)
				return true;
			else if (resolution_ == QVGA && hz_ == _30HZ)
				return true;			
			else
				return false;
		}
		if (device_ == XTION_100)
		{
		    if (resolution_ == QVGA && hz_ == _60HZ)
				return true;
			else if (resolution_ == QVGA && hz_ == _30HZ)
				return true;
			else if (resolution_ == VGA && hz_ == _30HZ)
				return true;
		}
		if (device_ == XTION_600)
		{
			if (resolution_ == QVGA && hz_ == _60HZ)
				return true;
			else if (resolution_ == QVGA && hz_ == _30HZ)
				return true;
			else if (resolution_ == VGA && hz_ == _30HZ)
				return true;
		}

		if (device_ == XTION_606)
		{
			if (resolution_ == QVGA && hz_ == _60HZ)
				return true;
			else if (resolution_ == QVGA && hz_ == _30HZ)
				return true;
			else if (resolution_ == VGA && hz_ == _30HZ)
				return true;
		}


		return false;
	}

	void printConfig()
	{
		char* params[7] = { "DEVICE", "RESOLUTION", "HZ", "ITERATION", "MIN_DELTA", "CL_DEVICE", "LEVELS" };
		char* devices[6] = { "KINECT", "COMPACT_STREO", "XTION_606", "XTION_600", "XTION_100", "TOF" };
		char* resolution[3] = { "QVGA", "VGA", "SXGA" };
		char* cldevice[2] = { "NVIDIA", "INTEL" };
		char* hz[3] = { "15", "30", "60" };


#ifdef __ANDROID__
		LOGD("------------------------------------------\n");
		LOGD("%s : %s \n", params[0], devices[device_]);
		LOGD("%s : %s \n", params[1], resolution[resolution_]);
		LOGD("%s : %s \n", params[2], hz[hz_]);
		LOGD("%s : %d %d %d \n", params[3], iters_[0], iters_[1], iters_[2]);
		LOGD("%s : %lf \n", params[4], min_delta_);
		LOGD("%s : %s \n", params[5], cldevice[clDevice_]);
		LOGD("%s : %d \n", params[6], levels_);
		LOGD("------------------------------------------\n");
#else 

		printf("------------------------------------------\n");
		printf("%s : %s \n", params[0], devices[device_]);
		printf("%s : %s \n", params[1], resolution[resolution_]);
		printf("%s : %s \n", params[2], hz[hz_]);
		printf("%s : %d %d %d \n", params[3], iters_[0], iters_[1], iters_[2]);
		printf("%s : %lf \n", params[4], min_delta_);
		printf("%s : %s \n", params[5], cldevice[clDevice_]);
		printf("%s : %d \n", params[6], levels_);
		printf("------------------------------------------\n");


#endif
	}

};







#endif /* PCL_KINFU_KINFUTRACKER_CONFIG_HPP_ */
