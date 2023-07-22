// OpenCL Test 4.cpp : Defines the entry point for the console application.
//

#include <CL\cl.hpp>
#include <CL\opencl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdlib.h>  
#include <time.h>
#include <random>

#include "Log.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 *
 * Mersenne Twister random number generator.
 *
 */
double randf(const double range = 1.0, bool sign = false)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	return dis(gen) * range * (1.0 + (double)sign) - range * sign;
}

struct v3
{
	float x, y, z;

	v3 operator * (float a)
	{
		return v3(x * a, y * a, z * a);
	}
	v3 operator / (float a)
	{
		if (a > 0) { return v3(x / a, y / a, z / a); }
	}


	inline v3(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
};


static const int localWorkGroupSize = 256;
static const int nt = 256 + 0;
static const int testDataSize = localWorkGroupSize * nt;
float dt = 1;
float dtinc = dt / 10;
float maxr = 10000;
float maxm = 1;
static const float eps = maxr / (100);
float initpxlsize = 0.9;
float pxlsize = initpxlsize;
float brtness = 0.015;
float brtnessinc = brtness / 10;
const char* ker = "NBODYBH2";
float zoom = 10;
float zoominc = 0.5;
int dustitr = 9;
float maxspd = 3 + 0.25 * (nt - 64) / 64;

cl_float4 mousepos = { 0.0f, 0.0f, 0.0f, 0.0f };

v3 target = v3(0.0f, 0.0f, 0.0f);

v3 partc = v3(1.0f, 0.9f, 0.9f);

cl_float4 p[testDataSize];

cl_float4 v[testDataSize];

cl_mem pBuffer, vBuffer;

cl_command_queue queue;
cl_kernel kernel;
std::vector<cl_device_id> deviceIds;
cl_int error;
cl_context context;

inline void CheckError(cl_int error);

std::string filename = "output/test";

int width = 360;
int height = 360;
int components = 1;
float w = width / (2.0f * maxr);
float h = height / (2.0f * maxr);

uint8_t* pixelData = new uint8_t[width * height * components];

// writes pixelData out to a PNG using stb_image_write.h
static void writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data)
{
	auto time = currentISO8601TimeUTC();
	std::replace(time.begin(),
		time.end(),
		':',
		'_');
	std::stringstream ss;
	if (filename.empty())
		ss << "img_" << time << ".png";
	else
		ss << filename << ".png";
	LOG("Writing out render to PNG image: " + ss.str());
	stbi_write_png(ss.str().c_str(), w, h, c, data, w * c);
}

static uint8_t sqrtColour(double x, double y, double gamma)
{
	return pow(x / y, 1.0 / gamma) * UCHAR_MAX;
}

void readData(int step)
{
	// Get the results back to the host
	CheckError(clEnqueueReadBuffer(queue, pBuffer, CL_TRUE, 0,
		sizeof(cl_float4) * testDataSize,
		p,
		0, nullptr, nullptr));

	for (int i = 0; i < width * height * components; ++i)
		pixelData[i] = 0;

	for (int i = 0; i < testDataSize; i++)
	{
		int x = (p[i].s[0] - (-maxr)) * w;
		int y = (p[i].s[1] - (-maxr)) * h;

		if (x >= 0 && x < width && y >= 0 && y < height)
			pixelData[(y * width + x)] += p[i].s[3] * 2;
	}

	double maxVal = 1;
	for (int i = 0; i < w * h; ++i)
		maxVal = std::max(maxVal, (double)pixelData[i]);

	for (int i = 0; i < w * h; ++i)
		pixelData[i] = sqrtColour(pixelData[i], maxVal, 2.0f);

	writeToPNG(filename.empty() ? "" : filename + std::to_string(step), width, height, components, pixelData);
}

std::string GetPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

std::string GetDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

inline void CheckError(cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::system("PAUSE");
		std::exit(1);
	}
}

std::string LoadKernel(const char* name)
{
	std::ifstream in(name);
	std::string result(
		(std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	return result;
}

cl_program CreateProgram(const std::string& source,
	cl_context context)
{
	size_t lengths[1] = { source.size() };
	const char* sources[1] = { source.data() };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
	CheckError(error);

	return program;
}

inline void doCalc(int steps = 1)
{
	for (int i = 0; i < steps; ++i)
	{
		clSetKernelArg(kernel, 0, sizeof(float), &dt);
		clSetKernelArg(kernel, 1, sizeof(float), &eps);
		clSetKernelArg(kernel, 2, localWorkGroupSize * sizeof(cl_float4), nullptr);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), &pBuffer);
		clSetKernelArg(kernel, 4, sizeof(cl_mem), &vBuffer);
		clSetKernelArg(kernel, 5, sizeof(int), &testDataSize);
		//	if (ker == "SBODY") 
		clSetKernelArg(kernel, 6, sizeof(cl_float4), &mousepos);

		cl_event eve;
		const size_t globalWorkSize[] = { testDataSize, 0, 0 };
		const size_t localWorkSize[] = { localWorkGroupSize, 0, 0 };
		CheckError(clEnqueueNDRangeKernel(queue, kernel, 1,
			nullptr,
			globalWorkSize,
			localWorkSize,
			0, nullptr, &eve));

		CheckError(clFlush(queue));
	}
}

int main(int argc, char* argv[])
{
	srand(time(NULL));

	for (int i = 0; i < testDataSize; ++i)
	{
		float a = randf() * 2 * 3.141; //((i % 2 + 1) * 1 * 3.141f);
		float r = randf() * maxr;
		float rinvr = maxspd;// (1 - r / 360) * 50;
		p[i].s[0] = randf(1.0, true) * maxr;// rcos(a)* r;
		p[i].s[1] = randf(1.0, true) * maxr; //sin(a) * r * 1;
		p[i].s[2] = 0;
		p[i].s[3] = 1; // randf() * 1;
		v3 dir = { p[i].s[0], p[i].s[1], 0 };
		float d = sqrt(dir.x * dir.x + dir.y * dir.y + 0.001);
		v[i].s[0] = 0; // -dir.y / d * maxspd;
		v[i].s[1] = 0; // dir.x / d * maxspd;
		v[i].s[2] = 0;
		v[i].s[3] = 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	}
	else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	}
	else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}
	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data(), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		cl_uint info = 0;
		std::cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;

		clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info), &info, nullptr);
		std::cout << "\t\t CL_DEVICE_MAX_COMPUTE_UNITS: " << info << std::endl;
		clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info), &info, nullptr);
		std::cout << "\t\t CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << info << std::endl;
		clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info), &info, nullptr);
		std::cout << "\t\t CL_DEVICE_MAX_WORK_GROUP_SIZE: " << info << std::endl;
		clGetDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(info), &info, nullptr);
		std::cout << "\t\t CL_DEVICE_MAX_WORK_ITEM_SIZES: " << info << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	context = clCreateContext(contextProperties, deviceIdCount,
		deviceIds.data(), nullptr, nullptr, &error);
	CheckError(error);

	std::cout << "Context created" << std::endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////



	cl_program program = CreateProgram(LoadKernel("saxpy.c"),
		context);

	CheckError(clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr));

	kernel = clCreateKernel(program, ker, &error);
	CheckError(error);

	queue = clCreateCommandQueue(context, deviceIds[0],
		0, &error);
	CheckError(error);


	// Prepare some test data

	pBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4) * (testDataSize),
		p, &error);
	CheckError(error);

	vBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float4) * (testDataSize),
		v, &error);
	CheckError(error);

	for (int i = 0; i < 500; ++i)
	{
		doCalc(25);
		readData(i);
	}

	clReleaseCommandQueue(queue);

	clReleaseMemObject(pBuffer);
	clReleaseMemObject(vBuffer);

	clReleaseKernel(kernel);

	clReleaseContext(context);

	return 0;
}




