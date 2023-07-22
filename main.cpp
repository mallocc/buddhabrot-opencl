// OpenCL Test 4.cpp : Defines the entry point for the console application.
//

#include <CL\cl.h>
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
#include <filesystem>
#include <omp.h>
#include <chrono>

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
float randf(const float minimum = 0.0, const float maximum = 1.0)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<float> dis(0.0, 1.0);
	return dis(gen) * (maximum - minimum) + minimum;
}

template<class T>
class Timer {
public:
	Timer() : totalDuration(0), numSamples(0) {}

	void start() {
		startTime = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<T>(endTime - startTime).count();

		totalDuration.fetch_add(duration);
		numSamples.fetch_add(1);
	}

	double getAverageTime() const {
		if (numSamples == 0) {
			return 0.0;
		}
		return static_cast<double>(totalDuration) / numSamples;
	}

	double getRemainingTime(int totalIterations, int completedIterations) const {
		int remainingIterations = totalIterations - completedIterations;
		if (remainingIterations <= 0) {
			return 0.0;
		}
		double averageTime = getAverageTime();
		return averageTime * remainingIterations;
	}

	double getAverageTimeInSecs() const {
		return convertToSeconds(getAverageTime());
	}

	static double convertToSeconds(double duration) {
		if (std::is_same<T, std::chrono::nanoseconds>::value) {
			return duration / 1e9;
		}
		else if (std::is_same<T, std::chrono::microseconds>::value) {
			return duration / 1e6;
		}
		else if (std::is_same<T, std::chrono::milliseconds>::value) {
			return duration / 1e3;
		}
		else if (std::is_same<T, std::chrono::seconds>::value) {
			return duration;
		}
		else if (std::is_same<T, std::chrono::minutes>::value) {
			return duration * 60;
		}
		else if (std::is_same<T, std::chrono::hours>::value) {
			return duration * 3600;
		}
		else {
			return 0.0;
		}
	}


private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::atomic<long long> totalDuration;
	std::atomic<int> numSamples;
};



static const int localNum = 1 << 19;
static const int localSize = 1 << 9;
static const int globalSize = localNum * localSize;


const char* kernalFunc = "buddhabrot";
cl_command_queue queue;
cl_kernel kernel;
std::vector<cl_device_id> deviceIds;
cl_int error;
cl_context context;

inline void CheckError(cl_int error);

std::string filename = "output/test";

const int width = 128;
const int height = 128;
int components = 1;

const int histogram_size = width * height;
cl_int histogram[histogram_size];
cl_mem histogram_buf;

const int inital_samples_size = globalSize;
cl_uchar initial_samples[inital_samples_size];
cl_mem initial_samples_buf;
cl_uint seed = randf() * UINT_MAX;

int iterations = 1000;
cl_float2 v0 = { -2,-2 };
cl_float2 v1 = { 2,2 };

uint8_t* pixelData = new uint8_t[width * height * components];

bool createDirectories(const std::string& filepath) {
	std::filesystem::path path(filepath);

	// Extract the directory path
	std::filesystem::path directory = path.parent_path();

	// Create directories recursively
	try {
		std::filesystem::create_directories(directory);
	}
	catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "Error creating directories: " << e.what() << std::endl;
		return false;
	}

	return true;
}

// writes pixelData out to a PNG using stb_image_write.h
void writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data)
{
	LOG("Writing out render to PNG image...");
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
	createDirectories(ss.str().c_str());
	stbi_write_png(ss.str().c_str(), w, h, c, data, w * c);
}

static double smoothstep(double x, double minVal, double maxVal)
{
	// Ensure x is within the range [minVal, maxVal]
	x = std::clamp((x - minVal) / (maxVal - minVal), 0.0, 1.0);

	// Apply the smoothstep interpolation formula
	return x * x * (3 - 2 * x);
}

static double smootherstep(double x, double minVal, double maxVal)
{
	return smoothstep(smoothstep(x, minVal, maxVal), minVal, maxVal);
}

static uint8_t sqrtColour(double x, double y, double gamma)
{
	return pow(x / y, 1.0 / gamma) * UCHAR_MAX;
}

void readData()
{
	// Get the results back to the host
	CheckError(clEnqueueReadBuffer(queue, histogram_buf, CL_TRUE, 0,
		sizeof(cl_int) * histogram_size,
		histogram,
		0, nullptr, nullptr));

	double minValue = *std::min_element(histogram, histogram + histogram_size);
	double maxValue = *std::max_element(histogram, histogram + histogram_size);
	// Subtract the minimum value and divide by the range
	double range = maxValue - minValue;

	double gamma = 2.5;

	int componentOffset = 0;
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			double newValue = histogram[y * width + x];
			for (int c = 0; c < components; ++c)
				if (componentOffset == -1 || componentOffset == c)
					pixelData[(y * width + x) * components + c] = pow(smoothstep((newValue - minValue) / range, 0.0f, 1.0f), 1.0 / gamma) * UCHAR_MAX;
		}

	writeToPNG(filename.empty() ? "" : filename + std::to_string(0), width, height, components, pixelData);
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

inline void execute()
{
	std::cout << "inital_samples_size = " << inital_samples_size << std::endl;

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &initial_samples_buf);
	clSetKernelArg(kernel, 1, sizeof(cl_int), &inital_samples_size);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &histogram_buf);
	clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
	clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
	clSetKernelArg(kernel, 5, sizeof(cl_int), &iterations);
	clSetKernelArg(kernel, 6, sizeof(cl_float2), &v0);
	clSetKernelArg(kernel, 7, sizeof(cl_float2), &v1);
	clSetKernelArg(kernel, 8, sizeof(cl_uint), &seed);

	cl_event eve;
	const size_t globalWorkSize[] = { globalSize, 0, 0 };
	const size_t localWorkSize[] = { localSize, 0, 0 };
	CheckError(clEnqueueNDRangeKernel(queue, kernel, 1,
		nullptr,
		globalWorkSize,
		localWorkSize,
		0, nullptr, &eve));

	CheckError(clFlush(queue));
}

int main(int argc, char* argv[])
{
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

	cl_program program = CreateProgram(LoadKernel("kernel.cl"),
		context);

	std::cout << "clBuildProgram" << std::endl;
	clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr);

	// Check for build errors and get the build log
	cl_build_status buildStatus;
	clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);

	if (buildStatus != CL_BUILD_SUCCESS)
	{
		// Get the build log size
		size_t buildLogSize;
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);

		// Allocate memory for the build log
		char* buildLog = (char*)malloc(buildLogSize);

		// Get the build log
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);

		// Output the build log (error messages)
		printf("OpenCL program build log:\n%s\n", buildLog);

		// Free allocated memory
		free(buildLog);
	}

	std::cout << "clCreateKernel" << std::endl;
	kernel = clCreateKernel(program, kernalFunc, &error);

	// Check for errors during kernel creation
	if (error != CL_SUCCESS)
	{
		// Kernel creation failed, check the error code
		std::cout << "Error creating kernel: " << error << std::endl;

		// Get the size of the build log
		size_t buildLogSize;
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);

		// Allocate memory for the build log
		char* buildLog = (char*)malloc(buildLogSize);

		// Get the build log
		clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);

		// Output the build log (error messages)
		std::cout << "Build log: " << std::endl << buildLog << std::endl;

		// Free allocated memory
		free(buildLog);

	}
	else
	{
		// Get the function name using clGetKernelInfo
		size_t functionNameSize;
		error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &functionNameSize);
		if (error == CL_SUCCESS)
		{
			// Allocate memory for the function name
			char* functionName = (char*)malloc(functionNameSize);

			// Get the function name
			error = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, functionNameSize, functionName, NULL);
			if (error == CL_SUCCESS)
			{
				// Output the kernel function name
				std::cout << "Kernel Function Name: " << functionName << std::endl;
			}

			// Free allocated memory
			free(functionName);
		}

		// Additional error checking for clGetKernelInfo if needed
		if (error != CL_SUCCESS)
		{
			std::cout << "Error getting kernel info: " << error << std::endl;
		}
	}

	std::cout << "clCreateCommandQueue" << std::endl;
	queue = clCreateCommandQueue(context, deviceIds[0],
		0, &error);
	CheckError(error);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	srand(time(NULL));

	Timer<std::chrono::milliseconds> timer;
	std::cout << "Generating samples..." << std::endl;
	timer.start();
#ifndef _DEBUG
#pragma omp parallel for num_threads(23)
#endif
	for (int i = 0; i < inital_samples_size; ++i)
		initial_samples[i] = 0;// { 0, 0 };
		//initial_samples[i] = { randf(v0.x, v1.x), randf(v0.y, v1.y) };
	timer.stop();
	std::cout << "Sample generation took: " << timer.getAverageTimeInSecs() << "s" << std::endl;

	// Prepare some test data
	std::cout << "clCreateBuffer - initial_samples_buf" << std::endl;
	initial_samples_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uchar) * (inital_samples_size),
		initial_samples, &error);
	CheckError(error);

	std::cout << "clCreateBuffer - histogram_buf" << std::endl;
	histogram_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int) * (histogram_size),
		histogram, &error);
	CheckError(error);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Processing samples..." << std::endl;
	Timer<std::chrono::milliseconds> timerExecute;
	timerExecute.start();
	execute();
	readData();
	timerExecute.stop();
	std::cout << "Processing samples took: " << timerExecute.getAverageTimeInSecs() << "s" << std::endl;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::cout << "Total time: " << timer.getAverageTimeInSecs() + timerExecute.getAverageTimeInSecs() << "s" << std::endl;

	clReleaseCommandQueue(queue);

	clReleaseMemObject(initial_samples_buf);
	clReleaseMemObject(histogram_buf);

	clReleaseKernel(kernel);

	clReleaseContext(context);


	return 0;
}




