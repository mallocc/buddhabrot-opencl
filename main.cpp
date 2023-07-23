// OpenCL Test 4.cpp : Defines the entry point for the console application.
//

#include "CLManager.h"

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

static const int localNum = 1 << 17;
static const int localSize = 1 << 9;
static const int globalSize = localNum * localSize;

CLManager clm;

std::string filename = "output/test";

const int width = 4096;
const int height = 4096;
int components = 3;

int iterations = 1000;
int iterations_min = 0;
int iterations_colours[3] = { 500, 50, 10 };
cl_float2 v0 = { -2,-1.5 };
cl_float2 v1 = { 1,1.5 };

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
bool writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data)
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

	return 
		createDirectories(ss.str().c_str()) &&
		stbi_write_png(ss.str().c_str(), w, h, c, data, w * c);
}

static double smoothstep(double x, double minVal, double maxVal)
{
	// Ensure x is within the range [minVal, maxVal]
	x = std::clamp((x - minVal) / (maxVal - minVal), 0.0, 1.0);

	// Apply the smoothstep interpolation formula
	return x * x * (3 - 2 * x);
}

void readHistogramData(const CLBuf<cl_int>& buffer, int componentOffset)
{
	double minValue = *std::min_element(buffer.data.data(), buffer.data.data() + buffer.data.size());
	double maxValue = *std::max_element(buffer.data.data(), buffer.data.data() + buffer.data.size());
	// Subtract the minimum value and divide by the range
	double range = maxValue - minValue;

	double gamma = 2.5;

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			double newValue = buffer.data.data()[y * width + x];
			for (int c = 0; c < components; ++c)
				if (componentOffset == -1 || componentOffset == c)
					pixelData[(y * width + x) * components + c] = pow(smoothstep((newValue - minValue) / range, 0.0f, 1.0f), 1.0 / gamma) * UCHAR_MAX;
		}
}

int main(int argc, char* argv[])
{
	srand(time(NULL));

	LOG("Program Entry");

	clm.kernelFile = "kernel.cl";
	clm.kernelFunc = "buddhabrot";
	if (!clm.init())
		return 1;

	CLVar<cl_int> cl_initalSamplesSize(30);
	CLVar<cl_int> cl_width(width);
	CLVar<cl_int> cl_height(height);
	CLVar<cl_int> cl_iterations;
	CLVar<cl_int> cl_iterationsMin(iterations_min);
	CLVar<cl_float2> cl_v0(v0);
	CLVar<cl_float2> cl_v1(v1);
	CLVar<cl_uint> cl_seed;

	CLBuf<cl_int> cl_histogram;
	cl_histogram.allocateSize(width * height);
	memset(cl_histogram.data.data(), 0, cl_histogram.data.size() * sizeof(cl_int));
	if (!cl_histogram.load(clm))
		return 1;

	LOG("Processing samples...");
	Timer<std::chrono::milliseconds> timerExecute;
	timerExecute.start();

	for (int i = 0; i < components; ++i)
	{
		cl_iterations.data = iterations_colours[i];
		LOG(std::format("iterations = {}", cl_iterations.data));

		cl_seed.data = randf() * UINT_MAX;

		memset(cl_histogram.data.data(), 0, cl_histogram.data.size() * sizeof(cl_int));

		LOG(std::format("samples = {}", globalSize));

		if (!clm.setKernelArgs({
			&cl_initalSamplesSize,
			&cl_histogram,
			&cl_width,
			&cl_height,
			&cl_iterations,
			&cl_iterationsMin,
			&cl_v0,
			&cl_v1,
			&cl_seed
			}))
			return 1;

		//cl_initalSamplesSize.setKernelArg(clm, 0);
		//cl_histogram.setKernelArg(clm, 1);
		//cl_width.setKernelArg(clm, 2);
		//cl_height.setKernelArg(clm, 3);
		//cl_iterations.setKernelArg(clm, 4);
		//cl_iterationsMin.setKernelArg(clm, 5);
		//cl_v0.setKernelArg(clm, 6);
		//cl_v1.setKernelArg(clm, 7);
		//cl_seed.setKernelArg(clm, 8);

		if (!clm.execute(globalSize, localSize))
			return 1;
		if (!cl_histogram.read(clm))
			return 1;

		readHistogramData(cl_histogram, i);
	}

	writeToPNG(filename.empty() ? "" : filename + std::to_string(0), width, height, components, pixelData);

	timerExecute.stop();

	LOG(std::format("Processing samples took: {}s", timerExecute.getAverageTimeInSecs()));

	return 0;
}




