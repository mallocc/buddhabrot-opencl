#pragma once
#include "Complex.h"
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <format>
#include <functional>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "CLManager.h"
#include "Timer.h"

// Delicious
#define PI 3.1415926

class BuddhabrotRenderer
{
public:
	struct Stage
	{
		Complex v0 = Complex(-2, -2);
		Complex v1 = Complex(2, 2);
		Complex zt = Complex();
		Complex ct = Complex();
		Complex j = Complex();
		bool bezier = false;
		int steps = 1;
		double mhRatio = 0.5;
		double gamma = 2;

		double alpha = 0;
		double beta = 0;
		double theta = 0;
		double phi = 0;

		double zScalerC = 0;
		double zAngleC = 0;
		double zYScaleC = 1;

		double zScalerB = 1;
		double zAngleB = 0;
		double zYScaleB = 1;

		double zScalerA = 0;
		double zAngleA = 0;
		double zYScaleA = 1;

		double iterations = 0;
		double iterationsMin = 0;
		double iterationsR = 0;
		double iterationsG = 0;
		double iterationsB = 0;

		size_t samples = 0;

		// Linear interpolation
		template <typename T>
		static double lerp(T t, T x0, T x1)
		{
			return x0 + (x1 - x0) * t;
		}

		void lerpTo(double t, const Stage& next);
	};

	std::vector<Stage> stages;
	int currentStep = 0;
	int currentStage = 0;
	int stepsLeft = 0;
	int totalSteps = 0;

	std::string lastMessage;

	size_t globalSize = 0;
	size_t currentSample = 0;

	size_t substepSize = 0;

	double throttleFactor = 0;
	bool throttling = false;

	CLManager clm;

	std::string filename = "output/test";

	bool silent = false;

	int width = 0;
	int height = 0;
	int components = 0;

	int iterations = 0;
	int iterationsMin = 0;
	int iterationsR = 0;
	int iterationsG = 0;
	int iterationsB = 0;

	int counterOffset = 0;

	bool generateOnlyInRegion = false;

	double radius = 4.0f;

	int initialSamplesSize = 0;

	bool hybrid = false;

	bool scale = false;

	std::vector<uint8_t> pixelData;

	int jobs = 1;

	std::vector<Complex> samples;
	std::vector<double> sampleContributions;
	bool initialSamplesGenerated = false;

	CLVar<cl_int> cl_width;
	CLVar<cl_int> cl_height;
	CLVar<cl_int> cl_iterations;
	CLVar<cl_int> cl_iterationsMin;
	CLVar<cl_float2> cl_v0;
	CLVar<cl_float2> cl_v1;
	CLVar<cl_uint> cl_seed;
	CLVar<cl_float2> cl_size;
	CLVar<cl_float2> cl_center;
	CLVar<cl_float2> cl_windowSize;
	CLVar<cl_bool> cl_generateOnlyInRegion;

	CLBuf<cl_int> cl_histogram;

	CLBuf<cl_float> cl_rotationMatrix;

	CLVar<cl_float4> cl_zm1;
	CLVar<cl_float4> cl_zm2;
	CLVar<cl_float4> cl_zm3;

	CLBuf<cl_float3> cl_initialSamples;
	CLVar<cl_int> cl_initialSamplesSize;

	TimerMS timer;
	TimerMS subTimer;

	// Find the maximum global work size that is compatible with the device's maximum work item size
	size_t findMaxGlobalSize(size_t globalSize, size_t maxWorkItemSize);

	// Initialize the application and set up the required resources
	void init();

	// Clear the last 'n' lines from the console output
	void clearLastLines(int n);

	// Count the number of lines in a given stringstream
	int countLinesInStringStream(std::stringstream& ss);

	// Draw a progress bar with a given progress value and optional stall indication
	std::string drawProgressBar(double progress, bool stalled);

	// Print the given string to the console
	void print(const std::string& str);

	// Smoothstep function that interpolates between two values based on a given input value
	static double smoothstep(double x, double minVal, double maxVal);

	// Smootherstep function that provides a smoother interpolation between two values
	static double smootherstep(double x, double minVal, double maxVal);

	// Create directories along the specified filepath if they don't exist
	bool createDirectories(const std::string& filepath);

	// Write pixel data to a PNG file using stb_image_write.h
	bool writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data);

	// Read histogram data from a CLBuf object and update pixelData accordingly
	void readHistogramData(CLBuf<cl_int>& buffer, int componentOffset, const Stage& stage, bool applyMediumFilter = false);

	// Mutate a complex number using a given size, min, max, and threshold
	Complex mutate(Complex& c, Complex& size, const Complex& minc, const Complex& maxc, double threshold = 0.5);

	// Calculate the contribution of the orbit to the fractal image
	double contrib(int len, std::vector<Complex>& orbit, const Complex& minc, const Complex& maxc);

	// Perform Mandelbrot iteration on the given complex number 'z' using a given 'c' and matrices
	inline static Complex mandelise(const Complex& z, const Complex& c, const glm::mat2& zm1, const glm::mat2& zm2, const glm::mat2& zm3);

	// Generate initial samples for the fractal generation process
	void generateInitialSamples(int targetAmount, int iter, const Stage& stage);

	// Adjust the region to be centered after considering the aspect ratio
	void centeredRegionAfterAspect(Complex& v0, Complex& v1);

	// Convert a glm::mat4 matrix to a vector of floats
	std::vector<float> mat4ToVector(const glm::mat4& matrix);

	// Build a 2D transformation matrix with the given theta, scaler, and yscale values
	glm::mat2 buildMatrix(double theta, double scaler, double yscale);

	// Create a 4D rotation matrix using alpha, beta, gamma, and delta angles
	glm::mat4 create4DRotationMatrix(double alpha, double beta, double gamma, double delta);

	// Process the fractal generation for a given number of iterations and a stage
	bool process(int iter, const Stage& stage);

	// Process a single frame with the provided properties for a given step
	bool processFrame(const Stage& stage, const int step);

	// Run the fractal generation process
	bool run();

	// Initialize the OpenCL resources
	bool initCL();

	// Zero all elements of the histogram array
	void clearHistogram();

	// Zero all elements of both the histogram and pixelData arrays
	void clearAll();

};