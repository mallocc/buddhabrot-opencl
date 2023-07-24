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
#include <glm/gtc/type_ptr.hpp> // For accessing matrix data

#include "randf.h"
#include "Log.h"
#include "CLManager.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Delicious
#define PI 3.1415926

class Animator
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

		// Linear interpolation
		template <typename T>
		static double lerp(T t, T x0, T x1)
		{
			return x0 + (x1 - x0) * t;
		}

		void lerpTo(double t, const Stage& next)
		{
			alpha = lerp(t, alpha, next.alpha) / 180 * PI;
			beta = lerp(t, beta, next.beta) / 180 * PI;
			theta = lerp(t, theta, next.theta) / 180 * PI;
			phi = lerp(t, phi, next.phi) / 180 * PI;

			gamma = lerp(t, gamma, next.gamma);

			zScalerC = lerp(t, zScalerC, next.zScalerC);
			zAngleC = lerp(t, zAngleC, next.zAngleC);
			zYScaleC = lerp(t, zYScaleC, next.zYScaleC);

			zScalerB = lerp(t, zScalerB, next.zScalerB);
			zAngleB = lerp(t, zAngleB, next.zAngleB);
			zYScaleB = lerp(t, zYScaleB, next.zYScaleB);

			zScalerA = lerp(t, zScalerA, next.zScalerA);
			zAngleA = lerp(t, zAngleA, next.zAngleA);
			zYScaleA = lerp(t, zYScaleA, next.zYScaleA);

			mhRatio = lerp(t, mhRatio, next.mhRatio);
			v0.re = lerp(t, v0.re, next.v0.re);
			v0.im = lerp(t, v0.im, next.v0.im);
			v1.re = lerp(t, v1.re, next.v1.re);
			v1.im = lerp(t, v1.im, next.v1.im);
			zt.re = lerp(t, zt.re, next.zt.re);
			zt.im = lerp(t, zt.im, next.zt.im);
			ct.re = lerp(t, ct.re, next.ct.re);
			ct.im = lerp(t, ct.im, next.ct.im);
			j.re = lerp(t, j.re, next.j.re);
			j.im = lerp(t, j.im, next.j.im);
		}
	};

	std::vector<Stage> stages;
	int currentStep = 0;
	int currentStage = 0;
	int stepsLeft = 0;
	int totalSteps = 0;

	std::string lastMessage;

	static const int localNum = 1 << 14;
	static const int localSize = 1 << 8;
	static const int globalSize = localNum * localSize;

	CLManager clm;

	std::string filename = "output/test";

	const int width = 1280;
	const int height = 720;
	int components = 3;

	int iterations = 1000;
	int iterationsMin = 50;
	int iterationsR = 1000;
	int iterationsG = 333;
	int iterationsB = 100;

	double radius = 4.0f;

	int initialSamplesSize = 0;

	std::vector<uint8_t> pixelData;

	int jobs = 23;

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

	CLBuf<cl_int> cl_histogram;

	CLVar<cl_float4> cl_zm1;
	CLVar<cl_float4> cl_zm2;
	CLVar<cl_float4> cl_zm3;

	CLBuf<cl_float3> cl_initialSamples;
	CLVar<cl_int> cl_initialSamplesSize;


	void init()
	{
		components = (iterationsR > 0
			|| iterationsG > 0
			|| iterationsB > 0) ? 3 : 1;

		pixelData = std::vector<uint8_t>(width * height * components, 0);

		cl_width.data = width;
		cl_height.data = height;
		cl_iterationsMin.data = iterationsMin;
		cl_initialSamplesSize.data = initialSamplesSize;
	}

	void clearLastLines(int n) {
		for (int i = 0; i < n; i++) {
			std::cout << "\033[2K";  // Clear the current line
			std::cout << "\033[A";   // Move the cursor up
		}
	}

	int countLinesInStringStream(std::stringstream& ss) {
		std::string line;
		int lineCount = 0;

		// Save the current position in the stream
		std::streampos originalPos = ss.tellg();

		// Count the lines by reading the stream line by line
		while (std::getline(ss, line)) {
			lineCount++;
		}

		// Restore the original position in the stream
		ss.clear();
		ss.seekg(originalPos);

		return lineCount;
	}

	std::string drawProgressBar(double progress) {
		int barWidth = 30;
		int pos = static_cast<int>(barWidth * progress);

		std::stringstream ss;
		ss << "[";
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos)
				ss << "=";
			else if (i == pos)
				ss << ">";
			else
				ss << " ";
		}
		ss << "] " << static_cast<int>(std::round(progress * 100.0f)) << "%";
		return ss.str();
	}

	void print(const std::string& str)
	{
		std::string totalProgress = drawProgressBar(1 - stepsLeft / (double)totalSteps);
		std::stringstream ss;
		ss
			//<< "\n          Average time: " << timer.getAverageTime() / 1000.0f << "s"
			//<< "\n        Est. time left: " << secondsToHHMMSS(stepsLeft * timer.getAverageTime() / 1000.0f)
			//<< "\n  Est. frame time left: " << secondsToHHMMSS(((samples - currentSamples) / (double)printInterval) * sampleTimeMs / 1000.0f)
			<< (stages.size() > 1 ? std::format("\n            Processing: STAGE {} / {},\tSTEP {} / {}", currentStage + 1, stages.size() - 1, currentStep + 1, stages[currentStage].steps) : "")
			//<< "\n        Current Sample: " << currentSamples
			<< "\n             Currently: " << (str.empty() ? lastMessage : str)
			//<< "\n        Frame Progress:" << frameProgress
			//<< "\n        Total Progress:" << (stages.size() == 1 ? frameProgress : totalProgress);
			<< "\n        Total Progress:" << totalProgress;
		clearLastLines(countLinesInStringStream(ss));
		std::cout << ss.str() << std::endl;
		if (!str.empty())
			lastMessage = str;
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
		print("Writing out render to PNG image...");

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

	Complex mutate(Complex& c, Complex& size, const Complex& minc, const Complex& maxc, double threshold = 0.5)
	{
		if (randf(0, 1) < threshold)
		{
			Complex n = c;

			double zoom = 4.0f / size.re;
			double phi = randf(0, 6.28318530718);
			double r = 0.01 / zoom;
			r *= randf(0, 1);

			n.re += r * cos(phi);
			n.im += r * sin(phi);

			return n;
		}
		else
		{
			c = { randf(minc.re, maxc.re), randf(minc.im, maxc.im) };
			//c = { randf(-2., 2.), randf(-2., 2.) };
			return c;
		}

	}

	double contrib(int len, std::vector<Complex>& orbit, const Complex& minc, const Complex& maxc)
	{
		double contrib = 0;
		int inside = 0, i;

		for (i = 0; i < len; i++)
			if (orbit[i].re >= minc.re && orbit[i].re < maxc.re && orbit[i].im >= minc.im && orbit[i].im < maxc.im)
				contrib++;

		return contrib / double(len);
	}

	void generateInitialSamples(int targetAmount, int iter, const Stage& stage)
	{
		// find the size of the viewable complex plane
		Complex size = stage.v1 - stage.v0;

		// the center of the viewable complex plane
		Complex center = size / 2.0 + stage.v0;

		auto evalOrbit = [&](std::vector<Complex>& orbit, int& i, Complex& c) {
			Complex z(c);
			for (i = 0; i < iter && z.mod2() < radius; ++i)
			{
				z = z * z + c;
				orbit[i] = z;
			}
			if (z.mod2() > radius)
				return true;
			return false;
		};

		std::function<bool(std::vector<Complex>&, Complex&, double, double, double, int)> FindInitialSample =
			[&](std::vector<Complex>& orbit, Complex& c, double x, double y, double rad, int f) -> bool
		{
			if (f > 150)
				return false;

			Complex ct = c, tmp, seed;

			int m = -1, i;
			double closest = 1e20;

			for (i = 0; i < 150; i++)
			{
				tmp = { randf(-rad, rad), randf(-rad, rad) };
				tmp.re += x;
				tmp.im += y;
				int orbitLen = 0;
				if (!evalOrbit(orbit, orbitLen, tmp))
					continue;

				if (contrib(orbitLen, orbit, stage.v0, stage.v1) > 0.0f)
				{
					c = tmp;
					return true;
				}

				for (int q = 0; q < orbit.size(); q++)
				{
					double d = (orbit[q] - center).mod2();

					if (d < closest)
						m = q,
						closest = d,
						seed = tmp;
				}
			}

			return FindInitialSample(orbit, c, seed.re, seed.im, rad / 2.0f, f + 1);
		};

		if (!initialSamplesGenerated)
		{
			initialSamplesGenerated = true;

			samples = {};
			sampleContributions = {};

			for (int maxTries = 0; samples.size() < targetAmount && maxTries < 1000; ++maxTries)
			{
#ifndef _DEBUG
#pragma omp parallel num_threads(jobs)
#endif
				{
					std::vector<Complex> tempOrbit(iter);
#ifndef _DEBUG
#pragma omp for
#endif
					for (int e = 0; e < targetAmount; ++e)
					{
						Complex m;
						if (FindInitialSample(tempOrbit, m, 0, 0, radius / 2.0f, 0))
						{
							int orbitLen = 0;
							evalOrbit(tempOrbit, orbitLen, m);
#ifndef _DEBUG
#pragma omp critical
#endif
							{
								samples.push_back(m);
								sampleContributions.push_back(contrib(orbitLen, tempOrbit, stage.v0, stage.v1));
							}
						}
					}
				}
			}
		}
		else
		{
#ifndef _DEBUG
#pragma omp parallel num_threads(jobs)
#endif
			{
				std::vector<Complex> tempOrbit(iter);
#ifndef _DEBUG
#pragma omp for
#endif
				for (int e = 0; e < samples.size(); ++e)
				{
					Complex m = mutate(samples[e], size, stage.v0, stage.v1);
					int orbitLen = 0;
					evalOrbit(tempOrbit, orbitLen, m);
					int con = contrib(orbitLen, tempOrbit, stage.v0, stage.v1);
					if (sampleContributions[e] < con)
					{
						samples[e] = m;
						sampleContributions[e] = con;
					}
				}
			}
		}
	}

	void centeredRegionAfterAspect(Complex& v0, Complex& v1)
	{
		// Calculate the aspect ratio of the viewable region
		double aspectRatio = static_cast<double>(width) / height;

		// Calculate the current width and height of the viewable region
		double currentWidth = v1.re - v0.re;
		double currentHeight = v1.im - v0.im;

		// Calculate the new width and height to maintain the aspect ratio
		double newWidth, newHeight;
		if (currentWidth / currentHeight > aspectRatio)
		{
			newWidth = currentHeight * aspectRatio;
			newHeight = currentHeight;
		}
		else
		{
			newWidth = currentWidth;
			newHeight = currentWidth / aspectRatio;
		}

		// Calculate the center of the original viewable region
		double centerX = (v0.re + v1.re) / 2.0;
		double centerY = (v0.im + v1.im) / 2.0;

		// Calculate the new v0 and v1 boundaries to center the viewable region
		v0 = { static_cast<float>(centerX - newWidth / 2.0), static_cast<float>(centerY - newHeight / 2.0) };
		v1 = { static_cast<float>(centerX + newWidth / 2.0), static_cast<float>(centerY + newHeight / 2.0) };
	}

	std::vector<float> mat4ToVector(const glm::mat4& matrix) {
		return std::vector<float>(glm::value_ptr(matrix), glm::value_ptr(matrix) + 16);
	}

	glm::mat2 buildMatrix(double theta, double scaler, double yscale) {
		double radians = theta / 180.0 * glm::pi<double>();
		return glm::mat2(glm::cos(radians) * scaler, glm::sin(radians) * scaler * yscale,
			-glm::sin(radians) * scaler, glm::cos(radians) * scaler * yscale);
	}

	glm::mat4 create4DRotationMatrix(double alpha, double beta, double gamma, double delta) {
		glm::mat4 rotationMatrix(1.0);
		rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(alpha), glm::vec3(1.0f, 0.0f, 0.0f));
		rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(beta), glm::vec3(0.0f, 1.0f, 0.0f));
		rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(gamma), glm::vec3(0.0f, 0.0f, 1.0f));
		rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(delta), glm::vec3(1.0f, 1.0f, 1.0f));

		return rotationMatrix;
	}

	bool process(int iter, const Stage& stage)
	{
		cl_iterations.data = iter;

		Complex v0 = stage.v0, v1 = stage.v1;
		centeredRegionAfterAspect(v0, v1);
		cl_v0.data = { (float)v0.re, (float)v0.im };
		cl_v1.data = { (float)v1.re, (float)v1.im };

		cl_seed.data = randf() * UINT_MAX;

		cl_int zero = 0;
		clEnqueueFillBuffer(clm.queue, cl_histogram.buf, &zero, sizeof(cl_int), 0, cl_histogram.data.size() * sizeof(cl_int), 0, nullptr, nullptr);

		glm::mat2 zm1 = buildMatrix(stage.zAngleA, stage.zScalerA, stage.zYScaleA);
		glm::mat2 zm2 = buildMatrix(stage.zAngleB, stage.zScalerB, stage.zYScaleB);
		glm::mat2 zm3 = buildMatrix(stage.zAngleC, stage.zScalerC, stage.zYScaleC);

		cl_zm1.data.x = zm1[0].x;
		cl_zm1.data.y = zm1[0].y;
		cl_zm1.data.z = zm1[1].x;
		cl_zm1.data.w = zm1[1].y;

		cl_zm2.data.x = zm2[0].x;
		cl_zm2.data.y = zm2[0].y;
		cl_zm2.data.z = zm2[1].x;
		cl_zm2.data.w = zm2[1].y;

		cl_zm3.data.x = zm3[0].x;
		cl_zm3.data.y = zm3[0].y;
		cl_zm3.data.z = zm3[1].x;
		cl_zm3.data.w = zm3[1].y;

		if (initialSamplesSize > 0)
			if (!clm.setKernelArgs({
				&cl_initialSamples,
				&cl_initialSamplesSize,
				&cl_histogram,
				&cl_width,
				&cl_height,
				&cl_iterations,
				&cl_iterationsMin,
				&cl_v0,
				&cl_v1,
				&cl_seed
				}))
				return false;

		if (!clm.setKernelArgs({
			&cl_histogram,
			&cl_width,
			&cl_height,
			&cl_iterations,
			&cl_iterationsMin,
			&cl_v0,
			&cl_v1,
			&cl_seed,
			&cl_zm1,
			&cl_zm2,
			&cl_zm3
			}))
			return false;

		if (!clm.execute(globalSize, localSize))
			return false;

		if (!cl_histogram.read(clm))
			return false;

		if (initialSamplesSize > 0)
			if (!cl_initialSamples.read(clm))
				return false;

		return true;
	}

	// Processes a single frame with the provided properties
	bool processFrame(const Stage& stage, const int step)
	{
		bool componentOverride = false;

		if (initialSamplesSize > 0)
		{
			print("Generating initial samples...");

			for (int i = 0; i < samples.size(); ++i)
			{
				samples[i] = { cl_initialSamples.data[i].x , cl_initialSamples.data[i].y };
				sampleContributions[i] = cl_initialSamples.data[i].z;
			}

			generateInitialSamples(initialSamplesSize,
				std::max<int>(iterationsR, std::max<int>(iterationsG, std::max<int>(iterationsR, iterations))),
				stage);

			for (int i = 0; i < samples.size(); ++i)
				cl_initialSamples.data[i] = (cl_float3({ (float)samples[i].re,
					(float)samples[i].im,
					(float)sampleContributions[i] }));
		}

		if (iterationsR > 0)
		{
			print("Processing red channel... ");

			if (!process(iterationsR, stage))
				return false;

			readHistogramData(cl_histogram, 0);

			clearHistogram();
			componentOverride = true;
		}
		if (iterationsG > 0)
		{
			print("Processing green channel... ");

			if (!process(iterationsG, stage))
				return false;

			readHistogramData(cl_histogram, 1);

			clearHistogram();
			componentOverride = true;
		}
		if (iterationsB > 0)
		{
			print("Processing blue channel... ");

			if (!process(iterationsB, stage))
				return false;

			readHistogramData(cl_histogram, 2);

			clearHistogram();
			componentOverride = true;
		}

		if (!componentOverride)
		{
			print("Processing greyscale channel... ");
			if (!process(iterations, stage))
				return false;

			readHistogramData(cl_histogram, -1);
		}

		if (!writeToPNG(filename.empty() ? "" : filename + (step == -1 ? "" : std::to_string(step)),
			width, height, components, pixelData.data()))
			return false;

		return true;
	}

	bool run()
	{
		LOG("Starting sequence...");
		std::cout << "\n\n\n\n\n\n\n";

		for (int stage = 0; stage < stages.size() - 1; ++stage)
			totalSteps += stages[stage].steps;

		if (stages.size() > 1)
		{
			int stepC = 0;
			for (int stage = 0; stage < stages.size() - 1; ++stage)
				for (int step = 0, steps = stages[stage].steps; step < steps; ++step, ++stepC)
				{
					currentStage = stage;
					currentStep = step;

					clearAll();

					Stage tStage = stages[stage];

					if (stage + 1 < stages.size())
					{
						double b = tStage.bezier ? smootherstep(step / (double)steps, 0., 1.) : (step / (double)steps);
						tStage.lerpTo(b, stages[stage + 1]);
					}

					if (!processFrame(tStage, stepC))
						return false;

					stepsLeft = (totalSteps - stepC);
				}
		}
		else if (!stages.empty())
		{
			clearAll();
			if (!processFrame(stages[0], -1))
				return false;
		}

		return true;
	}

	bool initCL()
	{
		clm.kernelFile = "kernel.cl";
		clm.kernelFunc = "buddhabrot";
		if (!clm.init())
			return false;

		cl_histogram.allocateSize(width * height);
		if (!cl_histogram.load(clm))
			return false;

		if (initialSamplesSize > 0)
		{
			cl_initialSamples.allocateSize(cl_initialSamplesSize.data);
			for (int i = 0; i < samples.size(); ++i)
				cl_initialSamples.data[i] = { 0 };
			if (!cl_initialSamples.load(clm))
				return false;
		}

		return true;
	}

	// Zeros all of the buddhaData array
	void clearHistogram()
	{
		cl_int zero = 0;
		clEnqueueFillBuffer(clm.queue, cl_histogram.buf, &zero, sizeof(cl_int), 0, cl_histogram.data.size() * sizeof(cl_int), 0, nullptr, nullptr);
	}

	// Zeros both buddhaData and pixelData arrays
	void clearAll()
	{
		clearHistogram();
		memset(pixelData.data(), 0, pixelData.size() * sizeof(uint8_t));
	}

};