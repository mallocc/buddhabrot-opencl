#include "BuddhabrotRenderer.h"

#include "randf.h"
#include "Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const double springConstant = 0.9; // Adjust this value to control springiness
const double dampingFactor = 0.1; // Adjust this value to control damping

void cubicHermiteSpline(const  BuddhabrotRenderer::Stage& startPoint, const  BuddhabrotRenderer::Stage& endPoint, double t, BuddhabrotRenderer::Stage& resultStage)
{
	// Calculate the t^2 and t^3 terms
	double t2 = t * t;
	double t3 = t2 * t;

	// Cubic Hermite spline interpolation formula for each parameter
	resultStage.v0.re = (2 * t3 - 3 * t2 + 1) * startPoint.v0.re + (t3 - 2 * t2 + t) * startPoint.v0.re + (-2 * t3 + 3 * t2) * endPoint.v0.re + (t3 - t2) * endPoint.v0.re;
	resultStage.v0.im = (2 * t3 - 3 * t2 + 1) * startPoint.v0.im + (t3 - 2 * t2 + t) * startPoint.v0.im + (-2 * t3 + 3 * t2) * endPoint.v0.im + (t3 - t2) * endPoint.v0.im;
	resultStage.v1.re = (2 * t3 - 3 * t2 + 1) * startPoint.v1.re + (t3 - 2 * t2 + t) * startPoint.v1.re + (-2 * t3 + 3 * t2) * endPoint.v1.re + (t3 - t2) * endPoint.v1.re;
	resultStage.v1.im = (2 * t3 - 3 * t2 + 1) * startPoint.v1.im + (t3 - 2 * t2 + t) * startPoint.v1.im + (-2 * t3 + 3 * t2) * endPoint.v1.im + (t3 - t2) * endPoint.v1.im;
}

void BuddhabrotRenderer::Stage::lerpTo(double t, const Stage& next)
{
	// Calculate the spring effect
	double springEffect = (1.0 - exp(-springConstant * t)) / (1.0 - exp(-springConstant));

	// Calculate the damping effect
	double dampingEffect = exp(-dampingFactor * t);

	// Combine the spring and damping effects to get the overall interpolation factor
	double interpolationFactor = springEffect * dampingEffect;

	t = interpolationFactor;

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
	//v0.re = lerp(t, v0.re, next.v0.re);
	//v0.im = lerp(t, v0.im, next.v0.im);
	//v1.re = lerp(t, v1.re, next.v1.re);
	//v1.im = lerp(t, v1.im, next.v1.im);
	zt.re = lerp(t, zt.re, next.zt.re);
	zt.im = lerp(t, zt.im, next.zt.im);
	ct.re = lerp(t, ct.re, next.ct.re);
	ct.im = lerp(t, ct.im, next.ct.im);
	j.re = lerp(t, j.re, next.j.re);
	j.im = lerp(t, j.im, next.j.im);

	iterations = lerp(t, iterations, next.iterations);
	iterationsMin = lerp(t, iterationsMin, next.iterationsMin);
	iterationsR = lerp(t, iterationsR, next.iterationsR);
	iterationsG = lerp(t, iterationsG, next.iterationsG);
	iterationsB = lerp(t, iterationsB, next.iterationsB);

	samples = size_t(lerp(t, double(samples), double(next.samples)));
}

size_t BuddhabrotRenderer::findMaxGlobalSize(size_t globalSize, size_t maxWorkItemSize) {
	size_t divisor = maxWorkItemSize;
	while (divisor > 0) {
		if (globalSize % divisor == 0) {
			return globalSize;
		}
		globalSize--;
	}
	// If no divisor is found, return the original globalSize.
	return globalSize;
}

void BuddhabrotRenderer::init()
{
	cl_width.data = width;
	cl_height.data = height;
	cl_initialSamplesSize.data = initialSamplesSize;
	cl_generateOnlyInRegion.data = generateOnlyInRegion;
}

void BuddhabrotRenderer::clearLastLines(int n) {
	for (int i = 0; i < n; i++) {
		std::cout << "\033[2K";  // Clear the current line
		std::cout << "\033[A";   // Move the cursor up
	}
}

int BuddhabrotRenderer::countLinesInStringStream(std::stringstream& ss) {
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

std::string BuddhabrotRenderer::drawProgressBar(double progress, bool stalled) {
	int barWidth = 30;
	int pos = static_cast<int>(barWidth * progress);

	std::stringstream ss;
	ss << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos)
			ss << "=";
		else if (i == pos)
			ss << (stalled ? ":" : ">");
		else
			ss << " ";
	}
	ss << "] " << static_cast<int>(std::round(progress * 100.0f)) << "%";
	return ss.str();
}

void BuddhabrotRenderer::print(const std::string& str)
{
	if (silent)
		return;

	std::string totalProgress = drawProgressBar(1 - stepsLeft / (double)totalSteps, false);
	std::string frameProgress = drawProgressBar(currentSample / (double)(globalSize), false);
	std::stringstream ss;
	ss
		<< "\n          Average time: " << timer.getAverageTimeInSecs() << "s"
		<< "\n        Est. time left: " << TimerMS::secondsToHHMMSS(stepsLeft * timer.getAverageTimeInSecs())
		<< "\n  Est. frame time left: " << TimerMS::secondsToHHMMSS(((globalSize - currentSample) / (double)substepSize) * subTimer.getAverageTimeInSecs())
		<< (stages.size() > 1 ? std::format("\n            Processing: STAGE {} / {},\tSTEP {} / {}", currentStage + 1, stages.size() - 1, currentStep + 1, stages[currentStage].steps) : "")
		<< "\n        Current Sample: " << currentSample << (throttling ? " THROTTLING" : "")
		<< "\n             Currently: " << (str.empty() ? lastMessage : str)
		<< "\n        Frame Progress:" << frameProgress
		<< "\n        Total Progress:" << (stages.size() == 1 ? frameProgress : totalProgress);
	//<< "\n        Total Progress:" << totalProgress;
	clearLastLines(countLinesInStringStream(ss));
	std::cout << ss.str() << std::endl;
	if (!str.empty())
		lastMessage = str;
}

double BuddhabrotRenderer::smoothstep(double x, double minVal, double maxVal)
{
	// Ensure x is within the range [minVal, maxVal]
	x = std::clamp((x - minVal) / (maxVal - minVal), 0.0, 1.0);

	// Apply the smoothstep interpolation formula
	return x * x * (3 - 2 * x);
}

double BuddhabrotRenderer::smootherstep(double x, double minVal, double maxVal)
{
	return smoothstep(smoothstep(x, minVal, maxVal), minVal, maxVal);
}

bool BuddhabrotRenderer::createDirectories(const std::string& filepath) {
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

bool BuddhabrotRenderer::writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data)
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

// Apply medium filter to a 2D array
static void mediumFilter2D(int w, int h, std::vector<cl_int>& input, int filterSize, double spatialSigma = 2, double intensitySigma = 100)
{
	std::vector<int> result(input.size());

	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			double totalWeight = 0.0;
			double weightedSum = 0.0;

			// Calculate the region of interest around the current pixel
			int startY = std::max<int>(0, y - filterSize / 2);
			int endY = std::min<int>(h - 1, y + filterSize / 2);
			int startX = std::max<int>(0, x - filterSize / 2);
			int endX = std::min<int>(w - 1, x + filterSize / 2);

			for (int j = startY; j <= endY; ++j)
			{
				for (int i = startX; i <= endX; ++i)
				{
					// Calculate spatial and intensity differences
					double spatialDist = std::sqrt((i - x) * (i - x) + (j - y) * (j - y));
					double intensityDist = std::abs(input[j * w + i] - input[y * w + x]);

					// Calculate the weight based on spatial and intensity differences
					double spatialWeight = std::exp(-spatialDist / (2.0 * spatialSigma * spatialSigma));
					double intensityWeight = std::exp(-intensityDist / (2.0 * intensitySigma * intensitySigma));
					double weight = spatialWeight * intensityWeight;

					// Accumulate the weighted sum and total weight
					weightedSum += weight * input[j * w + i];
					totalWeight += weight;
				}
			}

			// Calculate the new pixel value by shifting towards the weighted sum
			int newValue = static_cast<int>(weightedSum / totalWeight);
			result[y * w + x] = newValue;
		}
	}

	input = result;
}

void BuddhabrotRenderer::readHistogramData(CLBuf<cl_int>& buffer, int componentOffset, const Stage& stage, bool applyMediumFilter)
{
	//mediumFilter2D(width, height, buffer.data, 2, 3, 25);

	double minValue = *std::min_element(buffer.data.data(), buffer.data.data() + buffer.data.size());
	double maxValue = *std::max_element(buffer.data.data(), buffer.data.data() + buffer.data.size());
	// Subtract the minimum value and divide by the range
	double range = maxValue - minValue;

	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x)
		{
			double newValue = buffer.data.data()[y * width + x];
			for (int c = 0; c < components; ++c)
				if (componentOffset == -1 || componentOffset == c)
					pixelData[size_t(y * width + x) * components + c] =
					static_cast<cl_int>(
						pow(
							smoothstep(
								(newValue - minValue) / range,
								0.0f, 1.0f),
							1.0 / stage.gamma)
						* UCHAR_MAX /** (applyMediumFilter ? 0.5f : 1.0f)*/);
		}

}

Complex BuddhabrotRenderer::mutate(Complex& c, Complex& size, const Complex& minc, const Complex& maxc, double threshold)
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
		return c;
	}

}

double BuddhabrotRenderer::contrib(int len, std::vector<Complex>& orbit, const Complex& minc, const Complex& maxc)
{
	double contrib = 0;
	int inside = 0, i;

	for (i = 0; i < len; i++)
		if (orbit[i].re >= minc.re && orbit[i].re < maxc.re && orbit[i].im >= minc.im && orbit[i].im < maxc.im)
			contrib++;

	return contrib / double(len);
}

Complex BuddhabrotRenderer::mandelise(const Complex& z, const Complex& c, const glm::mat2& zm1, const glm::mat2& zm2, const glm::mat2& zm3) {
	glm::vec2 z1(z.re, z.im);
	float xx = z1.x * z1.x;
	float yy = z1.y * z1.y;
	glm::vec2 z2(xx - yy, z1.x * z1.y * 2.0f);
	glm::vec2 z3(xx * z1.x - 3.0f * z1.x * yy, 3.0f * xx * z1.y - yy * z1.y);
	glm::vec2 n = zm3 * z3 + zm2 * z2 + zm1 * z1 + glm::vec2(c.re, c.im);
	return { n.x, n.y };
}

void BuddhabrotRenderer::generateInitialSamples(int targetAmount, int iter, const Stage& stage)
{
	// find the size of the viewable complex plane
	Complex size = stage.v1 - stage.v0;

	// the center of the viewable complex plane
	Complex center = size / 2.0 + stage.v0;

	glm::mat2 zm1 = buildMatrix(stage.zAngleA, stage.zScalerA, stage.zYScaleA);
	glm::mat2 zm2 = buildMatrix(stage.zAngleB, stage.zScalerB, stage.zYScaleB);
	glm::mat2 zm3 = buildMatrix(stage.zAngleC, stage.zScalerC, stage.zYScaleC);

	auto evalOrbit = [&](std::vector<Complex>& orbit, int& i, Complex& c) {
		Complex z(c);
		for (i = 0; i < iter && z.mod2() < radius; ++i)
		{
			orbit[i] = z = mandelise(z, c, zm1, zm2, zm3);
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
					if (samples.size() >= targetAmount)
						break;
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
				int con = static_cast<int>(contrib(orbitLen, tempOrbit, stage.v0, stage.v1));
				if (sampleContributions[e] < con)
				{
					samples[e] = m;
					sampleContributions[e] = con;
				}
			}
		}
	}
}

void BuddhabrotRenderer::centeredRegionAfterAspect(Complex& v0, Complex& v1)
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

std::vector<float> BuddhabrotRenderer::mat4ToVector(const glm::mat4& matrix) {
	return std::vector<float>(glm::value_ptr(matrix), glm::value_ptr(matrix) + 16);
}

glm::mat2 BuddhabrotRenderer::buildMatrix(double theta, double scaler, double yscale) {
	double radians = theta / 180.0 * glm::pi<double>();
	return glm::mat2(glm::cos(radians) * scaler, glm::sin(radians) * scaler * yscale,
		-glm::sin(radians) * scaler, glm::cos(radians) * scaler * yscale);
}

glm::mat4 BuddhabrotRenderer::create4DRotationMatrix(double alpha, double beta, double gamma, double delta) {
	glm::mat4 rotationMatrix(1.0);
	rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(alpha), glm::vec3(1.0f, 0.0f, 0.0f));
	rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(beta), glm::vec3(0.0f, 1.0f, 0.0f));
	rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(gamma), glm::vec3(0.0f, 0.0f, 1.0f));
	rotationMatrix = glm::rotate(rotationMatrix, static_cast<float>(delta), glm::vec3(1.0f, 1.0f, 1.0f));

	return rotationMatrix;
}

bool BuddhabrotRenderer::process(int iter, const Stage& stage)
{
	Complex v0 = stage.v0, v1 = stage.v1;
	centeredRegionAfterAspect(v0, v1);
	cl_v0.data = { (float)v0.re, (float)v0.im };
	cl_v1.data = { (float)v1.re, (float)v1.im };

	Complex size = v1 - v0;
	cl_size.data = { (float)size.re,(float)size.im };

	double zoom = !generateOnlyInRegion && scale ? 16.0f / size.re : 1.0f;
	double zoomIter = scale ? 2.0f / size.re : 1.0f;

	cl_iterations.data = std::min<cl_int>(iter * zoomIter, 4096);
	cl_iterationsMin.data = static_cast<cl_int>(stage.iterationsMin);

	Complex center = size / 2.0f + v0;
	cl_center.data = { (float)center.re, (float)center.im };

	cl_windowSize.data = { width / (float)size.re, height / (float)size.im };

	glm::mat2 zm1 = buildMatrix(stage.zAngleA, stage.zScalerA, stage.zYScaleA);
	cl_zm1.data = { zm1[0].x , zm1[0].y ,zm1[1].x ,zm1[1].y };

	glm::mat2 zm2 = buildMatrix(stage.zAngleB, stage.zScalerB, stage.zYScaleB);
	cl_zm2.data = { zm2[0].x , zm2[0].y ,zm2[1].x ,zm2[1].y };

	glm::mat2 zm3 = buildMatrix(stage.zAngleC, stage.zScalerC, stage.zYScaleC);
	cl_zm3.data = { zm3[0].x , zm3[0].y ,zm3[1].x ,zm3[1].y };

	// Create 4D rotation matrices for alpha, beta, theta, and phi
	glm::mat4 rotation_alpha = create4DRotationMatrix(stage.alpha, 0.0, 0.0, 0.0);
	glm::mat4 rotation_beta = create4DRotationMatrix(0.0, stage.beta, 0.0, 0.0);
	glm::mat4 rotation_theta = create4DRotationMatrix(0.0, 0.0, stage.theta, 0.0);
	glm::mat4 rotation_phi = create4DRotationMatrix(0.0, 0.0, 0.0, stage.phi);
	// Combine the rotations in the specified order (z * x1 * x2)
	glm::mat4 rotationMatrix = rotation_phi * rotation_theta * rotation_alpha * rotation_beta;

	// Copy the data from rotationMatrixData to the cl_rotationMatrix buffer
	if (!cl_rotationMatrix.write(clm, mat4ToVector(rotationMatrix)))
		return false;

	if (initialSamplesSize > 0)
		if (!cl_initialSamples.write(clm))
			return false;

	if (!cl_histogram.fill(clm, 0))
		return false;

	subTimer.reset();

	auto sleepTime = (int)(1000 * throttleFactor);
	auto stms = std::chrono::milliseconds(sleepTime);

	auto oldSize = stage.samples * zoom;
	auto newSize = oldSize;
	if (!clm.maxWorkItemSize.empty())
	{
		newSize = findMaxGlobalSize(oldSize, clm.maxWorkItemSize[0]);
		//print(std::format("Modifing samples size for optimisation: {} -> {} samples", oldSize, newSize));
	}
	globalSize = newSize;

	substepSize = std::min<size_t>(newSize, clm.maxWorkItemSize[0] * 100000);

	for (currentSample = 0; currentSample < newSize; currentSample += substepSize)
	{
		subTimer.start();

		throttling = false;
		print("");

		cl_seed.data = static_cast<cl_uint>(randf() * UINT_MAX);

		if (!cl_histogram.write(clm))
			return false;
		if (initialSamplesSize > 0)
			if (!cl_initialSamples.write(clm))
				return false;

		if (!clm.setKernelArgs({
				&cl_initialSamples,
				&cl_initialSamplesSize,
				&cl_generateOnlyInRegion,
				&cl_size,
				&cl_histogram,
				&cl_width,
				&cl_height,
				&cl_iterations,
				&cl_iterationsMin,
				&cl_v0,
				&cl_v1,
				&cl_center,
				&cl_windowSize,
				&cl_seed,
				&cl_zm1,
				&cl_zm2,
				&cl_zm3,
				&cl_rotationMatrix
			}))
			return false;

		if (!clm.execute(substepSize))
			return false;

		if (!cl_histogram.read(clm))
			return false;
		if (initialSamplesSize > 0)
			if (!cl_initialSamples.read(clm))
				return false;

		if (sleepTime > 250)
		{
			throttling = true;
			print("");
		}

		std::this_thread::sleep_for(stms);

		subTimer.stop();
	}

	return true;
}

// Processes a single frame with the provided properties

bool BuddhabrotRenderer::processFrame(const Stage& stage, const int step)
{
	bool componentOverride = false;

	if (initialSamplesSize > 0 && samples.empty())
	{
		print("Generating initial samples...");

		for (int i = 0; i < samples.size(); ++i)
		{
			samples[i] = { cl_initialSamples.data[i].x , cl_initialSamples.data[i].y };
			sampleContributions[i] = cl_initialSamples.data[i].z;
		}

		generateInitialSamples(initialSamplesSize,
			int(std::max<double>(stage.iterationsR, std::max<double>(stage.iterationsG, std::max<double>(stage.iterationsR, stage.iterations)))),
			stage);

		for (int i = 0; i < cl_initialSamples.data.size(); ++i)
			cl_initialSamples.data[i] = (cl_float3({ (float)samples[i].re,
				(float)samples[i].im,
				(float)sampleContributions[i] }));
	}

	components = (stage.iterationsR > 0
		|| stage.iterationsG > 0
		|| stage.iterationsB > 0) ? 3 : 1;
	pixelData = std::vector<uint8_t>(width * height * components, 0);

	if (stage.iterationsR > 0)
	{
		print("Processing red channel... ");

		if (!process(static_cast<int>(stage.iterationsR), stage))
			return false;

		if (hybrid)
		{
			generateOnlyInRegion = !generateOnlyInRegion;

			auto histogramA = cl_histogram.data;

			if (!process(static_cast<int>(stage.iterationsR), stage))
				return false;

			for (int i = 0; i < cl_histogram.data.size(); ++i)
				cl_histogram.data[i] += histogramA[i];

			generateOnlyInRegion = !generateOnlyInRegion;
		}

		readHistogramData(cl_histogram, 0, stage, hybrid && !generateOnlyInRegion);

		clearHistogram();
		componentOverride = true;
	}
	if (stage.iterationsG > 0)
	{
		print("Processing green channel... ");

		if (!process(static_cast<int>(stage.iterationsG), stage))
			return false;

		if (hybrid)
		{
			generateOnlyInRegion = !generateOnlyInRegion;

			auto histogramA = cl_histogram.data;

			if (!process(static_cast<int>(stage.iterationsG), stage))
				return false;

			for (int i = 0; i < cl_histogram.data.size(); ++i)
				cl_histogram.data[i] += histogramA[i];

			generateOnlyInRegion = !generateOnlyInRegion;
		}

		readHistogramData(cl_histogram, 1, stage, hybrid && !generateOnlyInRegion);

		clearHistogram();
		componentOverride = true;
	}
	if (stage.iterationsB > 0)
	{
		print("Processing blue channel... ");

		if (!process(static_cast<int>(stage.iterationsB), stage))
			return false;

		if (hybrid)
		{
			generateOnlyInRegion = !generateOnlyInRegion;

			auto histogramA = cl_histogram.data;

			if (!process(static_cast<int>(stage.iterationsB), stage))
				return false;

			for (int i = 0; i < cl_histogram.data.size(); ++i)
				cl_histogram.data[i] += histogramA[i];

			generateOnlyInRegion = !generateOnlyInRegion;
		}

		readHistogramData(cl_histogram, 2, stage, hybrid && !generateOnlyInRegion);

		clearHistogram();
		componentOverride = true;
	}

	if (!componentOverride)
	{
		print("Processing greyscale channel... ");
		if (!process(static_cast<int>(stage.iterations), stage))
			return false;

		if (hybrid)
		{
			generateOnlyInRegion = !generateOnlyInRegion;

			auto histogramA = cl_histogram.data;

			if (!process(static_cast<int>(stage.iterations), stage))
				return false;

			for (int i = 0; i < cl_histogram.data.size(); ++i)
				cl_histogram.data[i] += histogramA[i];

			generateOnlyInRegion = !generateOnlyInRegion;
		}

		readHistogramData(cl_histogram, -1, stage);
	}

	if (!writeToPNG(filename.empty() ? "" : filename + (step == -1 ? "" : std::to_string(step)),
		width, height, components, pixelData.data()))
		return false;

	return true;
}

bool BuddhabrotRenderer::run()
{
	LOG("Starting sequence...");
	std::cout << "\n\n\n\n\n\n\n\n\n";

	for (int stage = 0; stage < stages.size() - 1; ++stage)
		totalSteps += stages[stage].steps;
	stepsLeft = totalSteps;

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
					
					cubicHermiteSpline(stages[stage], stages[stage + 1], step / (double)steps, tStage);
				}


				timer.start();

				if (!processFrame(tStage, stepC + counterOffset))
					return false;

				timer.stop();

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

bool BuddhabrotRenderer::initCL()
{
	clm.kernelFile = "kernel.cl";
	clm.kernelFunc = "buddhabrot";
	if (!clm.init())
		return false;

	cl_histogram.allocateSize(width * height);
	if (!cl_histogram.load(clm))
		return false;

	cl_rotationMatrix.allocateSize(16);
	if (!cl_rotationMatrix.load(clm, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR))
		return false;

	if (initialSamplesSize > 0)
	{
		cl_initialSamples.allocateSize(cl_initialSamplesSize.data);
		if (!cl_initialSamples.load(clm))
			return false;
	}

	auto oldSize = globalSize;
	if (!clm.maxWorkItemSize.empty())
	{
		globalSize = findMaxGlobalSize(globalSize, clm.maxWorkItemSize[0]);
		LOG(std::format("Modifing samples size for optimisation: {} -> {} samples", oldSize, globalSize));
	}

	return true;
}

// Zeros all of the buddhaData array

void BuddhabrotRenderer::clearHistogram()
{
	cl_int zero = 0;
	clEnqueueFillBuffer(clm.queue, cl_histogram.buf, &zero, sizeof(cl_int), 0, cl_histogram.data.size() * sizeof(cl_int), 0, nullptr, nullptr);
}

// Zeros both buddhaData and pixelData arrays

void BuddhabrotRenderer::clearAll()
{
	clearHistogram();
	memset(pixelData.data(), 0, pixelData.size() * sizeof(uint8_t));
}
