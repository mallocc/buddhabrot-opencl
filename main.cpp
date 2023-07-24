// OpenCL Test 4.cpp : Defines the entry point for the console application.
//

#include "Animator.h"

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

using TimerMS = Timer<std::chrono::milliseconds>;

int main(int argc, char* argv[])
{
	srand(time(NULL));

	LOG("Program Entry");

	TimerMS timer;
	timer.start();

	Animator ani;

	Animator::Stage stage;
	stage.steps = 30*10;
	stage.bezier = true;
	ani.stages.push_back(stage);

	stage.zScalerA = 0.5;
	stage.zScalerC = 0.5;
	ani.stages.push_back(stage);

	//stage.v0 = { -1.486, -0.174 };
	//stage.v1 = { -1.132, 0.18 };
	ani.init();

	if (!ani.initCL())
		return 1;

	if (!ani.run())
		return 1;

	timer.stop();

	LOG(std::format("Total Time: {}s", timer.getAverageTime() / 1000.0f));

	return 0;
}




