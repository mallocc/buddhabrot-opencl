#pragma once

#include <chrono>

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

	std::string getAverageTimestamp() const {
		return secondsToHHMMSS(convertToSeconds(getAverageTime()));
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


	static std::string secondsToHHMMSS(double duration) {
		std::chrono::seconds totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<double>(duration));
		int hours = static_cast<int>(totalSeconds.count()) / 3600;
		int minutes = (totalSeconds.count() % 3600) / 60;
		int seconds = totalSeconds.count() % 60;

		std::ostringstream oss;
		oss << std::setw(2) << std::setfill('0') << hours << ":";
		oss << std::setw(2) << std::setfill('0') << minutes << ":";
		oss << std::setw(2) << std::setfill('0') << seconds;

		return oss.str();
	}

private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::atomic<long long> totalDuration;
	std::atomic<int> numSamples;
};

using TimerMS = Timer<std::chrono::milliseconds>;
