#pragma once

#include <random>

/**
 * Mersenne Twister random number generator.
 */
float randf(const float minimum = 0.0, const float maximum = 1.0)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<float> dis(0.0, 1.0);
	return dis(gen) * (maximum - minimum) + minimum;
}