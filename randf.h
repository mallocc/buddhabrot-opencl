#pragma once

#include <random>

/**
 * Mersenne Twister random number generator.
 */
double randf(const double minimum = 0.0, const double maximum = 1.0)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	return dis(gen) * (maximum - minimum) + minimum;
}