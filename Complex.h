#pragma once
#include <cmath>

/**
 * Very basic complex struc with operations.
 */
struct Complex
{
	double re, im;

	inline Complex() : re(0.0), im(0.0) {}
	inline Complex(double re, double im) : re(re), im(im) {}

	inline Complex add(const Complex& a) { return Complex(re + a.re, im + a.im); }
	inline Complex operator+(const Complex& a) { return add(a); }
	inline friend Complex operator+(double scalar, const Complex& complex) {
		return Complex(scalar + complex.re, scalar + complex.im);
	}

	inline Complex sub(const Complex& a) { return Complex(re - a.re, im - a.im); }
	inline Complex operator-(const Complex& a) { return sub(a); }
	inline friend Complex operator-(double scalar, const Complex& complex) {
		return Complex(scalar - complex.re, scalar - complex.im);
	}
	inline friend Complex operator-(const Complex& a, const Complex& b) {
		return Complex(a.re - b.re, a.im - b.im);
	}
	inline Complex mult(const Complex& a) {
		return Complex(re * a.re - im * a.im,
			re * a.im + im * a.re);
	}
	inline static Complex mult(const Complex& a, const Complex& b) {
		return Complex(b.re * a.re - b.im * a.im,
			b.re * a.im + b.im * a.re);
	}
	inline Complex mult(const double& a) {
		return Complex(re * a, im * a);
	}
	inline Complex operator*(const Complex& a) { return mult(a); }
	inline Complex operator*(const double& a) { return mult(a); }
	inline Complex operator*(double a) { return Complex(re * a, im * a); }
	inline friend Complex operator*(double scalar, const Complex& complex) {
		return Complex(scalar * complex.re, scalar * complex.im);
	}
	inline friend Complex operator*(const Complex& c1, const Complex& c2) {
		return mult(c1, c2);
	}

	inline Complex operator/(double a) { return Complex(re / a, im / a); }

	// Function to calculate the magnitude (r) of the complex number
	inline double magnitude() const {
		return std::sqrt(re * re + im * im);
	}

	// Function to calculate the argument (theta) of the complex number
	inline double argument() const {
		return std::atan2(im, re);
	}

	// Function to raise a complex number to the power n using De Moivre's theorem
	inline Complex pow(double n) const {
		double r = magnitude();
		double theta = argument();

		double newR = std::pow(r, n);
		double newTheta = n * theta;

		return Complex(newR * std::cos(newTheta), newR * std::sin(newTheta));
	}

	//static Complex abs(const Complex& a) { return { std::abs(a.re), std::abs(a.im) }; }


	inline double mod2() { return re * re + im * im; }
};
