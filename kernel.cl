

#define MAX_ITERATIONS 2048
#define USE_ATOMIC_ADD 1

uint xorshift32(uint* state) {
	uint x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

// Function to generate a random float between min and max (inclusive)
float randf(uint* state, float min, float max) {
	uint random = xorshift32(state);
	float normalized = (float)random / (float)UINT_MAX;
	return min + normalized * (max - min);
}

float complex_mag2(float2 a)
{
	return a.x * a.x + a.y * a.y;
}

float2 mutate(uint* state, float2 c, float2 size, float2 minc, float2 maxc, float threshold)
{
	if (randf(state, 0.0f, 1.0f) <= threshold)
	{
		float zoom = 4.0f / size.x;
		float r = randf(state, 0.0f, 1 / zoom);

		c.x += randf(state, -r, r);
		c.y += randf(state, -r, r);

		return c;
	}
	else
	{
		c.x = randf(state, minc.x, maxc.x);
		c.y = randf(state, minc.y, maxc.y);
		return c;
	}
}

// Matrix-vector multiplication for 2x2 matrix represented as float4
float2 mat2xvec2(float4 mat, float2 vec) {
	return (float2)(dot(mat.xy, vec), dot(mat.zw, vec));
}

// Matrix multiplication for two 2x2 matrices represented as float4
float4 mat2xmat2(float4 mat1, float4 mat2) {
	float2 row1 = { dot(mat1.xy, mat2.xy), dot(mat1.zw, mat2.xy) };
	float2 row2 = { dot(mat1.xy, mat2.zw), dot(mat1.zw, mat2.zw) };
	return (float4)(row1, row2);
}

float2 mandelise(float2 z1, float2 c, float4 zm1, float4 zm2, float4 zm3) {
	float xx = z1.x * z1.x;
	float yy = z1.y * z1.y;
	float2 z2 = { xx - yy, z1.x * z1.y * 2.0f };
	float2 z3 = { xx * z1.x - 3.0f * z1.x * yy, 3.0f * xx * z1.y - yy * z1.y };
	return mat2xvec2(zm3, z3) + mat2xvec2(zm2, z2) + mat2xvec2(zm1, z1) + c;
}

__kernel void buddhabrot_basic(
	__global int* histogram, // The output histogram data
	int w, int h, // Width and height of the output histogram
	int iterations, // Maximum number of iterations to compute the Mandelbrot set
	int iterations_min, // Minimum number of iterations to consider in the histogram
	float2 v0, float2 v1, // Coordinates of the viewable complex plane
	float2 center,
	float2 windowSize,
	uint seed, // Seed value for random number generation
	float4 zm1,
	float4 zm2,
	float4 zm3,
	__global float* rm
)
{
	// this is the sample id
	int gId = get_global_id(0);

	// allocate array for storing the iterations that make up the orbit
	__private float2 orbit[MAX_ITERATIONS];

	// Initialize the random state with the seed value
	uint state = seed + gId;

	// get an initial sample
	float2 c = { randf(&state, v0.x, v1.x), randf(&state, v0.y, v1.y) };
	//float2 c = { randf(&state, -2, 2), randf(&state, -2, 2) };
	float2 z = c;

	// classic mandelbrot formula
	int iter = 0;
	for (; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
	{
		// store our position in the orbit for the histogram
		orbit[iter] = z = mandelise(z, c, zm1, zm2, zm3);
	}

	if (iter < iterations && iter >= iterations_min)
	{
		c = c - center;

		float p4d[4] = { 0.0f,0.0f,c.x,1.0f };
		float rotatedPoint[4];

		int x = 0, y = 0;
		// accumulate the orbit historgram locally
		for (int i = 0; i < iter; ++i)
		{
			float2 t = orbit[i];

			t = t - center;

			p4d[0] = t.x;
			p4d[1] = t.y;

			// Apply the rotation matrix on the 4D point
			rotatedPoint[0] = rm[0] * p4d[0] + rm[4] * p4d[1] + rm[8] * p4d[2] + rm[12] * p4d[3];
			rotatedPoint[1] = rm[1] * p4d[0] + rm[5] * p4d[1] + rm[9] * p4d[2] + rm[13] * p4d[3];
			rotatedPoint[2] = rm[2] * p4d[0] + rm[6] * p4d[1] + rm[10] * p4d[2] + rm[14] * p4d[3];
			rotatedPoint[3] = rm[3] * p4d[0] + rm[7] * p4d[1] + rm[11] * p4d[2] + rm[15] * p4d[3];

			// Convert the 4D point back to 3D
			// Update the point coordinates
			t.x = rotatedPoint[0] / rotatedPoint[3];
			t.y = rotatedPoint[1] / rotatedPoint[3];

			t = t + center;

			// project back to screen space
			x = (t.x - v0.x) * windowSize.x;
			y = (t.y - v0.y) * windowSize.y;
			// make sure it is in the viewable region
			if (x >= 0 && x < w && y >= 0 && y < h)
				histogram[y * w + x] += 1;
		}
	}
}


__kernel void buddhabrot(
	__global float3* inital_samples, // The initial samples array
	int initial_samples_size, // The size of the initial samples array
	int generateOnlyInRegion,
	float2 size, // The size of the initial samples array
	__global int* histogram, // The output histogram data
	int w, int h, // Width and height of the output histogram
	int iterations, // Maximum number of iterations to compute the Mandelbrot set
	int iterations_min, // Minimum number of iterations to consider in the histogram
	float2 v0, float2 v1, // Coordinates of the viewable complex plane
	float2 center,
	float2 windowSize,
	uint seed, // Seed value for random number generation
	float4 zm1,
	float4 zm2,
	float4 zm3,
	__global float* rm
)
{
	// this is the sample id
	int gId = get_global_id(0);

	// allocate array for storing the iterations that make up the orbit
	__private float2 orbit[MAX_ITERATIONS];

	// Initialize the random state with the seed value
	uint state = seed + gId;


	// get an initial sample
	int initial_sample_id = gId % initial_samples_size;
	float2 c = initial_samples_size > 0
		? mutate(&state, inital_samples[initial_sample_id].xy, size, v0, v1, 0.99)
		: (generateOnlyInRegion
			? (float2)(randf(&state, v0.x, v1.x), randf(&state, v0.y, v1.y))
			: (float2)(randf(&state, -2, 2), randf(&state, -2, 2)));

	// get an initial sample
	float2 z = c;

	// classic mandelbrot formula
	int iter = 0;
	for (; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
	{
		// store our position in the orbit for the histogram
		orbit[iter] = z = mandelise(z, c, zm1, zm2, zm3);
	}

	if (iter < iterations && iter >= iterations_min)
	{
		c = c - center;

		float p4d[4] = { 0.0f,0.0f,c.x,1.0f };
		float rotatedPoint[4];

		float newContribution = 0;

		int x = 0, y = 0;
		// accumulate the orbit historgram locally
		for (int i = 0; i < iter; ++i)
		{
			float2 t = orbit[i];

			t = t - center;

			p4d[0] = t.x;
			p4d[1] = t.y;

			// Apply the rotation matrix on the 4D point
			rotatedPoint[0] = rm[0] * p4d[0] + rm[4] * p4d[1] + rm[8] * p4d[2] + rm[12] * p4d[3];
			rotatedPoint[1] = rm[1] * p4d[0] + rm[5] * p4d[1] + rm[9] * p4d[2] + rm[13] * p4d[3];
			rotatedPoint[2] = rm[2] * p4d[0] + rm[6] * p4d[1] + rm[10] * p4d[2] + rm[14] * p4d[3];
			rotatedPoint[3] = rm[3] * p4d[0] + rm[7] * p4d[1] + rm[11] * p4d[2] + rm[15] * p4d[3];

			// Convert the 4D point back to 3D
			// Update the point coordinates
			t.x = rotatedPoint[0] / rotatedPoint[3];
			t.y = rotatedPoint[1] / rotatedPoint[3];

			t = t + center;

			// project back to screen space
			x = (t.x - v0.x) * windowSize.x;
			y = (t.y - v0.y) * windowSize.y;
			// make sure it is in the viewable region
			if (x >= 0 && x < w && y >= 0 && y < h)
			{
				// Atomic add guarantees that no histogram pixel will be overwritten by another worker,
				// but it will be slower (not by much from what I've seen :) )
				// It means it is the complete correct histogram we want
				if (USE_ATOMIC_ADD)
					atomic_add(&histogram[y * w + x], 1);
				else
					histogram[y * w + x] += 1;
				newContribution += 1;
			}
		}
		newContribution /= (float)iter;

		// Save better sample
		if (initial_samples_size > 0 && newContribution > inital_samples[initial_sample_id].z)
		{
			inital_samples[initial_sample_id].x = c.x;
			inital_samples[initial_sample_id].y = c.y;
			inital_samples[initial_sample_id].z = newContribution;
		}
	}
}
