

#define MAX_ITERATIONS 5000
#define INITIAL_SAMPLES_SIZE 3

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

__kernel void buddhabrot_mh(
	__global float3* inital_samples, // The initial samples array
	int initial_samples_size, // The size of the initial samples array
	__global int* histogram, // The output histogram data
	int w, int h, // Width and height of the output histogram
	int iterations, // Maximum number of iterations to compute the Mandelbrot set
	int iterations_min, // Minimum number of iterations to consider in the histogram
	float2 v0, float2 v1, // Coordinates of the viewable complex plane
	uint seed // Seed value for random number generation
)
{
	// this is the sample id
	int gId = get_global_id(0);
	// Calculate the work-group ID
	int wgId = get_group_id(0);

	// amount of samples
	int gSize = get_global_size(0);
	// amount of samples in this work group
	int wgSize = get_local_size(0);

	int workGroups = gSize / wgSize;

	// find the size of the viewable complex plane
	float2 size = v1 - v0;

	// the center of the viewable complex plane
	float2 center = size / 2.0f + v0;

	// for use in the loop for converting back to screen space
	float2 windowSize = { w / size.x, h / size.y };

	// allocate array for storing the iterations that make up the orbit
	__private float2 orbit[MAX_ITERATIONS];

	// Initialize the random state with the seed value
	uint state = seed + gId;

	// get an initial sample
	int initial_sample_id = gId % initial_samples_size;
	float2 c = inital_samples[initial_sample_id].xy;
	c = mutate(&state, c, size, v0, v1, 0.9);

	//float2 c = { randf(&state, v0.x, v1.x), randf(&state, v0.y, v1.y) };
	float2 z = c;

	// classic mandelbrot formula
	int iter = 0;
	for (float tmpX = 0.0f; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
	{
		tmpX = z.x; // Store the original value of z.x
		z.x = c.x + (z.x * z.x - z.y * z.y);
		z.y = c.y + (tmpX * z.y * 2.0f); // Use the original z.x value here

		// store our position in the orbit for the histogram
		orbit[iter] = z;
	}

	if (iter < iterations && iter >= iterations_min)
	{
		float newContribution = 0;

		int x = 0, y = 0;
		// accumulate the orbit historgram locally
		for (int i = 0; i < iter; ++i)
		{
			// project back to screen space
			x = (orbit[i].x - v0.x) * windowSize.x;
			y = (orbit[i].y - v0.y) * windowSize.y;
			// make sure it is in the viewable region
			if (x >= 0 && x < w && y >= 0 && y < h)
			{
				histogram[y * w + x] += 1;
				newContribution += 1;
			}
		}
		newContribution /= (float)iter;

		// Save better sample
		if (newContribution < inital_samples[initial_sample_id].z)
		{
			inital_samples[initial_sample_id].x = c.x;
			inital_samples[initial_sample_id].y = c.y;
			inital_samples[initial_sample_id].z = newContribution;
		}
	}
}

// Matrix-vector multiplication for 2x2 matrix represented as float4
float2 mat2xvec2(float4 mat, float2 vec) {
	return (float2)(dot(mat.xy, vec), dot(mat.zw, vec));
}

// Matrix multiplication for two 2x2 matrices represented as float4
float4 mat2xmat2(float4 mat1, float4 mat2) {
	float2 row1 = (float2)(dot(mat1.xy, mat2.xy), dot(mat1.zw, mat2.xy));
	float2 row2 = (float2)(dot(mat1.xy, mat2.zw), dot(mat1.zw, mat2.zw));
	return (float4)(row1, row2);
}

float2 mandelise(float2 z, float2 c, float4 zm1, float4 zm2, float4 zm3) {
	float2 z1 = z;
	float xx = z1.x * z1.x;
	float yy = z1.y * z1.y;
	float2 z2 = (float2)(xx - yy, z1.x * z1.y * 2.0f);
	float2 z3 = (float2)(xx * z1.x - 3.0f * z1.x * yy, 3.0f * xx * z1.y - yy * z1.y);

	// Perform matrix-vector multiplication and addition as if they are glm::mat2 and glm::vec2
	float2 n = mat2xvec2(zm3, z3) + mat2xvec2(zm2, z2) + mat2xvec2(zm1, z1) + c;
	return n;
}

__kernel void buddhabrot(
	__global int* histogram, // The output histogram data
	int w, int h, // Width and height of the output histogram
	int iterations, // Maximum number of iterations to compute the Mandelbrot set
	int iterations_min, // Minimum number of iterations to consider in the histogram
	float2 v0, float2 v1, // Coordinates of the viewable complex plane
	uint seed, // Seed value for random number generation
	float4 zm1,
	float4 zm2,
	float4 zm3
)
{
	// this is the sample id
	int gId = get_global_id(0);
	// Calculate the work-group ID
	int wgId = get_group_id(0);

	// amount of samples
	int gSize = get_global_size(0);
	// amount of samples in this work group
	int wgSize = get_local_size(0);

	int workGroups = gSize / wgSize;

	// find the size of the viewable complex plane
	float2 size = v1 - v0;

	// the center of the viewable complex plane
	float2 center = size / 2.0f + v0;

	// for use in the loop for converting back to screen space
	float2 windowSize = { w / size.x, h / size.y };

	// allocate array for storing the iterations that make up the orbit
	__private float2 orbit[MAX_ITERATIONS];

	// Initialize the random state with the seed value
	uint state = seed + gId;

	// get an initial sample
	//float2 c = { randf(&state, v0.x, v1.x), randf(&state, v0.y, v1.y) };
	float2 c = { randf(&state, -2, 2), randf(&state, -2, 2) };
	float2 z = c;

	// classic mandelbrot formula
	int iter = 0;
	for (float tmpX = 0.0f; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
	{
		//tmpX = z.x; // Store the original value of z.x
		//z.x = c.x + (z.x * z.x - z.y * z.y);
		//z.y = c.y + (tmpX * z.y * 2.0f); // Use the original z.x value here

		z = mandelise(z, c, zm1, zm2, zm3);

		// store our position in the orbit for the histogram
		orbit[iter] = z;
	}

	if (iter < iterations && iter >= iterations_min)
	{
		int x = 0, y = 0;
		// accumulate the orbit historgram locally
		for (int i = 0; i < iter; ++i)
		{
			// project back to screen space
			x = (orbit[i].x - v0.x) * windowSize.x;
			y = (orbit[i].y - v0.y) * windowSize.y;
			// make sure it is in the viewable region
			if (x >= 0 && x < w && y >= 0 && y < h)
				histogram[y * w + x] += 1;
		}
	}
}



//// Define the 4D point and the rotation matrix as float arrays
//p4d[2] = t.x;
//p4d[3] = t.y;
//
//// Apply the rotation matrix on the 4D point
//float rotatedPoint[4];
//rotatedPoint[0] = rm[0] * p4d[0] + rm[4] * p4d[1] + rm[8] * p4d[2] + rm[12] * p4d[3];
//rotatedPoint[1] = rm[1] * p4d[0] + rm[5] * p4d[1] + rm[9] * p4d[2] + rm[13] * p4d[3];
//rotatedPoint[2] = rm[2] * p4d[0] + rm[6] * p4d[1] + rm[10] * p4d[2] + rm[14] * p4d[3];
//rotatedPoint[3] = rm[3] * p4d[0] + rm[7] * p4d[1] + rm[11] * p4d[2] + rm[15] * p4d[3];
//
//// Convert the 4D point back to 3D
//float newX = rotatedPoint[0] / rotatedPoint[3];
//float newY = rotatedPoint[1] / rotatedPoint[3];
//float newZ = rotatedPoint[2] / rotatedPoint[3];
//
//// Update the point coordinates
//t.x = newX;
//t.y = newY;