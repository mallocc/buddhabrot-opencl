

#define MAX_ITERATIONS 5000

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

__kernel void buddhabrot(
	//__global uchar* inital_samples,
	int inital_samples_size, // The size of the initial samples array
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
	//float2 c = inital_samples[gId % inital_samples_size];
	float2 c = { randf(&state, v0.x, v1.x), randf(&state, v0.y, v1.y) };
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
