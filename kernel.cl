

#define MAX_ITERATIONS 1000

float complex_mag2(float2 a)
{
	return a.x * a.x + a.y * a.y;
}

__kernel void buddhabrot(
	__global float2* inital_samples,
	int inital_samples_size,
	__global int* histogram,
	int w, int h,
	int iterations,
	float2 v0, float2 v1
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

	// get an initial sample
	float2 c = inital_samples[gId % inital_samples_size];
	float2 z = c;

	// classic mandelbrot formula
	int iter = 0;
	for (; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
	{
		float tmpX = z.x; // Store the original value of z.x
		z.x = c.x + (z.x * z.x - z.y * z.y);
		z.y = c.y + (tmpX * z.y * 2.0f); // Use the original z.x value here

		// store our position in the orbit for the histogram
		orbit[iter] = z;
	}

	if (iter < iterations)
		// accumulate the orbit historgram locally
		for (int i = 0; i < iter; ++i)
		{
			float2 t = orbit[i];
			// project back to screen space
			int x = (t.x - v0.x) * windowSize.x;
			int y = (t.y - v0.y) * windowSize.y;
			// make sure it is in the viewable region
			if (x >= 0 && x < w && y >= 0 && y < h)
			{
				histogram[y * w + x] += 1;
				//atom_add(&histogram[y * w + x], 1);
			}
		}
}

// Define fixed dimensions for localHistogram
#define LOCAL_HISTOGRAM_WIDTH 128
#define LOCAL_HISTOGRAM_HEIGHT 128

//__kernel void buddhabrot_reduction(
//	__global float2* inital_samples,
//	int inital_samples_size,
//	__global int* histogram,
//	int w, int h,
//	int iterations,
//	float2 v0, float2 v1
//)
//{
//	// this is the sample id
//	int gId = get_global_id(0);
//	// Calculate the work-group ID
//	int wgId = get_group_id(0);
//
//	// amount of samples
//	int gSize = get_global_size(0);
//	// amount of samples in this work group
//	int wgSize = get_local_size(0);
//
//	int workGroups = gSize / wgSize;
//
//	// find the size of the viewable complex plane
//	float2 size = v1 - v0;
//
//	// the center of the viewable complex plane
//	float2 center = size / 2.0f + v0;
//
//	// for use in the loop for converting back to screen space
//	float2 windowSize = { w / size.x, h / size.y };
//
//	// allocate array for storing the iterations that make up the orbit
//	__private float2 orbit[MAX_ITERATIONS];
//
//	// get an initial sample
//	float2 c = inital_samples[gId % inital_samples_size];
//	float2 z = c;
//
//	// classic mandelbrot formula
//	int iter = 0;
//	for (; iter < iterations && complex_mag2(z) < 4.0f; ++iter)
//	{
//		z.x = c.x + (z.x * z.x - z.y * z.y);
//		z.y = c.y + (z.x * z.y * 2.0f);
//
//		// store our position in the orbit for the histogram
//		orbit[iter] = z;
//	}
//
//	// Local histogram buffer for each work-item
//	__local int localHistogram[LOCAL_HISTOGRAM_WIDTH * LOCAL_HISTOGRAM_HEIGHT];
//	for (int i = 0; i < w * h; i++)
//		localHistogram[i] = 0;
//
//	// accumulate the orbit historgram locally
//	for (int i = 0; i < iter; ++i)
//	{
//		float2 t = orbit[i];
//		// project back to screen space
//		int x = (t.x - v0.x) * windowSize.x;
//		int y = (t.y - v0.y) * windowSize.y;
//		// make sure it is in the viewable region
//		if (x >= 0 && x < w && y >= 0 && y < h)
//			localHistogram[y * w + x] += 1;
//	}
//
//	// Synchronize work-items within the work-group
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// Perform parallel reduction within the work-group using binary tree reduction
//	// Start with a step size of half the work-group size, and halve it in each iteration
//	// until the step size becomes 0.
//	for (int step = wgSize / 2; step > 0; step >>= 1)
//	{
//		// Each work-item with a local ID less than 'step' participates in the reduction.
//		// Work-items at local ID >= 'step' do not participate, as they have already combined
//		// their partial results with other work-items in the previous iteration.
//		if (wgId < step && wgId + step < w * h)
//		{
//			// Combine the partial histogram value of this work-item with the partial histogram
//			// value of the work-item 'step' positions away. This effectively reduces the number
//			// of active work-items in the reduction by half in each iteration.
//			localHistogram[wgId] += localHistogram[wgId + step];
//		}
//
//		// Synchronize the work-items within the work-group to ensure that all additions for
//		// the current step are completed before moving to the next step.
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//
//	// The reduced histogram value is stored in localHistogram[0]
//	// Now, we accumulate the work-group reductions into the global histogram
//	if (wgId == 0)
//	{
//		int groupId = gId / wgSize; // Calculate work-group ID outside the loop
//		for (int i = 0; i < w * h; i += wgSize)
//		{
//			int index = groupId * wgSize + i;
//			if (index < w * h)
//			{
//				histogram[index] += localHistogram[0];
//			}
//		}
//	}
//}