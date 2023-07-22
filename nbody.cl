#define UNROLL_FACTOR  1

__kernel void black_hole(float dt, float eps, 
		__local float4* pblock,
		__global float4* pos_old, 
		__global float4* vel_old, 
		int N)
{
	int gti = get_global_id(0);
	int ti = get_local_id(0);

	int n = get_global_size(0);
	int nt = get_local_size(0);
	int nb = n/nt;

	float4 p = pos_old[gti];
	float4 v = vel_old[gti];
	float m = p.w;
	float4 a = {0.0f,0.0f,0.0f,0.0f};

	for(int jb=0; jb < nb; jb++) 
	{ /* Foreach block ... */
	    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in the work-group */

		int j=0; 
        for(; (j+UNROLL_FACTOR)<nt; j++) 
		{ 
		#pragma unroll UNROLL_FACTOR
		for(int k = 0; k<UNROLL_FACTOR; j++,k++) 
			{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
			 }
        }
		for(; j<nt; j++) 
		{
			float4 p2 = pblock[j]; /* Read a cached particle position */
			float4 d = p2 -  p;
			float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
			float f = p2.w * invr*invr*invr;
			a += f*d; /* Accumulate acceleration */
		}

       barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
      }

	  //black hole
	float4 d = -p;
	float invr = rsqrt(d.x*d.x + d.y*d.y + eps*eps);
	float f = 100000.0f * invr*invr*invr;
	a += f*d; 

	float4 newpos = p + dt*v + 0.5f*dt*dt*a;
	float4 newvel = v + dt*a;

	newpos.w = m;
	pos_old[gti] = newpos;
	vel_old[gti] = newvel;
}