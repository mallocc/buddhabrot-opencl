__kernel 
void gravity(__global float3* pos, __global float3* vel , float deltaTime, float3 centre 
		,__global float3* newPosition, __global float3* newVelocity) {

    unsigned int gid = get_global_id(0);
    float3 myPos = pos[gid];

	//find force from centre
	float3 dir = centre.xy - myPos.xy;
	float r = sqrt(dir.x * dir.x + dir.y * dir.y);
	float3 n = dir / r;
	float force = centre.z * mypos.z / pow(r,3);

	//update velocity
	float3 newVel;
	newVel.xy = newVel.xy + (force * n) * deltaTime;

	//update position
	float3 newPos;
	newPos.xy =  newPos.xy + newVel.xy * deltaTime;

	    // write to global memory
    newPosition[gid] = newPos;
    newVelocity[gid] = newVel;
}