__global__ void extrapolate(float* xbar, float* xcur, float* xn, float theta, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (x < w && y < h) {
		int i;
		
		for (int z = 0; z < nc; z++) {
			i = x + w * y + w * h * z;
			xbar[i] = xn[i] + theta * (xn[i] - xcur[i]);
			xcur[i] = xn[i];
		}
	}
}
