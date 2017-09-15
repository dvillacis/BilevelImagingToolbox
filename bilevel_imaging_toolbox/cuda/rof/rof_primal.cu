__global__ void primal(float *y1, float *y2, float *xbar, float sigma, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (x < w && y < h) {
		int i;
		float x1, x2, val, norm;
					        
		for (int z = 0; z < nc; z++) {
			i = x + w * y + w * h * z;

			val = xbar[i];
			x1 = (x+1<w) ? (xbar[(x+1) + w * y + w * h * z] - val) : 0.f;
			x2 = (y+1<h) ? (xbar[x + w * (y+1) + w * h * z] - val) : 0.f;

			x1 = y1[i] + sigma * x1;
			x2 = y2[i] + sigma * x2;

			norm = sqrtf(x1*x1+x2*x2);

			y1[i] = x1 / fmax(1.f, norm);
			y2[i] = x2 / fmax(1.f, norm);
		}
	}
}
