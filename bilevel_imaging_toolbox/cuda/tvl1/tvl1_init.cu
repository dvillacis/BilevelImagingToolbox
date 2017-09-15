__global__ void init(float* xbar, float* xcur, float* xn, float* y1, float* y2, float* img, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h) {
		int i;
		float val;
		for (int z = 0; z < nc; z++) {
			i = x + w * y + w * h * z;
			val = img[i];
			xbar[i] = val;
			xn[i] = val;
			xcur[i] = val;
			y1[i] = 0.f;
			y2[i] = 0.f;
		}
	}
}
