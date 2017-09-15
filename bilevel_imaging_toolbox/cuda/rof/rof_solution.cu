__global__ void solution(float* img, float* xbar, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < w && y < h) {
		int i;
		
		for (int z = 0; z < nc; z++) {
			i = x + w * y + w * h * z;
			img[i] = xbar[i];
		}
	}
}
