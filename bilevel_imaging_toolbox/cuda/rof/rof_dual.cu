__global__ void dual(float* xn, float* xcur, float* y1, float* y2, float* img, float tau, float lambda, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (x < w && y < h) {
		int i;
		float d1, d2, val;
					            
		for (int z = 0; z < nc; z++) {
			i = x + w * y + w * h * z;
									            
			d1 = (x+1 < w ? y1[i] : 0.f) - (x>0 ? y1[(x-1) + w * y + w * h * z] : 0.f);
			d2 = (y+1 < h ? y2[i] : 0.f) - (y>0 ? y2[x + w * (y-1) + w * h * z] : 0.f);
			val = xcur[i] + tau * (d1 + d2);

			xn[i] = (val + tau * lambda * img[i]) / (1.f + tau * lambda);
		}
	}
}

