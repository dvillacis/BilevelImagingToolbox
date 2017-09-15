__global__ void dual(float* p1, float* p2, const float* u_, 
		                       const double lambda, const double sigma,
				                              const int X, const int Y)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// center point
	int c = y*X + x;

	float nabla_x = 0.0f;
	float nabla_y = 0.0f;

	if (x < X-1)
		nabla_x = u_[c+1]-u_[c];

	if (y < Y-1)
		nabla_y = u_[c+X]-u_[c];

		        //p1[c] = fmaxf(-lambda, fminf(lambda, p1[c] + sigma*nabla_x));
		        //p2[c] = fmaxf(-lambda, fminf(lambda, p2[c] + sigma*nabla_y));

	p1[c] += sigma*nabla_x;
	p2[c] += sigma*nabla_y;
	float denom = fmaxf(1.0f, sqrt(p1[c]*p1[c] + p2[c]*p2[c])/lambda);
	p1[c] /= denom;
	p2[c] /= denom;
}
