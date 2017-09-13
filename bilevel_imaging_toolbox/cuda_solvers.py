import numpy as np
import timeit
import os

# PyCUDA imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def chambolle_pock_ROF_CUDA(image, clambda, tau, sigma, iters=100):
	r""" 2D ROF CUDA solver using Chambolle-Pock Method

	Parameters
	----------
	image : numpy array
		The noisy image we are processing
	clambda : float
		The non-negative weight in the optimization problem
	tau : float
		Parameter of the proximal operator
	iters : int
		Number of iterations allowed

	"""
	print("2D Primal-Dual ROF CUDA solver using Chambolle-Pock method")

	start_time = timeit.default_timer()

	(X,Y) = image.shape
	
	os.chdir(os.path.dirname(__file__)) # Set the correct working directory
	
	primal_module = SourceModule(open('cuda/rof_chambolle_pock_primal_step.cu','r').read())
	dual_module = SourceModule(open('cuda/rof_chambolle_pock_dual_step.cu','r').read())
	
	blocksize = 16

	primal_func = primal_module.get_function('primal')
	dual_func = dual_module.get_function('dual')

	tau = 1/np.sqrt(8)
	sigma = 1/np.sqrt(8)

	single_image = image.astype(np.float32)

	# Allocate memory for gpu variables
	image_gpu = cuda.mem_alloc(single_image.nbytes)
	x_gpu = cuda.mem_alloc(single_image.nbytes)
	xnew_gpu = cuda.mem_alloc(single_image.nbytes)
	y1_gpu = cuda.mem_alloc(np.zeros(image.shape).astype(np.float32).nbytes)
	y2_gpu = cuda.mem_alloc(np.zeros(image.shape).astype(np.float32).nbytes)

	# Move content to variables
	cuda.memcpy_htod(image_gpu,single_image)
	cuda.memcpy_htod(x_gpu,single_image)

	#for i in range(iters):
	primal_func(x_gpu,xnew_gpu,image_gpu,y1_gpu,y2_gpu,np.float32(tau),np.int32(X),np.int32(Y),block=(blocksize,blocksize,1),shared=0)
	dual_func(y1_gpu,y2_gpu,xnew_gpu,np.float32(clambda),np.float32(sigma),np.int32(X),np.int32(Y),block=(blocksize,blocksize,1),shared=0)

	cuda.synchronize()

	u = np.empty_like(single_image)
	cuda.memcpy_dtoh(u,x_gpu)

	print("Finished Chambolle-Pock ROF CUDA denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

	return(u,0)









