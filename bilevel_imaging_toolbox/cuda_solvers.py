import numpy as np
import timeit

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

	(X,Y) = image.shape

	primal_step = SourceModule(open('cuda/rof_chambolle_pock_primal_step.cu','r').read())
	dual_step = SourceModule(open('cuda/rof_chambolle_pock_dual_step.cu','r').read())
	
	blocksize = 16

	primal_func = primal_step.get_function('primal')
	dual_func = dual_step.get_function('dual')

	tau = 1/np.sqrt(8)
	sigma = 1/np.sqrt(8)

	for i in range(iters):
		func(x_gpu,xnew_gpu,image_gpu,y1_gpu,y2_gpu,tau,Y,X)
		func(y1_gpu,y2_gpu,xnew_gpu,clambda,sigma,Y,X)

	print("Finished Chambolle-Pock ROF CUDA denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))










