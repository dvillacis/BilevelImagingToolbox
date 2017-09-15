import numpy as np
import timeit
import os

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as drv
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

	(h,w) = image.shape
	dim = w*h
	nc = 1

	# Load Modules
	init_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/rof/rof_init.cu','r').read())
	primal_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/rof/rof_primal.cu','r').read())
	dual_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/rof/rof_dual.cu','r').read())
	extrapolate_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/rof/rof_extrapolate.cu','r').read())
	solution_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/rof/rof_solution.cu','r').read())
	
	# Memory Allocation
	nbyted = image.astype(np.float32).nbytes
	d_imgInOut = drv.mem_alloc(nbyted)
	d_x = drv.mem_alloc(nbyted)
	d_xbar = drv.mem_alloc(nbyted)
	d_xcur = drv.mem_alloc(nbyted)
	d_y1 = drv.mem_alloc(nbyted)
	d_y2 = drv.mem_alloc(nbyted)

	# Variables
	w = np.int32(w)
	h = np.int32(h)
	nc = np.int32(nc)
	sigma = np.float32(sigma)
	tau = np.float32(tau)
	clambda = np.float32(clambda)

	# Copy host memory
	h_img = image.astype(np.float32)
	drv.memcpy_htod(d_imgInOut,h_img)

	# Launch kernel
	block = (16,16,1)
	grid = (np.ceil((w+block[0]-1)/block[0]),np.ceil((h+block[1]-1)/block[1]))
	grid = (int(grid[0]),int(grid[1]))
	
	# Function definition
	init_func = init_module.get_function('init')
	primal_func = primal_module.get_function('primal')
	dual_func = dual_module.get_function('dual')
	extrapolate_func = extrapolate_module.get_function('extrapolate')
	solution_func = solution_module.get_function('solution')

	# Initialization
	init_func(d_xbar, d_xcur, d_x, d_y1, d_y2, d_imgInOut, np.int32(w), np.int32(h), np.int32(nc), block=block, grid=grid)
	
	for i in range(iters):
		primal_func(d_y1,d_y2,d_xbar,sigma,w,h,nc,block=block,grid=grid)
		dual_func(d_x,d_xcur,d_y1,d_y2,d_imgInOut,tau,clambda,w,h,nc,block=block,grid=grid)
		extrapolate_func(d_xbar,d_xcur,d_x,np.float32(0.5),w,h,nc,block=block,grid=grid)
	solution_func(d_imgInOut,d_x,w,h,nc,block=block,grid=grid)

	drv.memcpy_dtoh(h_img,d_imgInOut)

	print("Finished Chambolle-Pock ROF CUDA denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

	return(h_img,0)

def chambolle_pock_TVl1_CUDA(image, clambda, tau, sigma, iters=100):
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
	print("2D Primal-Dual TV-l1 CUDA solver using Chambolle-Pock method")

	start_time = timeit.default_timer()

	(h,w) = image.shape
	dim = w*h
	nc = 1

	# Load Modules
	init_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/tvl1/tvl1_init.cu','r').read())
	primal_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/tvl1/tvl1_primal.cu','r').read())
	dual_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/tvl1/tvl1_dual.cu','r').read())
	extrapolate_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/tvl1/tvl1_extrapolate.cu','r').read())
	solution_module = SourceModule(open('../bilevel_imaging_toolbox/cuda/tvl1/tvl1_solution.cu','r').read())

	# Memory Allocation
	nbyted = image.astype(np.float32).nbytes
	d_imgInOut = drv.mem_alloc(nbyted)
	d_x = drv.mem_alloc(nbyted)
	d_xbar = drv.mem_alloc(nbyted)
	d_xcur = drv.mem_alloc(nbyted)
	d_y1 = drv.mem_alloc(nbyted)
	d_y2 = drv.mem_alloc(nbyted)

	# Copy host memory
	h_img = image.astype(np.float32)
	drv.memcpy_htod(d_imgInOut,h_img)

	# Launch kernel
	block = (16,16,1)
	grid = (np.ceil((w+block[0]-1)/block[0]),np.ceil((h+block[1]-1)/block[1]))
	grid = (int(grid[0]),int(grid[1]))

	# Function definition
	init_func = init_module.get_function('init')
	primal_func = primal_module.get_function('primal')
	dual_func = dual_module.get_function('dual')
	extrapolate_func = extrapolate_module.get_function('extrapolate')
	solution_func = solution_module.get_function('solution')
	
	# Initialization
	init_func(d_xbar, d_xcur, d_x, d_y1, d_y2, d_imgInOut, np.int32(w), np.int32(h), np.int32(nc), block=block, grid=grid)
	w = np.int32(w)
	h = np.int32(h)
	nc = np.int32(nc)
	sigma = np.float32(sigma)
	tau = np.float32(tau)
	clambda = np.float32(clambda)

	for i in range(iters):
		primal_func(d_y1,d_y2,d_xbar,sigma,w,h,nc,block=block,grid=grid)
		dual_func(d_x,d_xcur,d_y1,d_y2,d_imgInOut,tau,clambda,w,h,nc,block=block,grid=grid)
		extrapolate_func(d_xbar,d_xcur,d_x,np.float32(0.5),w,h,nc,block=block,grid=grid)
	solution_func(d_imgInOut,d_x,w,h,nc,block=block,grid=grid)

	drv.memcpy_dtoh(h_img,d_imgInOut)

	print("Finished Chambolle-Pock TV-l1 CUDA denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

	return(h_img,0)







