from bilevel_imaging_toolbox import cuda_solvers
from bilevel_imaging_toolbox import image_utils
import timeit
import numpy as np

s_image_list = ['../examples/images/Playing_Cards_1.png','../examples/images/Playing_Cards_2.png','../examples/images/Playing_Cards_3.png']
n_trials = 20 
n_iters = 100

print('Testing CP ROF GPU')
for image_path in s_image_list:
    times = np.zeros(n_trials)
    image = image_utils.load_image(image_path)
    image = image_utils.convert_to_grayscale(image)
    g_image = image_utils.add_gaussian_noise(image)

    # Parameter Definition
    clambda = 0.2
    sigma = 1.9
    tau = 0.9/sigma

    for k in range(n_trials):
        start_time = timeit.default_timer()
        (fb_image,fb_values) = cuda_solvers.chambolle_pock_ROF_CUDA(g_image,clambda,tau,sigma,iters=n_iters)
        times[k] = timeit.default_timer()-start_time

    print('Execution time for CP CUDA for ROF on '+image_path+': %f,%f'%(np.mean(times[1:]),np.std(times[1:])))

print('Testing CP TVl1 CPU')
for image_path in s_image_list:
    times = np.zeros(n_trials)
    image = image_utils.load_image(image_path)
    image = image_utils.convert_to_grayscale(image)
    g_image = image_utils.add_gaussian_noise(image)

    # Parameter Definition
    clambda = 0.2
    sigma = 1.9
    tau = 0.9/sigma

    for k in range(n_trials):
        start_time = timeit.default_timer()
        (fb_image,fb_values) = cuda_solvers.chambolle_pock_TVl1_CUDA(g_image,clambda,tau,sigma,iters=n_iters)
        times[k] = timeit.default_timer()-start_time

    print('Execution time for CP CUDA for TV-l1 on '+image_path+': %f,%f'%(np.mean(times[1:]),np.std(times[1:])))
