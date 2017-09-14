from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils
import timeit
import numpy as np

s_image_list = ['../examples/images/Playing_Cards_1.png','../examples/images/Playing_Cards_2.png','../examples/images/Playing_Cards_3.png']
n_trials = 4
n_iters = 100

print('Testing FB ROF CPU')
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
        (fb_image,fb_values) = solvers.forward_backward_ROF(g_image,clambda,tau,iters=n_iters)
        times[k] = timeit.default_timer()-start_time

    print('Execution time for FB for ROF on '+image_path+': %f,%f'%(np.mean(times),np.std(times)))

print('Testing CP ROF CPU')
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
        (fb_image,fb_values) = solvers.chambolle_pock_ROF(g_image,clambda,tau,sigma,iters=n_iters)
        times[k] = timeit.default_timer()-start_time

    print('Execution time for CP for ROF on '+image_path+': %f,%f'%(np.mean(times),np.std(times)))

print('Testing CP TVl1 CPU')
for image_path in s_image_list:
    times = np.zeros(n_iters)
    image = image_utils.load_image(image_path)
    image = image_utils.convert_to_grayscale(image)
    g_image = image_utils.add_gaussian_noise(image)

    # Parameter Definition
    clambda = 0.2
    sigma = 1.9
    tau = 0.9/sigma

    for k in range(n_trials):
        start_time = timeit.default_timer()
        (fb_image,fb_values) = solvers.chambolle_pock_TVl1(g_image,clambda,tau,sigma,iters=n_iters)
        times[k] = timeit.default_timer()-start_time

    print('Execution time for CP for ROF on '+image_path+': %f,%f'%(np.mean(times),np.std(times)))
