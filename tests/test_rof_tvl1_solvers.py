from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils
from bilevel_imaging_toolbox import plot_utils

### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/lena.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add impulse noise to the image
n_image = image_utils.add_impulse_noise(image,amount=0.2)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image,var=0.01)

# Save impulse noisy image
image_utils.save_image(n_image,'impulse_noise_lena.png')

# Plot both images
#image_utils.show_collection([image,n_image],["original","gaussian noise"])
# Parameter Definition
# clambda = 0.1
# sigma = 1.9
# tau = 0.9/sigma
clambda = 0.1
sigma = 1.9
tau = 0.9/sigma

# Call the solver using Forward-Backward
#(fb_image,fb_values) = solvers.forward_backward_ROF(n_image,clambda,tau,iters=50)
# Call the solver using Chambolle-Pock
(cp_image,cp_values) = solvers.chambolle_pock_ROF(n_image,clambda,tau,sigma,iters=200)
(cpg_image,cpg_values) = solvers.chambolle_pock_ROF(g_image,clambda,tau,sigma,iters=200)

# Save rof denoised images
image_utils.save_image(cp_image,'rof_impulse_denoised_lena.png')

clambda = 0.9
tau = 1
sigma = 1/tau
(cp2_image,cp2_values) = solvers.chambolle_pock_TVl1(n_image,clambda,tau,sigma,iters=200)
(cp2g_image,cp2g_values) = solvers.chambolle_pock_TVl1(g_image,clambda,tau,sigma,iters=200)

# Save tvl1 denoised images
image_utils.save_image(cp2_image,'tvl1_impulse_denoised_lena.png')

# Plot resulting images
image_utils.show_collection([image,g_image,cpg_image,cp2g_image],["original","gaussian noise","denoised ROF","denoised TV-l1"])
image_utils.show_collection([image,n_image,cp_image,cp2_image],["original","impulse noise","denoised ROF","denoised TV-l1"])
#plot_utils.plot_collection([fb_values,cp_values],["CP","CP"])
