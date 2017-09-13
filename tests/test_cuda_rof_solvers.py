from bilevel_imaging_toolbox import cuda_solvers
from bilevel_imaging_toolbox import image_utils


### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/lena.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image,var=0.02)

# Plot both images
#image_utils.show_collection([image,n_image],["original","gaussian noise"])
# Parameter Definition
clambda = 0.4
sigma = 1.9
tau = 0.9/sigma

# Call the solver using CUDA Chambolle-Pock
(cp_image,cp_values) = cuda_solvers.chambolle_pock_ROF_CUDA(g_image,clambda,tau,sigma,200)

# Plot resulting images
#image_utils.show_collection([image,g_image,fb_image,cp_image],["original","gaussian noise","denoised FB","denoised CP"])
#plot_utils.plot_collection([fb_values,cp_values],["FB","CP"])
