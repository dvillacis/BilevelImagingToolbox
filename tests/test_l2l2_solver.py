from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils
from bilevel_imaging_toolbox import plot_utils

### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/lena.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image,var=0.02)

# Parameter Definition
clambda = 0.4
sigma = 1.9
tau = 0.9/sigma

# Call the solver using Forward-Backward
(fb_image,fb_values) = solvers.forward_backward_l2_l2(g_image,clambda,tau,iters=200)

# Plot resulting images
image_utils.show_collection([image,g_image,fb_image],["original","gaussian noise","denoised FB"])
plot_utils.plot_collection([fb_values],["FB"])
