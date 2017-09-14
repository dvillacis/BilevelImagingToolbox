from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils
from bilevel_imaging_toolbox import plot_utils

### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/lena.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image)#,var=0.02)

# Parameter Definition
clambda = 0.2
sigma = 1.9
tau = 0.9/sigma

# Call the solver using Forward-Backward
(fb_image,fb_values) = solvers.forward_backward_ROF(g_image,clambda,tau,iters=50)
# Call the solver using Chambolle-Pock
(cp_image,cp_values) = solvers.chambolle_pock_ROF(g_image,clambda,tau,sigma,iters=50)

# Plot resulting images
image_utils.show_collection([image,g_image,fb_image,cp_image],["original","gaussian noise","denoised FB","denoised CP"])
plot_utils.plot_collection([fb_values,cp_values],["Forward-Backward","Chambolle-Pock"],title="ROF Model FB vs CP Comparison",save_tikz=True)
