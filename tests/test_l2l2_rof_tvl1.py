from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils
from bilevel_imaging_toolbox import plot_utils

### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/cameraman.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image,var=0.02)

# Parameter Definition
clambda = 0.4
tau = 0.5

# Call the l2l2 solver using Forward-Backward
(fb_image,fb_values) = solvers.forward_backward_l2_l2(g_image,clambda,tau,iters=200)
# Call ROF solver using Forward-Backward
clambda = 0.1
(cp_image,cp_values) = solvers.forward_backward_ROF(g_image,clambda,tau,iters=200)

# Plot resulting images
#image_utils.show_collection([image,g_image,fb_image,cp_image],["original","gaussian noise","denoised l2l2","denoised ROF"])
#plot_utils.plot_collection([fb_values],["FB"])

# Saving experiment
image_utils.save_image(image,'l2_vs_rof/original_cameraman.png')
image_utils.save_image(g_image,'l2_vs_rof/gaussian_noise_cameraman.png')
image_utils.save_image(cp_image,'l2_vs_rof/rof_cameraman.png')
image_utils.save_image(fb_image,'l2_vs_rof/l2l2_cameraman.png')
