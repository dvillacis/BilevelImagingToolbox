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

# Plot both images
#image_utils.show_collection([image,n_image],["original","gaussian noise"])
# Parameter Definition
clambda = 0.9
tau = 1
sigma = 1/tau

# Call the solver using Chambolle-Pock
(cp_image,cp_values) = solvers.chambolle_pock_TVl1(n_image,clambda,tau,sigma,iters=200)

# Plot resulting images
image_utils.show_collection([image,n_image,cp_image],["original","s&p","denoised impulse tv-l1"])
plot_utils.plot_collection([cp_values],["CP"])
