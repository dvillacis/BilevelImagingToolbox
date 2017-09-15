from bilevel_imaging_toolbox import cuda_solvers
from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils


### Testing dual rof model
# Loading image
image = image_utils.load_image('../examples/images/Playing_Cards_3.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
#n_image = image_utils.add_impulse_noise(image,amount=0.2)

clambda = 0.2
sigma = 1.9
tau = 0.9/sigma

#Call the solver using Chambolle-Pock
#(cp_image,cp_values) = solvers.chambolle_pock_ROF(g_image,clambda,tau,sigma,200)

# Call the solver using CUDA Chambolle-Pock
(ccp_image,ccp_values) = cuda_solvers.chambolle_pock_TVl1_CUDA(image,clambda,tau,sigma,100)
image_utils.save_image(ccp_image,'CUDA_TVl1_PC3.png')

# Plot resulting images
#image_utils.show_collection([image,g_image,fb_image,cp_image],["original","gaussian noise","denoised FB","denoised CP"])
#plot_utils.plot_collection([fb_values,cp_values],["FB","CP"])
