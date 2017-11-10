###################################################
##### BILEVEL OPTIMIZATION (SINGLE-GAUSSIAN) ######
###################################################

# DAVID VILLACIS
# REFERENCE: De Los Reyes-Schoenlieb '13

# FEATURES:
# - Gaussian fidelity
# - single training pair
# - BFGS optimisation with SSN solver



from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import image_utils

### Importing imaging data
# Loading image
image = image_utils.load_image('../examples/images/lena.png')
# Convert it to grayscale
image = image_utils.convert_to_grayscale(image)
# Add gaussian noise to the image
g_image = image_utils.add_gaussian_noise(image)

### Initialization
clambda = 100
sigma = 1.9
tau = 0.9/sigma
B = 1 # Initial BFGS Matrix
u = g_image # Initialize as the noisy image
iters = 50

for i in range(iters):
    print("###### Iteration %d ######"%(i+1))

    # Solve the state equation
    (u,u_log) = solvers.chambolle_pock_ROF(u,clambda,tau,sigma,10)

    # Adjoint Solver
