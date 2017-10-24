import numpy as np

from bilevel_imaging_toolbox import solvers
from bilevel_imaging_toolbox import plot_utils

### Testing LASSO solver using ISTA method

m,n = 15,20

# Create random A matrix
rng = np.random.RandomState(42)
A = rng.randn(m,n)

# Create initial point
x0 = rng.randn(n)
x0[x0 < 0.9] = 0 # Add sparcity to the initial point
b = np.dot(A,x0)
clambda = 0.9 # Regularization parameter
iters = 200

# Run ISTA Solver
(x_ista,vallog_ista) = solvers.ista_LASSO(x0,A,b,clambda,iters)

# Run FISTA Solver
(x_fista, vallog_fista) = solvers.fista_LASSO(x0,A,b,clambda,iters)

# Plot error evolution
plot_utils.plot_collection([vallog_ista,vallog_fista],['ISTA','FISTA'],title="LASSO ISTA - FISTA evolution")
