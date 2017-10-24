import numpy as np
from scipy import linalg
import timeit
from bilevel_imaging_toolbox import operators

def ROF_value(f,x,y,clambda):
    r""" Compute the ROF cost functional

    Parameters
    ----------
    f : numpy array
        Noisy input image
    x : numpy array
        Primal variable value
    y : numpy array
        Dual variable value
    clambda : float
        Tickonov regularization parameter
    """
    a = np.linalg.norm((f-x).flatten())**2/2
    b = np.sum(np.sqrt(np.sum(y**2,axis=2)).flatten())
    return a+clambda*b

def ROF_duality_gap(f,x,y,clambda):
    pass

def TVl1_value(f,x,y,clambda):
    r""" Compute the TV-l1 cost functional

    Parameters
    ----------
    f : numpy array
        Noisy input image
    x : numpy array
        Primal variable value
    y : numpy array
        Dual variable value
    clambda : float
        Tickonov regularization parameter
    """
    a = np.linalg.norm((f-x).flatten(),1)**2/2
    b = np.sum(np.sqrt(np.sum(y**2,axis=2)).flatten())
    return a+clambda*b

def forward_backward_ROF(image, clambda, tau, iters=100):
    r""" Dual ROF solver using Forward-Backward Splitting

    Parameters
    ----------
    image : numpy array
        The noisy image we are processing
    clambda : float
        The non-negative weight in the optimization problem
    tau : float
        Parameter of the proximal operator
    iters : int
        Number of iterations allowed

    """
    print("2D Dual ROF solver using Forward-Backward Splitting")

    start_time = timeit.default_timer()

    op = operators.make_finite_differences_operator(image.shape,'fn',1)
    tau = tau / op.bound**2
    y = op.val(image)
    x = image

    vallog = np.zeros(iters)

    for i in range(iters):
        y = operators.prox_project(clambda,y-tau*op.val((op.conj(y)-image)))
        x = image-op.conj(y) #Retrieve primal value
        vallog[i] = ROF_value(image,x,op.val(x),clambda)

    print("Finished Forward-Backward Dual ROF denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)

def chambolle_pock_ROF(image, clambda, tau, sigma, iters=100):
    r""" 2D ROF solver using Chambolle Pock Method

    Parameters
    ----------
    image : numpy array
        The noisy image we are processing
    clambda : float
        The non-negative weight in the optimization problem
    tau : float
        Parameter of the proximal operator
    iters : int
        Number of iterations allowed

    """
    print("2D Primal-Dual ROF solver using Chambolle-Pock method")

    start_time = timeit.default_timer()

    op = operators.make_finite_differences_operator(image.shape,'fn',1)
    tau = tau / op.bound
    sigma = sigma / op.bound
    y = op.val(image)
    x = image

    vallog = np.zeros(iters)

    x = image

    for i in range(iters):
        xnew = (x + tau*(image - op.conj(y)))/(1+tau)
        xbar = 2*xnew - x
        y = operators.prox_project(clambda, y+sigma*op.val(xbar))
        x = xnew
        vallog[i] = ROF_value(image,x,op.val(x),clambda)

    print("Finished Chambolle-Pock ROF denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)

def admm_ROF(image, clambda, tau, iters=100):
    pass

def chambolle_pock_TVl1(image, clambda, tau, sigma, iters=100):
    r""" 2D TV-l1 solver using Chambolle Pock Method

    Parameters
    ----------
    image : numpy array
        The noisy image we are processing
    clambda : float
        The non-negative weight in the optimization problem
    tau : float
        Parameter of the proximal operator
    sigma: float
        Parameter of the proximal operator
    iters : int
        Number of iterations allowed

    """
    print("2D Primal-Dual TV-l1 solver using Chambolle-Pock method")

    start_time = timeit.default_timer()

    op = operators.make_finite_differences_operator(image.shape,'fn',1)
    tau = tau / op.bound
    sigma = sigma / op.bound

    y = op.val(image)
    x = image

    vallog = np.zeros(iters)

    for i in range(iters):
        xhat = x - tau * op.conj(y)
        xnew = image + np.multiply(np.maximum(0,np.absolute(xhat-image)-tau),np.sign(xhat-image)) # Proximal of \|x-f\|_1
        xbar = 2*xnew - x
        y = operators.prox_project(clambda, y+sigma*op.val(xbar))
        x = xnew
        vallog[i] = TVl1_value(image,x,op.val(x),clambda)

    print("Finished Chambolle-Pock TV-l1 denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)

def forward_backward_l2_l2(image,clambda,tau,iters=100):
    print("2D l2-l2 solver using Forward-Backward Splitting")

    start_time = timeit.default_timer()

    op = operators.make_finite_differences_operator(image.shape,'fn',1)
    tau = tau / op.bound**2
    vallog = np.zeros(iters)
    x=image

    for i in range(iters):
        x = (x+tau*(image-clambda*op.conj(op.val(x))))/(1+tau)
        vallog[i] = ROF_value(image,x,op.val(x),clambda)

    print("Finished Forward-Backward l2-l2 denoising in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)

def ista_LASSO(x0,A,b,clambda,iters = 100):
    r""" Lasso Problem solver using ISTA Method:

    min 1/2 \|Ax-b\|_2^2 + clambda \|x\|_1

    Parameters
    ----------
    A : numpy array
        Matrix multiplying x variable
    b : numpy array
        Vector used for the linear fit
    x0 : numpy array
        Initial iteration point
    clambda : float
        The non-negative regularization parameter
    iters : int
        Number of iterations allowed

    """
    print("LASSO solver using ISTA Method")

    start_time = timeit.default_timer()

    x = x0
    L = linalg.norm(A) ** 2  #Lipschitz constant
    vallog = np.zeros(iters)

    for i in range(iters):
        x = operators.soft_thresholding(clambda/L, x + np.dot(A.T,b-A.dot(x))/L)
        vallog[i] = 0.5 * linalg.norm(A.dot(x)-b) ** 2 + clambda * linalg.norm(x,1)

    print("Finished ISTA LASSO solver in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)

def fista_LASSO(x0,A,b,clambda,iters=100):
    r""" Lasso Problem solver using ISTA Method:

    min 1/2 \|Ax-b\|_2^2 + clambda \|x\|_1

    Parameters
    ----------
    A : numpy array
        Matrix multiplying x variable
    b : numpy array
        Vector used for the linear fit
    x0 : numpy array
        Initial iteration point
    clambda : float
        The non-negative regularization parameter
    iters : int
        Number of iterations allowed

    """
    print("LASSO solver using ISTA Method")

    start_time = timeit.default_timer()

    x = x0
    z = x.copy()
    t = 1
    L = linalg.norm(A) ** 2 #Lipschitz constant
    vallog = np.zeros(iters)

    for i in range(iters):
        x_old = x.copy()
        z = z + A.T.dot(b-A.dot(z)) / L
        x = operators.soft_thresholding(clambda/L, z)
        t0 = t
        t = (1 + np.sqrt(1+4*t**2))/2
        z = x + ((t0 -1)/t)*(x-x_old)
        vallog[i] = 0.5 * linalg.norm(A.dot(x)-b) ** 2 + clambda * linalg.norm(x,1)

    print("Finished ISTA LASSO solver in %d iterations and %f sec"%(iters,timeit.default_timer()-start_time))

    return (x,vallog)
