import numpy as np

class Operator():
    def __init__(self):
        self.val = None
        self.conj = None
        self.bound = None
        self.lowerbound = None

def make_finite_differences_operator(dim, method, weight):
    r""" Create the finite differences operator

    Parameters
    ----------
    dim : int
        The dimension of the input signal

    method : string
        The parameters that describe which type of dicretization will be performed, it contains two letters.
        The first letter specifies the type of discretization technique:

        * ``'f'`` - forward finite differences
        * ``'b'`` - backward finite differences
        * ``'c'`` - centered finite differences

        The second letter specifies the boundary conditions:

        * ``'n'`` - Newmann boundary conditions

    weight : float
        This is a float number that will be multiplied element-wise for all the operator.

    Returns
    -------
    op : Operator
        Operator containing the differences matrix and its conjugate matrix (tensor)

    """
    METHODS = {
        'fn' : _forward_differences_newmann,
        'bn' : _backward_differences_newmann,
        'cn' : _centered_differences_newmann,
        '*fn' : _forward_differences_newmann_conj,
        '*bn' : _backward_differences_newmann_conj,
        '*cn' : _centered_differences_newmann_conj
    }
    assert method in METHODS
    assert weight >= 0

    op = Operator()
    op.val = lambda x : METHODS[method](x)
    op.conj = lambda y : METHODS['*'+method](y)
    op.bound = weight*np.sqrt(8)
    op.lowerbound = 0
    return op

def _forward_differences_newmann(x):
    res = np.zeros((x.shape[0],x.shape[1],2))
    res[:-1,:,0] = x[1:,:] - x[:-1,:]
    res[:,:-1,1] = x[:,1:] - x[:,:-1]
    return res

def _forward_differences_newmann_conj(y):
    res = np.zeros((y.shape[0],y.shape[1]))
    res[:-1,:] = -y[:-1,:,0]
    res[1:,:] = res[1:,:] + y[:-1,:,0]
    res[:,:-1] = res[:,:-1] - y[:,:-1,1]
    res[:,1:] = res[:,1:] + y[:,:-1,1]
    return res

def _backward_differences_newmann(x):
    res = np.zeros((x.shape[0],x.shape[1],2))
    res[1:,:,0] = x[1:,:] - x[:-1,:]
    res[:,1:,1] = x[:,1:] - x[:,:-1]
    return res

def _backward_differences_newmann_conj(y):
    res = np.zeros((y.shape[0],y.shape[1]))
    res[1:,:] = y[1:,:,0]
    res[:-1,:] = res[:-1,:] - y[1:,:,0]
    res[:,1:] = res[:,1:] + y[:,1:,1]
    res[:,:-1] = res[:,:-1] - y[:,1:,1]
    return res

def _centered_differences_newmann(x):
    res = np.zeros((x.shape[0],x.shape[1],2))
    res[1:-1,:,0] = (x[2:,:]-x[:-2,:])/2
    res[:,1:-1,1] = (x[:,2:]-x[:,:-2])/2
    return res

def _centered_differences_newmann_conj(y):
    res = np.zeros((y.shape[0],y.shape[1]))
    res[2:,:] = y[1:-1,:,0]/2
    res[:-2,:] = res[:-2,:] - y[1:-1,:,0]/2
    res[:,2:] = res[:,2:] + y[:,1:-1,1]/2
    res[:,:-2] = res[:, :-2] - y[:,1:-1,1]/2
    return res

def prox_project(clambda, z):
    r""" Projection to the clambda-ball

    Parameters
    ----------
    clambda : float
        Radius of the ball
    z : numpy array
        data to be projected

    """
    nrm = np.sqrt(z[:,:,0]**2 + z[:,:,1]**2)
    fact = np.minimum(clambda, nrm)
    fact = np.divide(fact,nrm, out=np.zeros_like(fact), where=nrm!=0)

    y = np.zeros(z.shape)
    y[:,:,0] = np.multiply(z[:,:,0],fact)
    y[:,:,1] = np.multiply(z[:,:,1],fact)
    return y

def soft_thresholding(clambda, z):
    r""" Soft-Thresholding Operator

    Parameters
    ----------
    clambda : float
        Regularization Parameter
    z : numpy array
        data

    """
    return np.sign(z)*np.maximum(np.abs(z)-clambda,0)
