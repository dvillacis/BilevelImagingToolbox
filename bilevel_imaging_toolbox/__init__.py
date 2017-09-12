import numpy as np
#from _bilevel_toolbox import ffi, lib # Connection to c++ or cuda implementation (NOT IMPLEMENTED)
from bilevel_imaging_toolbox import solvers

# The maximum number of returned info parameters
_N_INFO = 3

def _call(fn, *args):
    args_m = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            args_m.append(ffi.cast("double *", arg.ctypes.data))
        else:
            args_m.append(arg)
    print(fn)
    fn(*args_m)
    return fn(*args_m)

def force_float_scalar(x):
    r"""Forces an scalar value into float format

    Parameters
    ----------
    x: scalar value to check

    Returns
    -------
    float
        Float representation of the provided value
    """
    if not isinstance(x, float):
        return float
    else:
        return x


def force_float_matrix(x):
    r"""Forces a numpy matrix into float format

    Parameters
    ----------
    x: numpy array
        matrix to check

    Returns
    -------
    numpy array
        Float representation of the provided matrix
    """
    #Check if input is a numpy array
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except Exception:
            raise TypeError("input must be a numpy matrix or a compatible object")
    # Enforce float type
    if x.dtype != np.dtype('float64'):
        return x.astype('float')
    else:
        return x

def rof_1d(x, w, method='chambolle-pock', sigma=0.05):
    r"""1D proximal operator for the Rudin-Osher-Fatemi (ROF) model.

    Specifically, this optimizes the following problem:

    .. math::

    \mathrm{min}_y \frac{1}{2} \|x-y\|_2^2 + w \sum_i |y_i - y_{i+1}|.

    Parameters
    ----------
    x : numpy array
        The signal we are approximating.
    w : float
        The non-negative weight in the optimization problem.
    method : str
        The algorithm to be used, one of:

        * ``'forward-backward'`` - Forward-Backward Splitting
        * ``'douglas-rachford'`` - Douglas-Rachford Splitting
        * ``'admm'``             - Alternating Direction Method of Multipliers
        * ``'chambolle-pock'``   - Chambolle-Pock Method
        * ``'oesom'``            - Second-Order Orthant Based Method

    sigma : float
        Tolerance for sufficient descent (used only if ``method='pn'``)
    """
    METHODS = {
        'forward-backward' : _forward_backward_ROF,
        'douglas-rachford' : _douglas_rachford_ROF,
        'admm' : _admm_ROF,
        'chambolle-pock' : _chambolle_pock_ROF,
        'oesom' : _oesom_ROF
    }
    assert method in METHODS
    assert w >= 0
    w = force_float_scalar(w)
    x = force_float_matrix(x)
    y = np.zeros(np.size(x))
    METHODS[method](x, w, y, sigma=sigma)
    return y

def _forward_backward_ROF(x,w,y, **kwargs):
    """
    Forward Backward Splitting for the 1D ROF model
    """
    #_call(lib.forwardBackward_ROF, x, np.size(x), w, y) # Link to c++ or cuda implementation
    solvers.forward_backward_ROF(x,w,y)

def _douglas_rachford_ROF(x,w,y, **kwargs):
    #_call(lib.douglasRachford_ROF, x, np.size(x), w, y)
    return 1

def _admm_ROF(x,w,y, **kwargs):
    #_call(lib.admm_ROF, x, np.size(x), w, y)
    return 1

def _chambolle_pock_ROF(x,w,y, **kwargs):
    #_call(lib.chambollePock_ROF, x, np.size(x), w, y)
    return 1

def _oesom_ROF(x,w,y, **kwargs):
    #_call(lib.oesom_ROF, x, np.size(x), w, y)
    return 1

def _PN_ROF(x,w,y, **kwargs):
    """ Proximal Newton Method for the ROF (TV+L2) model """
    #info = np.zeros(_N_INFO) # Holds [num iterations, gap]
    #_call(lib.PN_ROF, x, w, y, info, np.size(x), kwargs['sigma'], ffi.NULL)
    return 1
