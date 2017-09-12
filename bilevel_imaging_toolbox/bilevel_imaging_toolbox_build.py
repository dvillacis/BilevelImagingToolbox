import os
import os.path
from sys import platform as _platform

from cffi import FFI 
ffi = FFI()

extra_compile_args = []
extra_link_args = []

extra_compile_args.append('-I'+'./src/')
if _platform == 'darwin':
    # if openblas was installed by homerew is present use this for lapacke.h
    if os.path.exists('/usr/local/opt/openblas/include'):
        extra_compile_args.append('-I/usr/local/opt/openblas/include')
else:
    # OSX clang does not (yet) support openmp, so don't add it to compile
    # args
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

sources = [os.path.join('./src/', fname) for fname in (
    'ROFopt.cpp','utils.cpp'
)]

ffi.set_source("_bilevel_toolbox",
   r""" // passed to the real C compiler
        #include <BilevelToolbox.h>
    """,
    sources=sources,
    source_extension='.cpp',
    extra_compile_args=extra_compile_args,
    libraries=['lapack'])   # or a list of libraries to link with
    # (more arguments like setup.py's Extension class:
    # include_dirs=[..], extra_objects=[..], and so on)

ffi.cdef("""
        typedef struct {
            ...;
        } Workspace;

        /* ROF Solvers 1D */
        int PN_ROF(double *y, double lam, double *x, double *info, int n, double sigma, Workspace *ws);
""")

if __name__ == "__main__":
    ffi.compile(verbose=True)

#ffi = FFI()
#ffi.cdef("""
#        typedef struct {
#            ...;
#        } Workspace;
#
#        /* ROF Solvers 1D */
#        void forwardBackward_ROF(double *signal, int n, double lam, double *prox);
#        void douglasRachford_ROF(double *signal, int n, double lam, double *prox);
#        void admm_ROF(double *signal, int n, double lam, double *prox);
#        void chambollePock_ROF(double *signal, int n, double lam, double *prox);
#        void oesom_ROF(double *signal, int n, double lam, double *prox);
#        int PN_ROF(double * signal, int n, double lam, double *x, double *info, double sigma, Workspace *ws);
#""")
#
#sources = [os.path.join('src', fname) for fname in (
#    'ROFopt.cpp'
#)]
#
#extra_compile_args = []
#extra_link_args = []
#if _platform == 'darwin':
#    if os.path.exists('/usr/local/opt/openblas/include'):
#        extra_compile_args.append('-I/usr/local/opt/openblas/include')
#else:
#    extra_compile_args.append('-fopenmp')
#    extra_link_args.append('-fopenmp')
#
#extra_compile_args.append('-I'+os.path.join('src'))
#
#ffi.set_source(
#    '_bilevel_toolbox',
#    """
#    #include "BilevelToolbox.h"
#    """,
#    sources=sources,
#    source_extension='.cpp',
#    extra_compile_args=extra_compile_args,
#    extra_link_args=extra_link_args,
#    libraries=['lapack']
#)
#
#if __name__ == '__main__':
#    ffi.compile(verbose=True)
