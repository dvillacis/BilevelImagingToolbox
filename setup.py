from setuptools import setup


setup(
    name="bilevel_imaging_toolbox",
    version="0.0.1",
    description="Bilevel Methods for Imaging Toolbox",
    long_description="This library provides fast implementations of popular methods used for imaging tasks such as Image Denoising, Image Inpainting, Image Segmentation, etc. And provides a learning framework using bilevel strategies to optimize such tasks.",
    packages=['bilevel_imaging_toolbox'],
    install_requires=[
        'numpy>=1.6.2',
        #'cffi>=1.0.0',
    ],
    #setup_requires=[
    #    'cffi>=1.0.0',
    #],
    #package_data={
    #    'bilevel_imaging_toolbox': ['src/demos/*']
    #},
    #cffi_modules=['bilevel_imaging_toolbox/bilevel_imaging_toolbox_build.py:ffi'],
    author="David Villacis",
    author_email="david.villacis01@epn.edu.ec",
    url='https://github.com/dvillacis/BilevelImagingToolbox',
    license='BSD',
    classifiers=[
        'Development Status :: 1 - Development/Alfa',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='total variation trust region bilevel optimization image processing machine learning',
    test_suite="nose.collector",
)
