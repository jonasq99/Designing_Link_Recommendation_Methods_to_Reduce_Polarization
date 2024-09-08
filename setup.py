import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        "icm_diffusion_optimized",
        ["icm_diffusion_optimized.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="random_walks",
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args=["build_ext", "--inplace"],
)
