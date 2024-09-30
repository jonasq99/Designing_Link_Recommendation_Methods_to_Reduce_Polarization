import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        "icm_diffusion_optimized",
        ["src/icm_diffusion_optimized.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="icm_diffusion_optimized",
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args=["build_ext", "--inplace"],
)
