import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        "seed_ima",
        ["src/seed_ima.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="seed_ima",
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args=["build_ext", "--inplace"],
)
