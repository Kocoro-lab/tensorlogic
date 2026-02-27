from setuptools import setup, find_packages

setup(
    name="tensorlogic",
    version="0.1.0",
    description="A unified programming language for AI combining neural and symbolic reasoning",
    author="Tensor Logic Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tensorly>=0.8.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "examples": ["matplotlib>=3.7.0", "scikit-learn>=1.2.0"],
        "dev": ["pytest>=7.4.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
