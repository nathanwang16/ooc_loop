"""Package setup for ooc-optimizer."""

from setuptools import setup, find_packages

setup(
    name="ooc-optimizer",
    version="0.1.0",
    description=(
        "Automated pipeline for organ-on-chip microfluidic device design, "
        "simulation, and Bayesian optimization"
    ),
    packages=find_packages(exclude=["tests", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "cadquery>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "botorch>=0.11.0",
        "gpytorch>=1.11.0",
        "torch>=2.1.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
