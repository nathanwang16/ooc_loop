"""
OoC-Optimizer: Automated pipeline for organ-on-chip microfluidic device
design, simulation, and optimization.

Pipeline stages:
    1. geometry  — Parametric CadQuery geometry generation (fluid domain + mold STL)
    2. cfd       — OpenFOAM meshing, solving, and metric extraction
    3. optimization — Bayesian optimization loop (BoTorch)
    4. analysis  — Results visualization and comparison plots
    5. validation — 3D CFD and experimental data analysis
"""

__version__ = "0.6.0"
