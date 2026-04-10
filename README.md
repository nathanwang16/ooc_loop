# ooc_loop
The engineering research project for automation processes of organ on chip system optimization and production

## Revision History

- **v0.1.0** — Scaffolded full project file structure reflecting all 5 pipeline phases and module dependencies (geometry, cfd, config, optimization, analysis, validation). Added OpenFOAM template case, default YAML config, entry-point scripts, and test stubs.
- **v0.2.0** — Module 1.1 complete: OpenFOAM solver verification. Implemented Poiseuille flow analytical solution, blockMeshDict generator, OpenFOAM field parser, automated verification script, and mesh convergence study. Verified <2% error at all refinement levels. Added collaboration tooling (environment.yml, pyproject.toml, Makefile, conftest.py).
- **v0.3.0** — Module 3.1 implemented: completed single-configuration BO runner and 8-configuration orchestrator with constrained objective handling, Sobol initialization, GP fitting, and run-state serialization. Added concrete constraint formulation tests in `tests/test_optimization.py`.
- **v0.4.0** — Implemented pillar-case geometry/meshing and analysis utilities: refactored CadQuery generator with validation and pillar-obstacle STL export, upgraded CFD setup for robust inlet/patch injection and pillar meshing workflow, and completed M3.2 plotting/report scripts (`convergence`, `comparison`, `wss_contours`, `run_analysis.py`) with smoke-tested figure generation.
- **v0.5.0** — Hardened pillar meshing quality handling: removed degraded blockMesh fallback, added per-case meshing diagnostics/log files and checkMesh summaries, exposed snappy/mesh quality thresholds in config, and enforced BO penalization plus infeasible filtering for low-quality meshes (`mesh_ok=False`).
