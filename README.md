# ooc_loop
The engineering research project for automation processes of organ on chip system optimization and production

## Revision History

- **v0.1.0** — Scaffolded full project file structure reflecting all 5 pipeline phases and module dependencies (geometry, cfd, config, optimization, analysis, validation). Added OpenFOAM template case, default YAML config, entry-point scripts, and test stubs.
- **v0.2.0** — Module 1.1 complete: OpenFOAM solver verification. Implemented Poiseuille flow analytical solution, blockMeshDict generator, OpenFOAM field parser, automated verification script, and mesh convergence study. Verified <2% error at all refinement levels. Added collaboration tooling (environment.yml, pyproject.toml, Makefile, conftest.py).
