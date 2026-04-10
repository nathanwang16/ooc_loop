# Tips & Bug Notes

- **OpenFOAM macOS**: `codedFixedValue` fails on DMG-mounted OpenFOAM (macOS .app bundle) because the dynamic code compiler can't write to the mounted volume. Use `fixedValue` with uniform inlet instead; the flow develops within L_dev ≈ 0.2 mm on a 10 mm channel.
- **OpenFOAM macOS install**: Use `brew tap gerlero/openfoam && brew install gerlero/openfoam/openfoam@2406`. Commands are accessed via the `openfoam2406` wrapper (e.g. `openfoam2406 -c 'simpleFoam -case ...'`).
- **turbulenceProperties**: `simpleFoam` requires `constant/turbulenceProperties` even for laminar flow — set `simulationType laminar;`.
- **numpy 2.x**: `np.trapz` removed; use `np.trapezoid`.
- **CAD dependency coupling**: Importing `ooc_optimizer.cfd` transitively imports geometry (`cadquery`). BO/CFD tests that import `ooc_optimizer.cfd.*` require `cadquery` installed even if geometry generation is not exercised.
- **macOS inplace edits**: Avoid `sed -i` for OpenFOAM case file updates on macOS. Use Python string replacement to patch `0/U` values safely.
- **Pillar meshing quality policy**: `snappyHexMesh` can still produce low-quality thin-slab meshes (e.g., concave cells). Pipeline no longer falls back to background `blockMesh`; it keeps the snappy mesh, logs `checkMesh` diagnostics, marks `mesh_ok=False`, and BO applies a penalty objective.
- **Manual STL+CFD check**: `source /Volumes/OpenFOAM-v2406/etc/bashrc` then `conda activate ooc` and run `python scripts/run_single_verification.py` from repo root (`--complex` for 3x6 pillars, H=300 um). Outputs under `data/manual_verification/` or `data/manual_verification_complex/`; see each folder's `VERIFICATION_MANIFEST.md`.
- **ParaView OpenFOAM loading**: Opening an OpenFOAM case directory via `All Files` can trigger the wrong reader (for example `vtkADIOS2CoreImageReader`) and show an empty view. Create a marker file with `touch <case>/<name>.foam`, then open that `.foam` file in ParaView and choose `OpenFOAMReader`/`POpenFOAMReader`.
