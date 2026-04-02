# Tips & Bug Notes

- **OpenFOAM macOS**: `codedFixedValue` fails on DMG-mounted OpenFOAM (macOS .app bundle) because the dynamic code compiler can't write to the mounted volume. Use `fixedValue` with uniform inlet instead; the flow develops within L_dev ≈ 0.2 mm on a 10 mm channel.
- **OpenFOAM macOS install**: Use `brew tap gerlero/openfoam && brew install gerlero/openfoam/openfoam@2406`. Commands are accessed via the `openfoam2406` wrapper (e.g. `openfoam2406 -c 'simpleFoam -case ...'`).
- **turbulenceProperties**: `simpleFoam` requires `constant/turbulenceProperties` even for laminar flow — set `simulationType laminar;`.
- **numpy 2.x**: `np.trapz` removed; use `np.trapezoid`.
