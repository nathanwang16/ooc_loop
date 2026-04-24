---
name: Bug report
about: Report something that does not work as expected
title: "[bug] <short description>"
labels: bug
assignees: ""
---

## What happened

<!-- A clear and concise description of the bug. -->

## What I expected

<!-- What you expected to happen. -->

## Minimal reproduction

```bash
# The exact command(s) you ran:
tumor-chip optimize --config configs/default_config.yaml
```

Relevant config (redact anything sensitive):

```yaml
# configs/my_config.yaml
```

## Environment

- Operating system:
- Python version: <!-- python --version -->
- Package version: <!-- tumor-chip version -->
- OpenFOAM version: <!-- simpleFoam -help | head -n 3 -->
- Installation method: <!-- pip / docker / conda / from-source -->

## Logs and tracebacks

<!--
If CFD was involved, please attach the relevant log files, e.g.:
  data/cases/<case_name>/simpleFoam.log
  data/cases/<case_name>/scalarTransportFoam.log
  data/cases/<case_name>/blockMesh.log
  data/cases/<case_name>/logs/checkMesh.log

If it was a Python exception, paste the full traceback below.
-->

```
<paste here>
```

## Additional context

<!-- Anything else that might help us reproduce or diagnose. -->
