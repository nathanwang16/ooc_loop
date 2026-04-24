# Publications

## Main manuscript

> *"Interpretable inverse design of tumor-on-chip chambers: learned
> surrogates reveal which geometric features control drug gradients and by
> how much."* — Lab on a Chip (in preparation).

**TL;DR**: The BO finds a geometry that matches a target concentration
profile; the Sobol / GP-gradient / tolerance-interval layer reports which
geometric parameters actually control the field and how tightly each one
needs to be held during SLA mould printing and PDMS casting.  The software
also ships with a second worked example (`examples/wss_uniformity/`) that
demonstrates the same engine optimising wall shear stress on a different
chamber class.

## JOSS companion paper

A short (~4 pages) JOSS submission covering the software architecture,
reproducibility scaffolding, and extensibility to other scalar inverse-
design problems is drafted in
[`paper/joss/paper.md`](https://github.com/ooc-loop/tumor-chip-design/blob/main/paper/joss/paper.md).
It is co-submitted with the main manuscript.

## How to cite

Use the [`CITATION.cff`](https://github.com/ooc-loop/tumor-chip-design/blob/main/CITATION.cff)
file at the repository root (GitHub renders a *"Cite this repository"*
button automatically from it).  For a specific release, prefer the
Zenodo DOI displayed as a README badge.
