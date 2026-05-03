# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: C:\Python\share\jupyter\kernels\python3
# ---

# ---
# format:
#   html:
#     code-links:
#       - text: Python script
#         icon: file-code
#         href: rve.py
#       - text: Jupyter notebook
#         icon: journal-code
#         href: rve.ipynb
# ---
#
# # Representative Volume Element {#sec-rve}
#
# ::: {.callout-important icon=false}
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial shows how to build a Representative Volume Element (RVE) in *Echoes* by assembling phases (ellipsoidal inclusions, n-layer spheres) with their volume fractions and properties.
#
# :::
#
# ::: {.callout-tip icon=false collapse=true}
#
# ## {{< iconify ix import >}} Imports

# +
#| error: false
#| warning: false
#| code-fold: false
#| code-summary: Code for library imports
#| include: true

import numpy as np
from echoes import *
import math

np.set_printoptions(precision=8, suppress=True)
# -

# :::
#
# ## Construction of a `rve` {#sec-rve-construction}
#
# In *Echoes* a `rve` object inherits from the Python `dict` (dictionary) with string keys and phase values. A *phase* is either an `ellipsoid` (see @sec-ellipsoid), a `sphere_nlayers` (see @sec-sphere_nlayers), or a `crack` (see @sec-crack-object).
#
# ::: {#fig-rve}
#
# ![](../img/ver_ell_spn.png){width=30%}
#
# Representative Volume Element
# :::
#
# A `rve` is defined with optional keyword-arguments:
#
# ```
# myrve = rve(matrix=phasename, prop={"C": C, ...})
# ```
#
# where `phasename` is a string referring to the name of one of the phases and the `prop` argument is a dictionary of tensors indexed by property names. The `matrix` keyword designates:
#
# - the phase playing the role of the **matrix** in matrix/inclusion schemes (e.g. Mori-Tanaka),
# - the phase used to **initialize** the fixed-point algorithm in the self-consistent scheme (unless `prop` is provided, in which case that property is used as initial value).
#
# ## Defining phases
#
# The `rve` is completed by defining phases:
#
# ```
# myrve["PHASE1"] = ellipsoid(shape=..., prop=..., fraction=f1, symmetrize=[ISO])
# myrve["PHASE2"] = ellipsoid(shape=..., prop=..., fraction=f2)
# myrve["PHASE3"] = sphere_nlayers(..., fraction=f3)
# myrve["CRACK"]  = crack(shape=spheroidal(ω), density=ε, prop={"C": tZ4})
# ```
#
# ::: {.callout-note}
#
# ## Notes
#
# - If the `matrix` keyword was not used in the definition of the `rve`, the first phase plays the role of the matrix by default if necessary.
# - The `fraction` keyword denotes the volume fraction within the `rve`. The sum of all fractions should be equal to 1. If it is not the case, the volume fraction of the matrix is adjusted.
# - The `symmetrize=[ISO]` argument "isotropifies" the directions of the ellipsoid: although the ellipsoid is defined with a specified direction, the phase is considered to represent an isotropic distribution of orientations. This argument should be used only if the reference medium of the homogenization scheme is itself isotropic.
# - For `crack` phases the `fraction` keyword is **not used**. Instead, the **Budiansky–O'Connell crack density** $\varepsilon = n\,a^3$ (see @sec-crack-object) is set via the `density` keyword. The volume fraction of a crack family vanishes as the aspect ratio $\omega \to 0$, so density is the relevant parameter.
#
# :::
#
# ## Modifying phases
#
# Each phase can be modified after construction:
#
# ```
# myrve["PHASE1"].fraction = ...
# myrve["PHASE1"].shape = ...
# myrve["PHASE1"].set_prop("C", C)
# myrve["PHASE1"].symmetrize(ISO)
# myrve["PHASE1"].set_param_eshelby(algo=NUMINT, epsroots=1.e-4,
#                                    epsabs=1.e-4, epsrel=1.e-4, maxnb=100000)
# ```
#
# ::: {.callout-tip}
#
# ## Tip
#
# `set_param_eshelby` can also be applied at the level of the `rve`: `myrve.set_param_eshelby(...)`. In this case the parameters apply to all ellipsoidal phases.
#
# :::
#
# $\,$
