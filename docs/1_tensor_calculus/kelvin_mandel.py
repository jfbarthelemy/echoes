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
#         href: kelvin_mandel.py
#       - text: Jupyter notebook
#         icon: file-code
#         href: kelvin_mandel.ipynb
# ---
#
# # Kelvin-Mandel notation {#sec-kelvin_mandel}
#
# ::: {.callout-important icon=false} 
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# Before introducing the specific objects of `echoes` devoted to tensor calculations in isotropic or anisotropic contexts, this tutorial aims at providing the syntax allowing to represent second-order or fourth-order tensors under the form of matrices in the Kelvin-Mandel notation as detailed in @sec-KM.
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
import math, random

np.set_printoptions(precision=8, suppress=True)
# to display only 8 significant digits of array components
# -

# :::
#
#
# A symmetric $3×3$ second-order matrix can be transformed in a vector of $\R^6$ by the function `KM` consistently with (@eq-KM2). The inverse is done by `invKM`.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

α = np.random.rand(3, 3) ; ε = (α+α.T)/2
print("ε =\n",ε)
print("KM(ε) =\n",KM(ε))
assert np.allclose(invKM(KM(ε)), ε), "error"
# -

# Given a $3×3×3×3$ array `c` (of type `numpy.ndarray`) satisfying major and minor symmetries (see @sec-KM),  the corresponding $6×6$ matrix `C` obtained by Kelvin-Mandel transform is calculated by `C = KM(c)`. Conversely, if `C` is a positive definite matrix, `c` is calculated by `c = invKM(C)`.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

A = np.random.rand(6,6)
C = A.T.dot(A) + np.eye(6) # generation of an arbitrary positive definite matrix
c = invKM(C)
print("C =\n",C)
print("c =\n",c)
assert np.allclose(KM(c), C), "error: KM(c) should be equal to C"
# -

# $\,$
