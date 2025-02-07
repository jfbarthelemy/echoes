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
#         href: special_tensors.py
#       - text: Jupyter notebook
#         icon: file-code
#         href: special_tensors.ipynb
# ---
#
# # Special tensors {#sec-special_tensors}
#
# ::: {.callout-important icon=false} 
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial presents the matrix representation of isotropic tensors of second and fourth orders as well as Walpole tensors useful for transverse isotropy. It strongly relies on conventions of tensor algebra introduced in @sec-tensor_algebra especially in terms of products and contractions.
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

np.set_printoptions(precision=6, suppress=True)
# to display only 6 significant digits of array components
# -

# :::
#
# ## Second-order identity
#
# The second-order identity $\uu{1}=\delta_{ij}\ve{i}\otimes\ve{j}$ is given in Kelvin-Mandel notation (i.e. vector of $\R^6$) by the constant vector `Id2`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

δ = Id2
print("𝟏 (Kelvin-Mandel notation) =\n",δ)
print("𝟏 =\n",invKM(δ))
# -

# ## Fourth-order isotropic tensors
#
# As detailed in @sec-ISO, the fourth-order identity tensor is
#
# $$
# \uuuu{I}=\uu{1}\sboxtimes\uu{1}=
# \frac{\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}}{2}\,\ve{i}\otimes\ve{j}\otimes\ve{k}\otimes\ve{l}
# $$
#
# and the projectors
#
# $$
# \uuuu{J}=\frac{1}{3}\uu{1}\otimes\uu{1}
# \quad \textrm{and} \quad
# \uuuu{K}=\uuuu{I}-\uuuu{J}=\uu{1}\sboxtimes\uu{1}-\frac{1}{3}\uu{1}\otimes\uu{1}
# $$
#
# which are provided in `echoes` by `Id4`, `J4` and `K4`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

for T in [Id4, J4, K4]:
    print(T)
# -

# ## Walpole bases
#
# The Walpole bases are useful to decompose transversely isotropic fourth-order tensors. They are presented in @sec-TI. The Kelvin-Mandel representation of the $i^\textrm{th}$ Walpole tensor oriented along an axis $\n$ is constructed by `W(i,n=e₃)` ($i \in \{0,..,5\}$ and the normal is by default oriented along the third axis). The symmetrized version is provided by `WS(i,n=e₃)` ($i \in \{0,..,4\}$).
#
# ::: {.callout-warning} 
#
# ## Warning
#
# Note again the shift in indices between the Python convention starting at 0 and the tensors presented in @sec-TI.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

for i in range(6):
    print("𝕎"+str(i+1)+" =\n",W(i))

for i in range(5):
    print("𝕎ˢ"+str(i+1)+" =\n",WS(i))
# -

# $\,$
