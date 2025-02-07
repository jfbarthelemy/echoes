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
#         href: rot_matrices.py
#       - text: Jupyter notebook
#         icon: file-code
#         href: rot_matrices.ipynb
# ---
#
# # Rotation matrices {#sec-rot_tensors}
#
# ::: {.callout-important icon=false} 
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial presents the construction of rotation matrices in $\R^3$ in the convention proposed in @sec-rottens for Euler angles.
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
# Given the Euler angles `θ, ϕ, ψ` defined in @sec-rottens and more particularly in @fig-eulerangles, the rotation matrix (@eq-rot3) recalled here
# $$
# \small
# \mat{R}=
#    \left(
#    \begin{array}{ccc}
#    c_\theta  c_\psi  c_\phi - s_\psi  s_\phi & - c_\theta  c_\phi  s_\psi - c_\psi  s_\phi & c_\phi  s_\theta \\
#    c_\theta  c_\psi  s_\phi + c_\phi  s_\psi & - c_\theta  s_\psi  s_\phi + c_\psi  c_\phi & s_\theta  s_\phi \\
#    - c_\psi  s_\theta & s_\theta  s_\psi & c_\theta \\
#    \end{array}
#    \right) 
# $$
#
# can be built with `rot3(θ, ϕ, ψ)`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

π = math.pi
θ, ϕ, ψ = π/3, π/4, π/5
R = rot3(θ, ϕ, ψ)
print("R =\n",R)
# -

# The vectors of the spherical basis correspond to the column of the rotation matrix for $\psi=0$. They can individually be obtained by the function `es(i, θ, ϕ)`
#
# ::: {.callout-warning} 
#
# ## Warning
#
# Python numbering starts at 0 so `i` takes the values 0, 1 and 2.
#
# Besides vectors of the spherical basis are ordered as $\ve{\theta}$, $\ve{\phi}$, $\ve{r}$.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

θ, ϕ = π/8, π/5
R = rot3(θ, ϕ)
print("R =\n",R)
for i in range(3): print(f"es({i}, θ, ϕ) = ", es(i, θ, ϕ), f" → e̱{['θ','ϕ','r'][i]}")
# -

# As shown in @sec-rottens the rotation matrix applying on fourth-order tensors in Kelvin-Mandel notation can be deduced from the $3×3$ rotation matrix from (@eq-rot6)
# $$
# \scriptsize
#    \left(
# \begin{array}{cccccc}
# R_{1 1}^{2} & R_{1 2}^{2} & R_{1 3}^{2} & \sqrt{2}  R_{1 2}  R_{1 3} & \sqrt{2}  R_{1 1}  R_{1 3} & \sqrt{2}  R_{1 1}  R_{1 2} \\
# R_{2 1}^{2} & R_{2 2}^{2} & R_{2 3}^{2} & \sqrt{2}  R_{2 2}  R_{2 3} & \sqrt{2}  R_{2 1}  R_{2 3} & \sqrt{2}  R_{2 1}  R_{2 2} \\
# R_{3 1}^{2} & R_{3 2}^{2} & R_{3 3}^{2} & \sqrt{2}  R_{3 2}  R_{3 3} & \sqrt{2}  R_{3 1}  R_{3 3} & \sqrt{2}  R_{3 1}  R_{3 2} \\
# \sqrt{2}  R_{2 1}  R_{3 1} & \sqrt{2}  R_{2 2}  R_{3 2} & \sqrt{2}  R_{2 3}  R_{3 3} & R_{2 2}  R_{3 3} + R_{2 3}  R_{3 2} & R_{2 1}  R_{3 3} + R_{2 3}  R_{3 1} & R_{2 1}  R_{3 2} + R_{2 2}  R_{3 1} \\
# \sqrt{2}  R_{1 1}  R_{3 1} & \sqrt{2}  R_{1 2}  R_{3 2} & \sqrt{2}  R_{1 3}  R_{3 3} & R_{1 2}  R_{3 3} + R_{1 3}  R_{3 2} & R_{1 1}  R_{3 3} + R_{1 3}  R_{3 1} & R_{1 1}  R_{3 2} + R_{1 2}  R_{3 1} \\
# \sqrt{2}  R_{1 1}  R_{2 1} & \sqrt{2}  R_{1 2}  R_{2 2} & \sqrt{2}  R_{1 3}  R_{2 3} & R_{1 2}  R_{2 3} + R_{1 3}  R_{2 2} & R_{1 1}  R_{2 3} + R_{1 3}  R_{2 1} & R_{1 1}  R_{2 2} + R_{1 2}  R_{2 1} \\
# \end{array}
# \right)
# $$
#
# It can be directly obtained by `rot6(θ, ϕ, ψ)`. 

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

θ, ϕ, ψ = π/3, π/4, π/5
R = rot3(θ, ϕ, ψ)
sboxtimes=lambda a,b:0.5*(np.einsum('ik,jl',a,b)+np.einsum('il,jk',a,b))
print("R⊠ˢR =\n",KM(sboxtimes(R,R)))

ℝ = rot6(θ, ϕ, ψ)
print("ℝ =\n",ℝ)
# -

# $\,$
