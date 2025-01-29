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
#         href: tensor.py
#       - text: Jupyter notebook
#         icon: file-code
#         href: tensor.ipynb
# ---
#
# # The `tensor` object {#sec-tensor}
#
# ::: {.callout-important icon=false} 
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial presents the object `tensor` which is the main structure of `echoes` allowing to represent symmetric second-order or fourth-order tensors both in matrix and synthetic forms and containing information about anisotropy.
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
# ## Definition of the `tensor` object
#
# The `tensor` object is a structure designed to represent **symmetric** second-order or fourth-order tensors. It gathers four members
#
# - **the material symmetry**: `ISO` (isotropic), `TI` (transversely isotropic), 	`ORTHO` (orthotropic) or `ANISO` (anisotropic) *[other material symmetries exist but have not been implemented]*
# - **a condensed vector of parameters** *[as shown below the size of this vector unambiguously defines the order of the tensor]*
# - **a vector of angles (in radians)** *[if required]*
# - **a matrix** *[$3×3$ for a second-order or $6×6$ for a fourth-order tensor]*
#
# ::: {.callout-warning} 
#
# ## Warning
#
# It is important to note that the `tensor` object is designed only to contain or retrieve information about a symmetric second-order or fourth-order tensor (for instance the material symmetry) and not to perform calculation. Although simple calculations such as addition, subtraction, multiplication by a scalar, inversion are available, most of other operations (products, contractions...) have not been implemented. To do so, it is necessary to extract the parameters or matrix and perform operations using usual `numpy` or `scipy` operations (`dot`, `einsum`...) before eventually building a new `tensor`.
#
# :::
#
# If `C` is a python variable of `tensor` type, its members can be accessed by
#
# - **material symmetry**: `C.sym`
# - **parameters**: `C.param` or `C.p`
# - **angles**: `C.angles`
# - **matrix**: `C.array` or `C.a`
#
# ## Second-order tensors
#
# A symmetric second-order `tensor` object is designed by means of its three eigenvalues and the orientation of the eigenbasis through three Euler angles $\theta$, $\phi$ and $\psi$ (see @fig-eulerangles). It is built using one of the following constructors
#
# - `T = tensor(T₁, T₂, T₃, θ=0, ϕ=0, ψ=0)` # default values of the angles are 0
# - `T = tensor([T₁, T₂, T₃], angles=[0, 0, 0])`
#
# ::: {.callout-warning} 
# ## Warning
# - The eigenvalues and eigenvectors are systematically reordered in decreasing order of eigenvalues and angles are recomputed accordingly.
# - The eigenvalues are analyzed in order to characterize the symmetry type between `ISO`, `TI` and `ORTHO`.
# - The Euler angles are not unique since unit eigenvectors are determined up to a multiplicative factor of ±1.
# - Note that the six parameters of the constructor `tensor` should not be confused with the components of the Kelvin-Mandel representation of symmetric second-order tensors (see @sec-KM).
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

T = tensor(2., 1., 3.) ; print(T)

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

π = math.pi
T = tensor(2., 1., 2., π/3, π/4, π/5) ; print(T)

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

T = tensor([2., 1., 3.], angles=[π/3, π/4, π/5]) ; print(T)
# -

# A symmetric second-order `tensor` can also be built from a symmetric $3×3$ matrix. In this case a diagonalization is performed so as to find the `param` vector (the eigenvalues) and the `angles` determining the orientation of the eigenvectors (see @fig-eulerangles)

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

M = np.array([ [1.7668, 0.659068, 0.703171], 
               [0.659068, 2.24232, 0.120771], 
               [0.703171, 0.120771, 1.99088] ])
T = tensor(M) ; print(T)
# -

# ## Isotropic fourth-order tensors (`ISO`)
#
# The special isotropic tensors $\uuuu{I}$, $\uuuu{J}$ and $\uuuu{K}$ are available under the global variables `tId4`, `tJ4` and `tK4`.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

for T in [tId4, tJ4, tK4]:
    print(T)
# -

# Any fourth-order tensor depends on two parameters $\alpha$ and $\beta$ such that $\uuuu{T}=\alpha \uuuu{J} + \beta \uuuu{K}$ and can be built by one of these constructors
#
# - `T = tensor(α, β)`
# - `T = tensor([α, β])`
# - `T = tensor(np.array([α, β]))`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

α, β = 7.2, 6.1
𝕋 = tensor(α, β)
print("𝕋 =\n", 𝕋)
print("𝕋.sym =", 𝕋.sym)
print("𝕋.a =\n", 𝕋.a)
print("𝕋.angles =", 𝕋.angles)
print("𝕋.p =", 𝕋.p)

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

𝕀, 𝕁, 𝕂 = tId4, tJ4, tK4 # just to make it look nice!

𝕋2 = α*𝕁 + β*𝕂
print("𝕋2 =\n", 𝕋2)
assert np.allclose(𝕋.a, 𝕋2.a), "error 𝕋.a and 𝕋2.a should be equal"
assert np.allclose(𝕋.p, 𝕋2.p), "error 𝕋.p and 𝕋2.p should be equal"
# -

# A symmetric fourth-order tensor representing a stiffness tensor can also be built by means of
#
# - *bulk and shear moduli* `stiff_kmu(k, µ)` → $3\,k \uuuu{J} + 2\,\mu \uuuu{K}$
# - *Young modulus and Poisson ratio* `stiff_Enu(E, ν)` → $\frac{E}{1-2\,\nu} \uuuu{J} + \frac{E}{1+\nu} \uuuu{K}$
# - *Lamé moduli* `stiff_lambdamu(λ, μ)` → $3\,\lambda \uuuu{J} + 2\,\mu \uuuu{I}$
#     
# Some converters are also available (see [conversion table](https://en.wikipedia.org/wiki/Lam%C3%A9_parameters))
#
# - `E = E_from_kmu(k, µ)`
# - `ν = nu_from_kmu(k, µ)`
# - `k = k_from_Enu(E, ν)`
# - `µ = mu_from_Enu(E, ν)`
# - `E, ν = Enu_from_kmu(k, µ)`
# - `k, µ = kmu_from_Enu(E, ν)`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

k, µ = 72.5, 32.7
ℂ = stiff_kmu(k, µ)
print("ℂ =\n", ℂ)
print(f"E = {round(ℂ.E,2)} ; ν = {round(ℂ.nu,2)} ; λ = {round(ℂ.lamelambda,2)} ; μ = {round(ℂ.mu,2)}")
E, ν = Enu_from_kmu(k, µ)
assert np.allclose([k, μ], [*kmu_from_Enu(E, ν)]), "error kmu_from_Enu(E, ν) should return k, μ"
# -

# ## Symmetric transversely isotropic fourth-order tensors (`TI`)
#
# Such a tensor depends on 5 parameters and 2 angles $\theta$ and $\phi$ defining the orientation of the normal to the isotropy plane (see @fig-eulerangles).
#
# The decomposition on the symmetric Walpole tensors writes by means of five parameters $(t_i)_{i=1,\ldots,5}$. In the frame where $\uv{n}=\ve{3}$ the matrix is given by
#
# $$
# \uuuu{T}=\sum_{i=1}^5 t_i \uuuu{W}^s_i
# \mapsto
# \left(
# \begin{array}{cccccc}
# \frac{t_2 + t_4}{2} & \frac{t_2 - t_4}{2} & \frac{\sqrt{2} \, t_3}{2} & 0 & 0 & 0 \\
# \frac{t_2 - t_4}{2} & \frac{t_2 + t_4}{2} & \frac{\sqrt{2} \, t_3}{2} & 0 & 0 & 0 \\
# \frac{\sqrt{2} \, t_3}{2} & \frac{\sqrt{2} \, t_3}{2} & t_1 & 0 & 0 & 0 \\
# 0 & 0 & 0 & t_5 & 0 & 0 \\
# 0 & 0 & 0 & 0 & t_5 & 0 \\
# 0 & 0 & 0 & 0 & 0 & t_4 \\
# \end{array}
# \right)
# $$
#
# A fourth-order transversely isotropic tensor can be built by one of these constructors
#
# - `T = tensor(t1, t2, t3, t4, t5, θ=0, ϕ=0)`
# - `T = tensor(param, angles=[0, 0])`
#
# where `param` is a `list` or a `numpy.ndarray` of 5 items and `angles` is a `list` or a `numpy.ndarray` of 2 items `θ, ϕ` (if `angles` is omitted the angles are null and the axis is oriented along $\ve{3}$)

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

param = [3., 4., math.sqrt(2), 2., 1.]
angles = [π/3, π/4]
print(tensor(param))
print(tensor(param, angles))
# -

# It is also possible to build a `TI` tensor by means of 5 $C_{ijkl}$ moduli:
#
# $$
# (T_{1 1 1 1}, T_{1 1 2 2}, T_{1 1 3 3}, T_{3 3 3 3}, T_{2 3 2 3})
# \mapsto
# \left(
# \begin{array}{cccccc}
# T_{1 1 1 1} & T_{1 1 2 2} & T_{1 1 3 3} & 0 & 0 & 0 \\
# T_{1 1 2 2} & T_{1 1 1 1} & T_{1 1 3 3} & 0 & 0 & 0 \\
# T_{1 1 3 3} & T_{1 1 3 3} & T_{3 3 3 3} & 0 & 0 & 0 \\
# 0 & 0 & 0 & 2 \, T_{2 3 2 3} & 0 & 0 \\
# 0 & 0 & 0 & 0 & 2 \, T_{2 3 2 3} & 0 \\
# 0 & 0 & 0 & 0 & 0 & T_{1 1 1 1} - T_{1 1 2 2} \\
# \end{array}
# \right)
# $$
#
# by
#
# `T = stiff_TI(T₁₁₁₁, T₁₁₂₂, T₁₁₃₃, T₃₃₃₃, T₂₃₂₃, θ=0, ϕ=0)`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

T = stiff_TI(6.2, 5.6, 1.3, 2.1, 2.3)
assert all(v > 0 for v in np.linalg.eigvals(T.a)), "one eigenvalue is negative"
# check that T is positive definite
print(T)
print(stiff_TI(6.2, 5.6, 1.3, 2.1, 2.3, π/3, π/4))
# -

# A engineering construction based on directional moduli and Poisson ratio is also available:
#
# $$
# (E_1, E_3, \nu_{1 2}, \nu_{3 1}, G_{3 1})
# \mapsto
# \left(
# \begin{array}{cccccc}
# \frac{1}{E_1} & \frac{ - \nu_{1 2}}{E_1} & \frac{ - \nu_{3 1}}{E_3} & 0 & 0 & 0 \\
# \frac{ - \nu_{1 2}}{E_1} & \frac{1}{E_1} & \frac{ - \nu_{3 1}}{E_3} & 0 & 0 & 0 \\
# \frac{ - \nu_{3 1}}{E_3} & \frac{ - \nu_{3 1}}{E_3} & \frac{1}{E_3} & 0 & 0 & 0 \\
# 0 & 0 & 0 & \frac{1}{2 \, G_{3 1}} & 0 & 0 \\
# 0 & 0 & 0 & 0 & \frac{1}{2 \, G_{3 1}} & 0 \\
# 0 & 0 & 0 & 0 & 0 & \frac{1 + \nu_{1 2}}{E_1} \\
# \end{array}
# \right)
# $$
#
# by
#
# `S = comp_TI(E₁, E₃, ν₁₂, ν₃₁, G₃₁, θ=0, ϕ=0)`

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

S = comp_TI(1., 4., 0.2, 0.1, 2.)
print(S)
print(inv(S))
print(comp_TI(1., 4., 0.2, 0.1, 2., π/3, π/4))
# -

# ## Symmetric orthotropic fourth-order tensors (`ORTHO`)
#
# Such a tensor depends on 9 parameters and 3 angles $\theta$, $\phi$ and $\psi$ defining the orientation of an orthonormal frame (see @fig-eulerangles).
#
# The convention here relies on the components $T_{1 1 1 1}, T_{1 1 2 2}, T_{1 1 3 3}, T_{2 2 2 2}, T_{2 2 3 3}, T_{3 3 3 3}, T_{2 3 2 3}, T_{3 1 3 1}, T_{1 2 1 2}$ as inputs. In the canonical frame the matrix is given by
#
# $$
# \small
# (T_{1 1 1 1}, T_{1 1 2 2}, T_{1 1 3 3}, T_{2 2 2 2}, T_{2 2 3 3}, T_{3 3 3 3}, T_{2 3 2 3}, T_{3 1 3 1}, T_{1 2 1 2})
# \mapsto
# \left(
# \begin{array}{cccccc}
# T_{1 1 1 1} & T_{1 1 2 2} & T_{1 1 3 3} & 0 & 0 & 0 \\
# T_{1 1 2 2} & T_{2 2 2 2} & T_{2 2 3 3} & 0 & 0 & 0 \\
# T_{1 1 3 3} & T_{2 2 3 3} & T_{3 3 3 3} & 0 & 0 & 0 \\
# 0 & 0 & 0 & 2 \, T_{2 3 2 3} & 0 & 0 \\
# 0 & 0 & 0 & 0 & 2 \, T_{3 1 3 1} & 0 \\
# 0 & 0 & 0 & 0 & 0 & 2 \, T_{1 2 1 2} \\
# \end{array}
# \right)
# $$
#
# A fourth-order isotropic tensor can be built by one of these constructors
#
# - `T = tensor(T₁₁₁₁, T₁₁₂₂, T₁₁₃₃, T₂₂₂₂, T₂₂₃₃, T₃₃₃₃, T₂₃₂₃, T₃₁₃₁, T₁₂₁₂, θ=0, ϕ=0, ψ=0)`
# - `T = tensor(param, angles=[0, 0, 0])`
#
# where `param` is a `list` or a `numpy.ndarray` of 9 items and `angles` is a `list` or a `numpy.ndarray` of 3 items `θ, ϕ, ψ` (if `angles` is omitted the frame is the canonical one).

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

T = tensor([9.,3.,5.,6.,3.,7.,1.,4.,3.])
assert all(v > 0 for v in np.linalg.eigvals(T.a)), "one eigenvalue is negative" # check that T is positive definite
print(T)
print(tensor([9.,3.,5.,6.,3.,7.,1.,4.,3.], [π/3, π/4, π/5]))
# -

# ## Symmetric anisotropic fourth-order tensors (`ANISO`)
#
# Such a tensor depends on 21 parameters, by convention here the 21 components of the Kelvin-Mandel notation. Components are considered line after line of the upper triangular part of the Kelvin-Mandel matrix
#
# $$
# \begin{array}{cccccc}
#    (T_{11},&T_{12},&T_{13},&T_{14},&T_{15},&T_{16}\\
#                           &T_{22},&T_{23},&T_{24},&T_{25},&T_{26}\\
#                           &            &T_{33},&T_{34},&T_{35},&T_{36}\\
#                           &            &             &T_{44},&T_{45},&T_{46}\\
#                           &            &             &            &T_{55},&T_{56}\\
#                           &            &             &            &            &T_{66})
#    \end{array}
# \mapsto
# \left(
# \begin{array}{cccccc}
#      T_{11}&T_{12}&T_{13}&T_{14}&T_{15}&T_{16}\\
#      T_{12}&T_{22}&T_{23}&T_{24}&T_{25}&T_{26}\\
#      T_{13}&T_{23}&T_{33}&T_{34}&T_{35}&T_{36}\\
#      T_{14}&T_{24}&T_{34}&T_{44}&T_{45}&T_{46}\\
#      T_{15}&T_{25}&T_{35}&T_{45}&T_{55}&T_{56}\\
#      T_{16}&T_{26}&T_{36}&T_{46}&T_{56}&T_{66}
# \end{array}
# \right)
# $$
#
# ::: {.callout-warning} 
#
# ## Warning
#
# It is important to recall here that the parameters $T_{IJ}$ with $1\leq I,J\leq 6$ do not all coincide with the components $T_{ijkl}$. Indeed the convention presented in @sec-KM implies that $T_{11}=T_{1111}$ but $T_{14}=\sqrt{2}\,T_{1123}$ and $T_{44}=2\,T_{2323}$.
#
# :::
#
# Such a tensor is built by
#
# `T = tensor(param)`
#
# where `param` is a `list` or a `numpy.ndarray` of 21 items.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

param = [ 2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089, 2.743, 1.3367, 1.2962, 0.897632, 4.42684, 2.05632, 1.52686, 3.54431, 1.3445, 1.99356 ]
T = tensor(param)
assert all(v > 0 for v in np.linalg.eigvals(T.a)), "one eigenvalue is negative" # check that T is positive definite
print(T)
# -

# ## Construction of a fourth-order `tensor` from a $6×6$ matrix and projections
#
# Given a symmetric positive definite $6×6$ matrix `M` (warning: these properties are not checked on construction), a tensor can be built by one of the function
#
# - `T = tensor(M)`
# - `T = tensor(M, angles=[θ, ϕ])`
#
# By default, this function finds out whether `M` represents an `ISO`, `TI` or `ORTHO` tensor in the canonical frame if `angles` is not precised or in the frame defined by `angles`. If none of these symmetries is suitable, the tensor is considered `ANISO`.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

angles = [π/3, π/4, π/5]
T = tensor([9.,3.,5.,6.,3.,7.,1.,4.,3.], angles)
M = T.a
print("M =\n", M) # the information about ORTHO symmetry has vanished in M
T2 = tensor(M, angles=angles)
print(T2)
# -

# Given a $6×6$ matrix supposed to represent a tensor in the canonical frame, it is possible to find the closest tensor (in the sense of least-squares relatively to the euclidean matrix norm) among a family of given material symmetry. The syntax is simply `T = tensor(M, SYM)` where `SYM` denotes the chosen class of symmetry (`ISO`, `TI`, `ORTHO`).
#
# For example the closest `ISO` tensor is calculated by
#
# `T = tensor(M, ISO)`
#
# In this case it boils down to the "isotropisation" of the tensor $\uuuu{T}$ given by [@bornert2001a]
#
# $$
# \ISO(\uuuu{T})=\left(\uuuu{T}::\uuuu{J}\right)\,\uuuu{J}+\left(\frac{\uuuu{T}::\uuuu{K}}{5}\right)\,\uuuu{K}
# $$

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

M = np.array([ [2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384],
               [1.26432, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089], 
               [0.662772, 1.78964, 2.743, 1.3367, 1.2962, 0.897632], 
               [1.9402, 2.0247 ,1.3367, 4.42684, 2.05632, 1.52686], 
               [1.54905, 1.2701, 1.2962, 2.05632, 3.54431, 1.3445], 
               [1.10384, 1.1089, 0.897632, 1.52686, 1.3445, 1.99356] ])

T = tensor(M,ISO)
print("T =\n", T)

cont4=lambda T1,T2:np.einsum('ij,ij',T1,T2) # T1 and T2 in KM notation so 2 indices
T2 = tensor(cont4(M,J4), cont4(M,K4)/5)
print("T2 =\n", T2)
# -

# Check a well-known result stating that the inverse of the closest isotropic tensor is in general **not** the closest isotropic tensor of the inverse.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

M = np.array([ [2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384],
               [1.26432, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089], 
               [0.662772, 1.78964, 2.743, 1.3367, 1.2962, 0.897632], 
               [1.9402, 2.0247 ,1.3367, 4.42684, 2.05632, 1.52686], 
               [1.54905, 1.2701, 1.2962, 2.05632, 3.54431, 1.3445], 
               [1.10384, 1.1089, 0.897632, 1.52686, 1.3445, 1.99356] ])
T = tensor(M, ISO)
print("inv(ISO(M)) =\n", inv(tensor(M, ISO)))
print("ISO(inv(M)) =\n", tensor(np.linalg.inv(M), ISO))
# -

# The closest `TI` tensor in the frame defined by given angles is calculated by
#
# `T = tensor(M, TI, angles=[θ, ϕ], epsrel=1.e-3)` 
#
# where `epsrel` is a relative stop criterion for the iterative least-square algorithm.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

print(tensor(M, TI, angles=[π/3,π/4]))
# -

# One can also optimize with respect to the angles so as to find the best frame for a given symmetry
#
# `T = tensor(M, TI, epsrel=1.e-3, optiangles=True)`
#
# where `epsrel` is a relative stop criterion for the iterative least-square algorithm.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

print(tensor(M, TI, epsrel=1.e-3, optiangles=True))
# -

# ::: {.callout-caution icon=false} 
#
# ## {{< iconify healthicons exercise-outline >}} Exercise
#
# 1. Find the closest `ORTHO` tensor `T` from the previous matrix `M` (also optimizing the angles)
#
# 1. Build a new tensor from the matrix of `T` (in the canonical frame)
#
# 1. Build a new tensor from the matrix of `T` and the angles of `T`
#
# 1. Find the tensor of best symmetry from the matrix of `T` with angle optimization

# +
#| error: false
#| warning: false
#| code-fold: true
#| code-summary: Solution
#| include: true

M = np.array([ [2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384],
               [1.26432, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089], 
               [0.662772, 1.78964, 2.743, 1.3367, 1.2962, 0.897632], 
               [1.9402, 2.0247 ,1.3367, 4.42684, 2.05632, 1.52686], 
               [1.54905, 1.2701, 1.2962, 2.05632, 3.54431, 1.3445], 
               [1.10384, 1.1089, 0.897632, 1.52686, 1.3445, 1.99356] ])
T = tensor(M, ORTHO, optiangles=True) ; print(T)
print(tensor(T.a))
print(tensor(T.a, angles = T.angles))
print(tensor(T.a, optiangles = True))
# -

# :::
#
# $\,$
