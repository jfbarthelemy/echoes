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

# # Eshelby and Hill polarization tensors {#sec-eshelby_hill}
#
# ::: {.callout-important icon=false} 
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# Before introducing the specific objects of `echoes` devoted to tensor calculations in isotropic or anisotropic contexts, this tutorial aims at providing the syntax allowing to represent second-order or fourth-order tensors under the form of matrices in the Kelvin-Mandel notation as detailed in @sec-KM.
#
# :::
#
# ::: {.callout-note icon=false collapse=true} 
#
# ## {{< iconify flowbite download-outline >}} Download
#
# - [Python script](eshelby_hill.py)
#
# - [Jupyter notebook](eshelby_hill.ipynb)
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
# to display only 8 significant digits of array components
# -

# :::
#
# ## The Eshelby inclusion problem in elasticity [@eshelby1957]
#
# Let the studied domain be the whole $\R^3$ space here filled with a linear elastic material of uniform stiffness tensor $\uuuu{C}$. An ellipsoid centered at the origin of the domain is defined by the equation
#
# $$
#    \uv{x}\in\mathcal{E}_{\uu{A}}
# \quad\Leftrightarrow\quad   
# \uv{x}\cdot(\trans{\uu{A}}\cdot\uu{A})^{-1}\cdot\uv{x}\leq 1
# $$
# where $\uu{A}$ is an invertible second-order tensor so that $\trans{\uu{A}}\cdot\uu{A}$ is a positive definite symmetric tensor associated to 3 radii (eigenvalues $a\geq b \geq c$ possibly written $\rho_1 \geq \rho_2 \geq \rho_3$ for convenience) and 3 angles (orientation of the frame of eigenvectors $\uv{e}_1, \uv{e}_2, \uv{e}_3$)
#
# $$
# \trans{\uu{A}}\cdot\uu{A}=a^2 \uv{e}_1\otimes\uv{e}_1 + b^2 \uv{e}_2\otimes\uv{e}_2 + c^2 \uv{e}_3\otimes\uv{e}_3 = \sum_{i=1}^3 \rho_i \uv{e}_i\otimes\uv{e}_i
# $$
#
# A polarization field is introduced as a second-order symmetric tensor field $\uu{\tau}$ such that the constitutive law of the medium is given by
#
# $$
#     \sig(\x)=\uuuu{C}:\eps(\x)+\uu{\tau}(\x)\quad\textrm{where}\quad
#     \uu{\tau}(\x)=
#     \left\{
#     \begin{array}{ll}
#     \uu{0}&\textrm{ if }\x \notin \mathcal{E}_{\uu{A}}\\
#     \uu{\tau}\textrm{ (uniform)}&\textrm{ if }\x \in \mathcal{E}_{\uu{A}}
#     \end{array}
#     \right.
# $${#eq-law}
#
# In addition the displacement tends towards $\uv{0}$ at infinity.
#
# ::: {#fig-eurlerangles}
#
# ![](../img/eshelbypbincl.png){width=30%}
#
# Eshelby inclusion problem
# :::
#
# The important result derived in @eshelby1957 is that the strain tensor field solution to this problem is uniform within the ellipsoidal domain (not outside!) and writes
#
# $$
#    \forall \x \in \mathcal{E}_{\uu{A}},\quad \eps(\x)=-\uuuu{P}:\uu{\tau}\quad\textrm{(uniform)}
# $${#eq-defHill}
# where $\uuuu{P}$, called the Hill polarization tensor, depends on the shape and orientation of the ellipsoid (i.e. $\uu{A}$ up to a multiplicative constant) and on the matrix behavior (i.e. $\uuuu{C}$), in other words $\uuuu{P}=\uuuu{P}(\uu{A},\uuuu{C})$.
#
# The polarization problem can be alternatively posed thanks to an eigenstrain tensor $\eps^*$, uniform in the ellipsoid, such that $\eps^*=-\uuuu{C}^{-1}:\uu{\tau}$. It follows that @eq-law rewrites
#
# $$
#     \sig(\x)=\uuuu{C}:(\eps(\x)-\eps^*(\x))\quad\textrm{where}\quad
#     \eps^*(\x)=
#     \left\{
#     \begin{array}{ll}
#     \uu{0}&\textrm{ if }\x \notin \mathcal{E}_{\uu{A}}\\
#     \eps^*\textrm{ (uniform)}&\textrm{ if }\x \in \mathcal{E}_{\uu{A}}
#     \end{array}
#     \right.
# $${#eq-law2}
# and the Eshelby tensor $\uuuu{S}$ is introduced by rewriting @eq-defHill under the form
# $$
#    \forall \x \in \mathcal{E}_{\uu{A}},\quad \eps(\x)=\uuuu{S}:\eps^*\quad\textrm{where}\quad\uuuu{S}=\uuuu{P}:\uuuu{C}
# $$
#
# See in @sec-hill_elas explicit expressions to calculate the Hill polarization tensor.
#
# ## Definition of the `ellipsoidal` shape
#
# An `ellipsoidal` shape intended for use in the construction of $\uuuu{P}$ can be built by means of the following constructor
#
# `shape = ellipsoidal(a, b, c, θ=0., ϕ=0., ψ=0.) # default values 0 for the angles`
#
# where $a, b, c$ are the radii and $\theta, \phi, \psi$ the Euler angles already presented in @fig-eulerangles
#
# ::: {.callout-warning} 
#
# ## Warning
#
# If the radii are not entered in decreasing order, the constructor reorders them and recomputes the angles accordingly.
#
# :::
#
# When two radii are equal the ellipsoidal shape becomes a spheroidal one which can also be directly defined by
#
# `shape = spheroidal(ω, θ=0., ϕ=0.) # default values 0 for the angles`
#
# where $\omega$ is the aspect ratio ($\omega \leq 1$ for an oblate shape and $\omega \geq 1$ for a prolate shape) (see @fig-spheroidal).
#
# ::: {#fig-spheroidal layout-ncol=2}
#
# ![Oblate](../img/oblate){#fig-oblate}
#
# ![Prolate](../img/prolate){#fig-prolate}
#
# Spheroidal shapes
# :::
#
# Finally if the three radii are equal, the shape becomes spherical
#
# `shape = spherical`
#
# ::: {.callout-warning} 
#
# ## Warning
#
# These objects are supposed to describe an inclusion in the Eshelby problem, thus they are defined up to a multiplicative constant. This explains why the spheroidal shape is defined only with an aspect ratio and the sphere is not defined with a radius. On the contrary the general ellipsoidal shape depends on three radii but the result of $\uuuu{P}$ will only depend on two ratios.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

π = math.pi
a, b, c = 10., 5., 2.
θ, ϕ, ψ = π/3, π/4, π/5
print(ellipsoidal(a, b, c, θ, ϕ, ψ))
print(ellipsoidal(c, b, a, θ, ϕ, ψ)) # the constructor reorders the radii and recomputes the angles accordingly
ω = 0.1
print(spheroidal(ω, θ, ϕ))
print(spherical)
# -

# ## Calculation of Hill and Eshelby tensors for an isotropic matrix
#
# The Hill and Eshelby tensors are simply calculated by
#
# `P = hill(shape,C)`
# `S = eshelby(shape,C)`
#
# where `shape` is an ellipsoidal shape as previously defined and `C` is an isotropic tensor.
#
# Note that an isotropic tensor `C` simply allows to resort to analytical expressions as seen in @sec-hill_elas. The anisotropic cases are also available with numerical algorithms as will be seen below.
#
# ::: {.callout-warning} 
#
# ## Warning
#
# The functions `hill` and `eshelby` return $6×6$ matrices i.e. of type `numpy.ndarray` (not `tensor`). It is however possible to build a `tensor` object from the matrix for the Hill tensor $\uuuu{P}$ since the latter is always positive definite. This is not in general the case for the Eshelby tensor $\uuuu{S}$.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

a, b, c = 10., 5., 2.
θ, ϕ, ψ = π/3, π/4, π/5
k, µ = 72., 32. 
shape = ellipsoidal(a, b, c, θ, ϕ, ψ)
C = stiff_kmu(k, µ)
P = hill(shape,C)
print("P=\n",P)
print("tensor P=\n",tensor(P, optiangles=True))
S = eshelby(shape,C)
print("S=\n",S)
# -

# ::: {.callout-caution icon=false} 
#
# ## {{< iconify healthicons exercise-outline >}} Exercise
#
# For a spherical shape we have

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

k, µ = 72., 32. ; C = stiff_kmu(k, µ)
print("P=\n", hill(spherical,C))
# -

# Check the following formula for the Hill tensor related to a sphere in an isotropic matrix
#
# $$
# \uuuu{P}(\uu{1},3\,k\,\uuuu{J}+2\,\mu\,\uuuu{K})=\frac{1}{3\,k+4\,\mu}\,\left(\uuuu{J}+\frac{3\,(k+2\,\mu)}{5\,\mu}\uuuu{K}\right)
# $$

# +
#| error: false
#| warning: false
#| code-fold: true
#| code-summary: Solution
#| include: true

P = 1/(3*k+4*µ)*(J4+3*(k+2*µ)/(5*μ)*K4)
print("P=\n", P)
# -

# :::
#
# ## Calculation of Hill and Eshelby tensors for an arbitrary anisotropic matrix 
#
# The Hill and Eshelby tensors are numerically calculated by
#
# - `P = hill(shape, C, algo=NUMINT, epsroots=1.e-4, epsabs=1.e-4, epsrel=1.e-4, maxnb=100000)`
# - `S = eshelby(shape, C, algo=NUMINT, epsroots=1.e-4, epsabs=1.e-4, epsrel=1.e-4, maxnb=100000)`
#
# where the keyword argument `algo` can be either `NUMINT` or `RESIDUES` (default: `NUMINT`).
#
# - `NUMINT` based on the cubature DECUHR algorithm provided in @espelid1994 applied on the integral formulation @eq-Hill over the unit sphere
#     - `epsabs` is the tolerance of absolute error (default: 1.e-4)
#     - `epsrel` is the tolerance of relative error (default: 1.e-4)
#     - `maxnb` is the maximum number of subdivisions (default: 100000)
#     
#   For more information about these parameters see @espelid1994. 
#
# - `RESIDUES` based on the residue Cauchy theorem to transform the 2D cubature over the unit sphere @eq-Hill into a 1D quadrature as demonstrated in @masson2008
#     - `epsroots` is the tolerance for the criterion of multiple poles, i.e. if the distance between two poles of the integrand function is under this value they are considered as a unique multiple pole (default: 1.e-4)
#  
# ::: {.callout-warning} 
#
# ## Warning
#
# The `RESIDUES` algorithm is much faster than the `NUMINT` one in case of full anisotropy. Nevertheless the latter proves to be more robust in the neighborhood of symmetry axes where the presence of close poles may induce numerical errors in the application of the Cauchy residue theorem. This is why `NUMINT` is chosen by default.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

param = [ 2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089, 2.743, 1.3367, 1.2962, 0.897632, 4.42684, 2.05632, 1.52686, 3.54431, 1.3445, 1.99356 ]
C = tensor(param)
print(hill(shape, C, algo=NUMINT),"\n")
print(hill(shape, C, algo=RESIDUES))
# -

# See below how a too large `epsroots` parameter can degrade the precision of the result

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

param = [3.,4.,math.sqrt(2),2.,1.]
np.random.seed(1948)
α = 1.e-2*np.random.rand(6, 6)
C = tensor(param)+tensor((α+α.T)/2) ; print(C)
print(hill(shape, C, algo=NUMINT),"\n")
for ϵ in [1.e-4, 1.e-2, 1.e-1]:
    print(f"ϵ={ϵ}\n",hill(shape, C, algo=RESIDUES, epsroots=ϵ),"\n")
# -

# $\,$