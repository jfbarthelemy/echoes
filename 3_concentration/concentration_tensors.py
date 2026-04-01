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
#         href: concentration_tensors.py
# ---
#
# # Concentration tensors {#sec-concentration_tensors}
#
# ::: {.callout-important icon=false}
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial introduces the Eshelby inhomogeneity problem and the associated concentration tensors. It shows how to build an ellipsoidal inhomogeneity in *Echoes* and how to compute the four concentration tensors relating microscopic and macroscopic strains and stresses.
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
# ## The Eshelby inhomogeneity problem [@eshelby1957]
#
# As in the inclusion problem presented in @sec-eshelby_hill, the studied domain $\Omega$ is the whole $\R^3$ space. An ellipsoidal subdomain is still defined by $\uv{x}\cdot(\trans{\uu{A}}\cdot\uu{A})^{-1}\cdot\uv{x}\leq 1$.
#
# The material still obeys a linear elastic behavior with a uniform stiffness tensor $\uuuu{C}$ outside the ellipsoid but now the material filling the ellipsoid is characterized by another stiffness tensor $\uuuu{C}^\mathcal{I}$ without any polarization. This situation somehow corresponds to a fictitious polarization such that
#
# $$
#     \sig(\x)=\uuuu{C}:\eps(\x)+\uu{\tau}(\x)\quad\textrm{where}\quad
#     \uu{\tau}(\x)=
#     \left\{
#     \begin{array}{ll}
#     \uu{0}&\textrm{ if }\x \notin \mathcal{E}_{\uu{A}}\\
#     \left(\uuuu{C}^\mathcal{I}-\uuuu{C}\right):\eps(\x)=\delta\uuuu{C}:\eps(\x)  &\textrm{ if }\x \in \mathcal{E}_{\uu{A}}
#     \end{array}
#     \right.
# $${#eq-law-inhom}
#
# In addition the displacement is now of the form $\E\cdot\x$ at infinity (i.e. remote homogeneous strain condition such that the problem solution would be $\eps=\E$ if the ellipsoid and the matrix shared the same elasticity).
#
# ::: {#fig-eshelby-inhom}
#
# ![](../img/eshelbypb.png){width=30%}
#
# Eshelby inhomogeneity problem
# :::
#
# The second important result derived in [@eshelby1957] is that the strain tensor field solution to this problem is also uniform within the ellipsoidal domain (not outside!) and writes
#
# $$
#    \forall \x \in \mathcal{E}_{\uu{A}},\quad \eps(\x)=\uuuu{A}^E:\E \quad\textrm{ where }
#    \uuuu{A}^E=\left(\uuuu{I}+\uuuu{P}:\delta\uuuu{C}\right)^{-1}
# $${#eq-ell-concentration}
#
# $\uuuu{A}^E$ is called the **strain-strain concentration tensor** for the Eshelby problem and $\uuuu{P}(\uu{A},\uuuu{C})$ is the Hill polarization tensor already introduced in @sec-eshelby_hill, depending only on the shape and orientation of the ellipsoid $\uu{A}$ and the matrix behavior $\uuuu{C}$, not on that of the ellipsoidal inhomogeneity $\uuuu{C}^\mathcal{I}$.
#
# ## Ellipsoidal inhomogeneity in *Echoes* {#sec-ellipsoid}
#
# An inhomogeneity of ellipsoidal shape is built in *Echoes* by
#
# ```
# ell = ellipsoid(shape=..., prop={"C":..., "D":...})
# ```
#
# Such an `ellipsoid` object is defined by a `shape` (either `ellipsoidal`, `spheroidal` or `spherical` as presented in @sec-ellipsoidal) and a dictionary of properties of names arbitrarily chosen by the user.
#
# It is possible to change the `shape` and `prop` values after construction:
#
# ```
# ell.shape = spheroidal(0.1, math.pi/3, math.pi/4)
# ell.set_prop("C", stiff_kmu(3., 2.))
# ```
#
# In order to simulate the Eshelby problem so as to calculate the concentration tensor $\uuuu{A}^E$, it is necessary to define the reference medium in which the ellipsoid is embedded. This is done by
#
# ```
# C = ...
# ell.set_ref("C", C)
# ```
#
# ::: {.callout-warning}
#
# ## Warning
#
# The tensor introduced in `set_ref` must not be built on the fly, i.e. it must be built before using in `set_ref` and not destroyed before calculating concentration tensors. Moreover `set_ref` must be applied each time the `ellipsoid` is modified (by `shape` or `set_prop`).
#
# :::
#
# Once the reference medium is set for the desired property, the strain-strain concentration tensor is obtained by
#
# ```
# ell.eE
# ```
#
# ::: {.callout-note}
#
# ## Notes
#
# - The terminology `eE` recalls that this concentration tensor relates the microscopic strain (`e` for **e**psilon) to the macroscopic strain (`E` for $\E$).
# - The object returned is not a `tensor` (which is designed only for symmetric tensors) but a $6\times 6$ `numpy.ndarray` since $\uuuu{A}^E$ may not fulfill the major symmetry (i.e. not necessarily symmetric in its Kelvin-Mandel notation).
# - The Hill tensor required in the concentration tensor is calculated with default parameters (analytical if the matrix is isotropic, using the `NUMINT` algorithm if the matrix is anisotropic). However it is possible to force parameters by applying `ell.set_param_eshelby(algo=NUMINT, epsroots=1.e-4, epsabs=1.e-4, epsrel=1.e-4, maxnb=100000)`.
#
# :::

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

ell = ellipsoid(shape=spherical, prop={"C": stiff_kmu(5., 3.)})
C = stiff_kmu(72., 32.)
for ω in [0.1, 1.]:
    ell.shape = spheroidal(ω)
    ell.set_ref("C", C)
    A = ell.eE
    print(f"AE(ω={ω})=\n", A, "\n")
    # note that A is not major-symmetric if ω≠1

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

ell.shape = spheroidal(0.1, math.pi/3, math.pi/4)
ell.set_prop("C", stiff_kmu(3., 2.))
ell.set_ref("C", C)
print(ell)
print(ell.eE)
# -

# ## The four concentration tensors {#sec-four-concentration}
#
# Observing that $\sig=\uuuu{C}^\mathcal{I}:\eps$ within the ellipsoid and introducing the macroscopic stress tensor $\Sig=\uuuu{C}:\E$, all concentration tensors can be defined as
#
# $$
# \begin{aligned}
#     \eps_{|\mathcal{E}_{\uu{A}}}=\uuuu{A}^E:\E&&\\
#     \eps_{|\mathcal{E}_{\uu{A}}}=\uuuu{A}^\Sigma:\Sig&\quad\textrm{ with }\quad \uuuu{A}^\Sigma=\uuuu{A}^E:\uuuu{C}^{-1}&\\
#     \sig_{|\mathcal{E}_{\uu{A}}}=\uuuu{B}^E:\E&\quad\textrm{ with }\quad \uuuu{B}^E=\uuuu{C}^\mathcal{I}:\uuuu{A}^E&\\
#     \sig_{|\mathcal{E}_{\uu{A}}}=\uuuu{B}^\Sigma:\Sig&\quad\textrm{ with }\quad \uuuu{B}^\Sigma=\uuuu{C}^\mathcal{I}:\uuuu{A}^E:\uuuu{C}^{-1}&
# \end{aligned}
# $${#eq-ell-concentration2}
#
# These concentration tensors are accessible through the attributes `eE`, `eS`, `sE` and `sS`.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

ell = ellipsoid(shape=spherical, prop={"C": stiff_kmu(5., 3.)})
C = stiff_kmu(72., 32.)
for ω in [0.1, 1.]:
    ell.shape = spheroidal(ω)
    ell.set_ref("C", C)
    print(f"AE(ω={ω})=\n", ell.eE, "\n")
    print(f"AS(ω={ω})=\n", ell.eS, "\n")
    print(f"BE(ω={ω})=\n", ell.sE, "\n")
    print(f"BS(ω={ω})=\n", ell.sS, "\n")
# -

# $\,$
