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
#         href: derivatives.py
# ---
#
# # Derivatives of effective properties {#sec-derivatives}
#
# ::: {.callout-important icon=false}
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial presents the `homogenize_derivative` function for computing the derivatives of the effective stiffness tensor with respect to the moduli of individual phases. These derivatives give access to **second-order moments** (quadratic averages) of the local strain and stress fields, which are central to nonlinear extensions (see @sec-nonlinear).
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
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)
# -

# :::
#
# ## Second-order moments of local fields {#sec-second-moments}
#
# A key result of micromechanics [@kreher1990] is that the **second-order moments** (quadratic averages) of the local strain field in a phase $r$ can be obtained directly from the derivative of the effective stiffness with respect to the elastic moduli of that phase. For an isotropic phase $r$ with shear modulus $\mu_r$, the mean deviatoric strain variance in phase $r$ reads:
#
# $$
# f_r\,\langle\varepsilon_d^2\rangle_{\Omega_r} = \frac{1}{2}\,\uu{E}:\frac{\partial\uuuu{C}^{\rm hom}}{\partial\mu_r}:\uu{E}
# $${#eq-kreher-iso}
#
# where $f_r$ is the volume fraction of phase $r$, $\uu{E}$ the macroscopic strain, and $\varepsilon_d^2 = \uu{\varepsilon}':\uu{\varepsilon}'$ (with $\uu{\varepsilon}' = \uu{\varepsilon} - \frac{1}{3}\mathrm{tr}(\uu{\varepsilon})\,\uu{1}$ the deviatoric strain).
#
# More generally, the full second-order strain moment tensor in phase $r$ is related to the derivative with respect to the full stiffness tensor:
#
# $$
# f_r\,\langle\varepsilon_{ij}\,\varepsilon_{kl}\rangle_{\Omega_r} = \frac{1}{2}\,E_{mn}\,\frac{\partial C^{\rm hom}_{mnpq}}{\partial c^r_{ijkl}}\,E_{pq}
# $${#eq-kreher-aniso}
#
# where `index` in `homogenize_derivative` addresses specific components $c^r_{ijkl}$ in the condensed representation. These relations underpin the **modified secant method** for nonlinear composites [@suquet1997] (see @sec-nonlinear).
#
# ## The `homogenize_derivative` function {#sec-homogenize-derivative}
#
# After computing the effective properties of a RVE with `homogenize`, the derivative of the effective stiffness with respect to a phase modulus is obtained by:
#
# ```
# dChom = homogenize_derivative(prop="C", rve=myrve, scheme=sch,
#                                phase="PHASE_NAME", index=i, sym=ISO)
# ```
#
# where:
#
# - `phase` is the name of the phase whose modulus is differentiated
# - `index` is the index of the parameter in the condensed representation (0 for the first parameter, 1 for the second, etc.). For isotropic tensors (`ISO`): index 0 corresponds to $3k$ and index 1 to $2\mu$
# - `sym` is the symmetry class (`ISO`, `TI`, `ORTHO`, `ANISO`)
#
# For n-layer spheres, an additional `layer` argument specifies which layer:
#
# ```
# dChom = homogenize_derivative(prop="C", rve=myrve, scheme=sch,
#                                phase="SPN", layer=0, index=1, sym=ISO)
# ```
#
# ## Example: sensitivity to matrix shear modulus

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

myrve = rve(matrix="SOLID")
myrve["SOLID"] = ellipsoid(shape=spheroidal(1.), symmetrize=[ISO],
                           prop={"C": stiff_kmu(72., 32.)}, fraction=0.8)
myrve["PORE"] = ellipsoid(shape=spheroidal(0.5), symmetrize=[ISO],
                          prop={"C": stiff_kmu(1.e-6, 1.e-6)}, fraction=0.2)

C = homogenize(prop="C", rve=myrve, scheme=MT, verbose=False)
print("Chom =", C)
print(f"k_hom={C.k:.4f}, mu_hom={C.mu:.4f}")

# Derivative with respect to 2*mu of the solid phase
dC_dmu = homogenize_derivative(prop="C", rve=myrve, scheme=MT,
                                phase="SOLID", index=1, sym=ISO)
print("\ndChom/d(2mu_solid):\n", dC_dmu)
# -

# ## Validation by finite differences
#
# The analytical derivative can be validated against a finite-difference approximation:

# +
#| error: false
#| warning: false
#| code-summary: "Finite-difference validation"
#| include: true

mu_s = 32.
eps = 1.e-5

# Perturbed computations
myrve["SOLID"].set_prop("C", stiff_kmu(72., mu_s + eps))
Cp = homogenize(prop="C", rve=myrve, scheme=MT, verbose=False)

myrve["SOLID"].set_prop("C", stiff_kmu(72., mu_s - eps))
Cm = homogenize(prop="C", rve=myrve, scheme=MT, verbose=False)

# Restore original
myrve["SOLID"].set_prop("C", stiff_kmu(72., mu_s))

dC_fd = (Cp.array - Cm.array) / (2 * eps)
print("Finite difference dChom/dmu:\n", dC_fd)

# Compare with analytical (index=1 gives d/d(2mu), so multiply by 2)
print("\nAnalytical dChom/d(2mu) * 2:\n", 2 * dC_dmu.array)
print("\nMax difference:", np.max(np.abs(dC_fd - 2 * dC_dmu.array)))
# -

# ## Second-order strain moments in practice
#
# Using (@eq-kreher-iso), the mean deviatoric strain variance in the solid phase can be computed from the derivative with respect to $\mu_s$. For a macroscopic strain $\uu{E} = E\,\uv{e}_1 \otimes \uv{e}_1$ (uniaxial):

# +
#| error: false
#| warning: false
#| code-summary: "Second-order strain moments — Kreher formula"
#| include: true

# Restore the RVE from the previous example
myrve["SOLID"].set_prop("C", stiff_kmu(72., 32.))
C = homogenize(prop="C", rve=myrve, scheme=MT, verbose=False)

# Derivative w.r.t. mu_s  (index=1 gives d/d(2*mu))
dC_dmu = homogenize_derivative(prop="C", rve=myrve, scheme=MT,
                                phase="SOLID", index=1, sym=ISO)

# Macroscopic strain: uniaxial E_11 = 1, all other = 0
E = np.zeros((6, 1))
E[0, 0] = 1.   # E_11 in Kelvin-Mandel (index 0)

# f_s * <eps_d^2>_solid = (1/2) E : dChom/dmu_s : E
# Note: index=1 → d/d(2mu), so dChom/dmu = 2 * dC_dmu
f_s = myrve["SOLID"].fraction
eps_d2_solid = float(E.T @ (2. * dC_dmu.array) @ E) / (2. * f_s)

print(f"f_s = {f_s:.2f}")
print(f"<eps_d^2>_solid (uniaxial E_11=1) = {eps_d2_solid:.6f}")
# -

# $\,$
