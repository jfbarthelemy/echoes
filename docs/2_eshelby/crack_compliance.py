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
#         href: crack_compliance.py
# ---
#
# # Crack compliance tensors {#sec-crack_compliance}
#
# ::: {.callout-important icon=false}
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial shows how to calculate:
#
# - the **elastic crack compliance tensor** $\uuuu{H}$ of an elliptical crack in a matrix of arbitrary anisotropy;
# - the **linear spring interface model** that accounts for partial bonding through normal ($k_n$) and tangential ($k_t$) spring stiffnesses;
# - the **conductivity (2nd-order) crack compliance tensor** $\uu{h}$, the transport analog of $\uuuu{H}$.
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
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)
# to display only 8 significant digits of array components
# -

# :::
#
# ## Definition of the crack compliance
#
# Geometrically an elliptical crack can be defined as the limit domain of an ellipsoid for which the smallest aspect ratio tends towards 0. Keeping the definition of the ellipsoid by means of the tensor $\uu{A}$ in (@eq-defell), the latter tensor becomes here
# $$
# \uu{A}=
# \uv{\ell}\otimes\uv{\ell}+
# \eta\,\uv{m}\otimes\uv{m}+
# \omega\,\uv{n}\otimes\uv{n}
# \quad\textrm{with}\quad
# \eta=\frac{b}{a}
# \quad\textrm{and}\quad
# \omega=\frac{c}{a}
# $$
#
# ::: {#fig-crack}
# ::: {.content-visible unless-format="html"}
# ![](../img/crack){width=70%}
# :::
# ::: {.content-visible when-format="html"}
# ![](../img/crack.html){width=70%}
# [![](../img/arrow-expand.png){width=3%}](../img/crack.html)
# :::
# Ellipsoidal crack
# :::
#
# In the case of cracks, it is useful to recall the second Hill polarization tensor defined as (see (@eq-defsecondHill))
# $$
# \uuuu{Q}=\uuuu{C}-\uuuu{C}:\uuuu{P}:\uuuu{C}
# $$
# and in particular the so-called crack compliance [@kachanov1992], [@kachanov1993], [@sevostianov2002], [@barthelemyIJES2021]
# $$
# \uuuu{H}=\lim_{\omega\to 0}\omega\,\uuuu{Q}^{-1}
# $$
# in which it is recalled that $\uuuu{P}$ and thus $\uuuu{Q}$ depend on $\omega$ such that the components $\left(\uuuu{Q}^{-1}\right)_{nijk}$ (with $n$ corresponding to the crack normal) behave as $1/\omega$ when $\omega$ tends towards $0$. It is also shown in previously cited references that $\uuuu{H}$ actually derives from a symmetric second-order tensor $\uu{B}$ as
# $$
# \uuuu{H}=
# \lim_{\omega\to 0} \omega\,\uuuu{Q}^{-1}
# =\frac{3}{4}\,\uv{n}\sotimes\uu{B}\sotimes\uv{n}
# $${#eq-H}
# For more details about this limit, the reader can also refer to [@laws1977], [@laws1985], [@nemat-nasser1999], [@barthelemyIJSS2009], [@kachanov2018], [@barthelemyIJES2021].
#
# For an arbitrarly anisotropic matrix, an algorithm allowing to estimate the limit (@eq-H) is proposed in [@barthelemyIJSS2009] whereas in the isotropic case $\uu{B}$ writes
# $$
# \uu{B}=
# B_{nn}\,\uv{n}\otimes\uv{n}
# +
# B_{mm}\,\uv{m}\otimes\uv{m}
# +
# B_{\ell\ell}\,\uv{\ell}\otimes\uv{\ell}
# $$
# with
# $$\begin{align}
# B_{nn}&=\frac{8\,\eta\,(1-\nu^2)}{3\,E}\,
# \frac{1}{\mathcal{E}_\eta}\label{eq:Bnn}\\
# B_{mm}&=\frac{8\,\eta\,(1-\nu^2)}{3\,E}\,
# \frac{1-\eta^2}{\left(1-(1-\nu)\,\eta^2\right)
# \,\mathcal{E}_\eta-\nu\,\eta^2\,\mathcal{K}_\eta}\\
# B_{\ell\ell}&=\frac{8\,\eta\,(1-\nu^2)}{3\,E}\,
# \frac{1-\eta^2}{(1-\nu-\eta^2)\,\mathcal{E}_\eta+\nu\,\eta^2\,\mathcal{K}_\eta}
# \end{align}$$
# where $\mathcal{K}_\eta=\mathcal{K}(\sqrt{1-\eta^2})$ and $\mathcal{E}_\eta=\mathcal{E}(\sqrt{1-\eta^2})$
# are the complete elliptic integrals of respectively the first and second kind (see [@abramowitz1972]).
# If the crack is circular, the components of $\uu{B}$ become
# $$
# B_{nn}=\frac{16\,(1-\nu^2)}{3\,\pi\,E}
# \quad\textrm{;}\quad
# B_{mm}=B_{\ell\ell}=\frac{B_{nn}}{1-\nu/2}
# $$
#
# ## Numerical implementation
#
# The syntax allowing to calculate the compliance $\uuuu{H}$ of an elliptical crack embedded in an infinite matrix of elastic stiffness $\uuuu{C}$ is very close to that corresponding to `hill` and `eshelby` in @sec-eshelby_hill, in other words
#
# `H = crack_compliance(shape, C, algo=NUMINT, epsroots=1.e-4, epsabs=1.e-4, epsrel=1.e-4, maxnb=100000)`
#
# where the keyword argument `algo` can be either `NUMINT` or `RESIDUES` (default: `NUMINT`). See [@barthelemyIJSS2009] for more details.
#
#
# ::: {.callout-warning} 
#
# ## Warning
#
# As recalled above the elliptical crack is described as a flat ellipsoid. However the calculation of the crack compliance $\uuuu{H}$ (@eq-H) relies on an `ellipsoidal` argument (see @sec-ellipsoidal) in which even the smallest aspect ratio should be strictly positive for numerical reasons. It is then necessary to provide an aspect ratio $\omega$ for the crack even if the crack compliance is actually calculated as a limit (not depending on $\omega$).
#
# :::
#
# **Isotropic example**

# +
#| error: false
#| warning: false
#| code-fold: false
#| code-summary: Code
#| include: true

C = stiff_Enu(1.,0.2) ; print("C =\n", C)
ω = 1.e-4
H = crack_compliance(spheroidal(ω), C) ; print("H =\n", H)
# -

# **Anisotropic example**

# +
#| error: false
#| warning: false
#| code-fold: false
#| code-summary: Code
#| include: true

M = np.array([ [2.66011, 1.26432, 0.662772, 1.9402, 1.54905, 1.10384],
               [1.26432, 3.6072, 1.78964, 2.0247, 1.2701, 1.1089], 
               [0.662772, 1.78964, 2.743, 1.3367, 1.2962, 0.897632], 
               [1.9402, 2.0247 ,1.3367, 4.42684, 2.05632, 1.52686], 
               [1.54905, 1.2701, 1.2962, 2.05632, 3.54431, 1.3445], 
               [1.10384, 1.1089, 0.897632, 1.52686, 1.3445, 1.99356] ])
C = tensor(M) ; print(C)
ω = 1.e-4
H = crack_compliance(spheroidal(ω), C, algo=NUMINT) ; print("H =\n", H)

# +
#| error: false
#| warning: false
#| code-fold: false
#| code-summary: Code
#| include: true

H = crack_compliance(spheroidal(ω), C, algo=RESIDUES) ; print("H =\n", H)
# -

# **Transversely isotropic matrix**
#
# A transversely isotropic (TI) matrix is fully defined by 5 independent elastic constants. *Echoes* handles the crack compliance analytically in this case too.

# +
#| error: false
#| warning: false
#| code-fold: false
#| code-summary: Code
#| include: true

# TI stiffness: tensor([c1, c2, c3, c4, c5]) with axis of symmetry = e3
# Parameters from Hönig parametrization (E1=2.3, h=2.5, nu2=0.2, nu1=0.2, gamma=2.)
import math as _m
h, E1, nu2, nu1, gamma = 2.5, 2.3, 0.2, 0.2, 2.0
d = 1 - nu1 - 2*h*nu2**2
C_TI = tensor([h*E1*(1-nu1)/d, E1/d, _m.sqrt(2.)*h*nu2*E1/d, E1/(1+nu1), gamma*E1/(1+nu1)])
print("C (TI) =\n", C_TI, "\n")

H_TI = crack_compliance(spheroidal(1.e-4), C_TI)
print("H (TI matrix) =\n", H_TI)
# -

# ::: {.callout-caution icon=false}
#
# ## {{< iconify healthicons exercise-outline >}} Exercise
#
# This exercise aims at checking the aspect ratio for which the approximation $\omega\,\uuuu{Q}^{-1}\approx \lim_{\omega\to 0}\omega\,\uuuu{Q}^{-1}$ remains acceptable. For this purpose it is proposed to build the graph of relative distance between these two tensors as function of the aspect ratio $\omega$ for different reference stiffness tensors $\uuuu{C}$.
#
# *Hints: with `hill_dual` use preferably `algo=NUMINT` and decrease the values of `epsrel` and `espabs` to avoid roundoff error prior to inversion of $\uuuu{Q}$*

# +
#| label: fig-errorH
#| fig-cap: "Influence of the aspect ratio on the crack compliance contribution tensor"
#| error: false
#| warning: false
#| code-fold: true
#| code-summary: Solution
#| include: true

plt.figure(figsize=(8,3))

for i in range(4):
    A = np.random.rand(6,6)
    C = tensor(A.T.dot(A) + np.eye(6)) # generation of an arbitrary positive definite matrix
    ω = 1.e-4
    H = crack_compliance(spheroidal(ω), C)

    tω = np.logspace(-5,1,20)
    tabδ = []
    for ω in tω:
        Q = hill_dual(spheroidal(ω), C, algo=NUMINT, epsrel=1.e-12, epsabs=1.e-12)
        Hω = ω*np.linalg.inv(Q)
        δH = np.linalg.norm(Hω-H)/np.linalg.norm(H)
        tabδ.append(δH)
    plt.loglog(tω,tabδ,'+-')
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\frac{||\mathbb{H}-\omega\,\mathbb{Q}^{-1}||}{||\mathbb{H}||}$")
plt.grid(True,which='both')
plt.show()
# -

# :::
#
# ## Linear spring interface model {#sec-spring-model}
#
# In the fully open crack model considered so far, the two crack faces are traction-free. A more general model allows the faces to interact through a **linear spring interface**: the traction $\uv{T}=\uu{\sig}\cdot\uv{n}$ on the crack face is linearly related to the displacement jump $\jump{\uv{u}}$:
#
# $$
# \uv{T} = \left(k_n\,\uv{n}\otimes\uv{n} + k_t\,(\uu{1}-\uv{n}\otimes\uv{n})\right)\cdot\jump{\uv{u}}
# $${#eq-spring-law}
#
# where $k_n \geq 0$ is the normal spring stiffness and $k_t \geq 0$ the tangential one. The open crack ($k_n=k_t=0$) and the perfectly bonded interface ($k_n,k_t\to\infty$) are limiting cases.
#
# ### Implementation
#
# The spring stiffnesses are passed to the `crack` constructor via `interf_prop` (details in @sec-cracked-media):
#
# ```python
# ver["CRACK"] = crack(shape=spheroidal(ω), density=d,
#                      interf_prop={"C": [kn, kt]})
# ```
#
# The order of parameters is `[kn, kt]` (normal first, tangential second). Passing `prop={"C": tZ4}` with no `interf_prop` is equivalent to the open crack ($k_n=k_t=0$).
#
# ## Conductivity crack resistivity {#sec-crack-conduc}
#
# The proper transport analog of the elastic crack compliance tensor is the **crack resistivity tensor** $\uu{H}$ [@kachanov2018]. By analogy with the elastic case, it is defined from the 2nd-order second Hill tensor $\uu{Q}=\uu{K}-\uu{K}\cdot\uu{P}\cdot\uu{K}$ as:
#
# $$
# \uu{H} = \lim_{\omega\to 0}\,\omega\,\uu{Q}^{-1}
# $${#eq-H-cond}
#
# The result is a positive semi-definite 3×3 tensor (rank-1 for an isotropic matrix, since only the normal component of the flux can jump across a crack).
#
# ::: {.callout-warning}
#
# ## Terminology in *Echoes*
#
# Although the physically correct term for the transport analog is **crack resistivity**, the function `crack_compliance` is used for both elasticity and transport in *Echoes* by analogy: passing a 2nd-order tensor `K` as the second argument automatically returns the 3×3 crack resistivity tensor $\uu{H}$.
#
# :::
#
# ### API
#
# `crack_compliance(shape, K)` with a **2nd-order tensor** `K` returns a 3×3 `numpy.ndarray`. As for the elastic case, $\uu{H}$ does not depend on the aspect ratio $\omega$ when $\omega\to 0$, so only a small value needs to be provided.
#
# For a spring-interface crack in conductivity, `interf_prop={"K": [Ka, Kb, Kc]}` sets the conductance along the three local axes of the crack (the two tangential axes first, then the normal):
#
# ```python
# ver["CRACK"] = crack(shape=spheroidal(ω), density=d,
#                      interf_prop={"K": [Kt1, Kt2, Kn]})
# ```
#
# For a spheroidal crack with rotational symmetry, $K_{t1}=K_{t2}=K_t$ so the call reduces to `[Kt, Kt, Kn]`.
#
# ### Isotropic matrix

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

K_val = 2.5
K_iso = K_val * tId2

H_cond = crack_compliance(spheroidal(1.e-4), K_iso)
print("H (crack resistivity, isotropic) =\n", H_cond)

# Analytical formula: H_n = 2/(πK)  (n = e3, index 2)
H_n_formula = 2.0 / (math.pi * K_val)
print(f"\nFormula H[2,2] = 2/(πK) = {H_n_formula:.8f}")
print(f"Echoes  H[2,2]          = {H_cond[2,2]:.8f}")
print(f"Error:                    {abs(H_cond[2,2] - H_n_formula):.2e}")
# -

# ### Anisotropic matrix
#
# Just as for the 2nd-order Hill tensor (see @sec-hill2), the crack resistivity is computed analytically for any matrix anisotropy.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

# Orthotropic conductivity
K_aniso = tensor([3.2, 0.5, 1./(3.2*0.5)], angles=[math.pi/3, math.pi/4, math.pi/5])

H_aniso = crack_compliance(spheroidal(1.e-4), K_aniso)
print("H (crack resistivity, orthotropic) =\n", H_aniso)
# -

# $\,$
