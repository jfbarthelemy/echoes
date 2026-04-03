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

# # ALV Hill (polarization) tensor kernel {#sec-hill_alv}
#
# The elastic Hill (polarization) tensor $\mathbb{P}_\mathcal{E}(\mathbb{C})$ for an ellipsoidal inclusion $\mathcal{E}$ embedded in an elastic medium with stiffness $\mathbb{C}$ is defined in @sec-hill_elas. This appendix establishes its counterpart in **ageing linear viscoelasticity (ALV)**: the Hill tensor *kernel* $\uuuu{P}_\mathcal{E}(t,t')$ and the key block property that makes ALV homogenization tractable.
#
# ## ALV Hill tensor kernel formula
#
# Consider an ellipsoidal inclusion $\mathcal{E}$ (characterized by shape tensor $\uu{A}$, see @eq-ellipsoid) embedded in an infinite ALV matrix with 4th-order relaxation kernel $\uuuu{C}(t,t')$. The response of the medium to a uniform polarization $\uu{p}(t)$ in $\mathcal{E}$ yields a uniform strain field inside the inclusion, related to the polarization by a **Volterra kernel**. This kernel is the ALV Hill tensor [@barthelemyIJSS2016]:
#
# $$
# \uuuu{P}_\mathcal{E}(t,t') = \frac{\det\uu{A}}{4\pi}
# \int_{\|\uv{\xi}\|=1}
# \frac{\uv{\xi} \sotimes \volt{(\uv{\xi}\cdot\uuuu{C}\cdot\uv{\xi})}_{(t,t')} \sotimes \uv{\xi}}{\|\uu{A}\cdot\uv{\xi}\|^3}
# \ud S_{\uv{\xi}}
# $$ {#eq-alv-hill-kernel}
#
# where:
#
# - $\volt{(\uv{\xi}\cdot\uuuu{C}\cdot\uv{\xi})}(t,t')$ is the Volterra inverse (in the sense of @sec-volterra-operator) of the acoustic tensor $\uu{K}_{\uv{\xi}}(t,t') = (\uv{\xi}\cdot\uuuu{C}\cdot\uv{\xi})(t,t')$,
# - $\sotimes$ denotes the symmetrized tensor product,
# - $\uu{A}$ is the shape tensor whose eigenvalues give the semi-axes $a\geq b\geq c$ of the ellipsoid.
#
# This formula is the exact analogue of the elastic Hill tensor (see @sec-hill_elas) with the elastic matrix inverse $(\uv{\xi}\cdot\mathbb{C}\cdot\uv{\xi})^{-1}$ replaced by the Volterra inverse $\volt{(\uv{\xi}\cdot\uuuu{C}\cdot\uv{\xi})}$. The proof rests on the Green's kernel of the ALV medium and is detailed in [@barthelemyIJSS2016, Section 3].
#
# ::: {.callout-note}
# **Connection to the elastic case.** If $\uuuu{C}(t,t') = \mathbb{C}_E\,H(t-t')$ (elastic kernel, proportional to the Heaviside function), the Volterra inverse reduces to $\volt{\uuuu{C}}(t,t') = \mathbb{C}_E^{-1}\,H(t-t')$, and @eq-alv-hill-kernel recovers $\uuuu{P}_\mathcal{E}(t,t') = \mathbb{P}_\mathcal{E}(\mathbb{C}_E)\,H(t-t')$.
# :::
#
# ## Block property {#sec-block-hill-property}
#
# The following theorem is the key computational result enabling ALV homogenization [@barthelemyIJSS2016, Theorem 1]. Over a discrete time series $t_0 < t_1 < \cdots < t_n$, let $\mathbf{R}^0$ denote the **block relaxation matrix** of the reference medium: a lower-triangular $6(n+1)\times6(n+1)$ matrix with $6\times6$ blocks, built from the trapezoidal discretization of the Volterra integral (see @sec-visco-discretization in @sec-viscoelasticity-time). The **block Hill matrix** satisfies:
#
# $$
# \mathbf{P}_\mathcal{E}(\mathbf{R}^0) = \mathbb{P}_\mathcal{E}(\mathbf{R}^0)
# $$ {#eq-block-hill-prop}
#
# In words: **the block Hill matrix is obtained by applying the elastic Hill tensor formula to the block relaxation matrix $\mathbf{R}^0$**, with $\mathbb{C}$ replaced by $\mathbf{R}^0$ in the integral formula of @sec-hill_elas and all matrix operations interpreted block-wise.
#
# **Proof sketch.** For a causal lower-triangular block matrix $\mathbf{K}$, the Volterra inverse coincides with the algebraic (block matrix) inverse: $\volt{\mathbf{K}} = \mathbf{K}^{-1}$. Applied to the acoustic block matrix $\mathbf{K}_{\uv{\xi}} = \uv{\xi}\cdot\mathbf{R}^0\cdot\uv{\xi}$, the Volterra inverse $\volt{(\uv{\xi}\cdot\uuuu{C}^0\cdot\uv{\xi})}$ appearing in @eq-alv-hill-kernel is replaced by the ordinary matrix inverse $(\uv{\xi}\cdot\mathbf{R}^0\cdot\uv{\xi})^{-1}$. Integrating over $\uv{\xi}$ then yields exactly the elastic Hill formula (see @sec-hill_elas) applied to $\mathbf{R}^0$. Full details are in [@barthelemyIJSS2016, Section 3.2].
#
# ::: {.callout-important}
# **Computational implication.** The block Hill matrix $\mathbf{P}_\mathcal{E}(\mathbf{R}^0)$ is computed by a **single application** of the elastic Hill tensor formula to the $6(n+1)\times6(n+1)$ block matrix $\mathbf{R}^0$. ALV homogenization is thus directly compatible with elastic Hill tensor routines, with the $6\times6$ stiffness simply replaced by the full block relaxation matrix.
# :::
#
# ## Isotropic ALV matrix
#
# When the matrix relaxation kernel is isotropic,
#
# $$
# \uuuu{C}(t,t') = 3k(t,t')\,\mathbb{J} + 2\mu(t,t')\,\mathbb{K},
# $$
#
# the acoustic tensor $\uv{\xi}\cdot\uuuu{C}\cdot\uv{\xi}$ is isotropic in the $(\uv{\xi}\otimes\uv{\xi},\,\uu{I}-\uv{\xi}\otimes\uv{\xi})$ basis and can be inverted analytically in the Volterra sense. Substituting into @eq-alv-hill-kernel yields [@barthelemyIJSS2016, Section 4; @barthelemyIJES2019]:
#
# $$
# \uuuu{P}_\mathcal{E}(t,t') =
# \volt{\!\left(k + \tfrac{4}{3}\mu\right)}_{(t,t')}\,\uuuu{U}^{\uu{A}}
# + \volt{\mu}_{(t,t')}\,\bigl(\uuuu{V}^{\uu{A}} - \uuuu{U}^{\uu{A}}\bigr)
# $$ {#eq-alv-hill-iso}
#
# where $\uuuu{U}^{\uu{A}}$ and $\uuuu{V}^{\uu{A}}$ are the purely **geometric** tensors of @sec-hill_elas:
#
# $$
# \uuuu{U}^{\uu{A}} = \frac{\det\uu{A}}{4\pi}\int_{\|\uv{\xi}\|=1}
# \frac{\uv{\xi}\otimes\uv{\xi}\otimes\uv{\xi}\otimes\uv{\xi}}{\|\uu{A}\cdot\uv{\xi}\|^3}\ud S_{\uv{\xi}},
# \qquad
# \uuuu{V}^{\uu{A}} = \frac{\det\uu{A}}{4\pi}\int_{\|\uv{\xi}\|=1}
# \frac{\uv{\xi}\sotimes\uuuu{I}\sotimes\uv{\xi}}{\|\uu{A}\cdot\uv{\xi}\|^3}\ud S_{\uv{\xi}}
# $$
#
# ::: {.callout-tip}
# **Time-space decoupling.** @eq-alv-hill-iso shows that the ALV Hill tensor kernel factorizes into a purely temporal part ($\volt{(k+4\mu/3)}$ and $\volt{\mu}$, scalar Volterra inverses) and a purely geometric part ($\uuuu{U}^{\uu{A}}$ and $\uuuu{V}^{\uu{A}}$, the same as in the elastic case). For a given ellipsoidal shape, the geometric tensors are computed once; only the scalar Volterra inverses carry the time dependence.
# :::
#
# **Special cases.** For a **sphere** ($\uu{A} = a\,\uu{I}$):
#
# $$
# \uuuu{U}^{\uu{A}} = \tfrac{1}{3}\mathbb{J} + \tfrac{2}{15}\mathbb{K},
# \qquad
# \uuuu{V}^{\uu{A}} = \tfrac{1}{3}\uuuu{I},
# $$
#
# so that the ALV Hill tensor kernel of a sphere in an isotropic matrix is:
#
# $$
# \uuuu{P}_{sphere}(t,t') =
# \frac{\volt{(k+4\mu/3)}_{(t,t')}}{3}\,\mathbb{J}
# + \frac{\volt{(k+4\mu/3)}_{(t,t')} + 5\volt{\mu}_{(t,t')}}{15}\,\mathbb{K}
# + \frac{\volt{\mu}_{(t,t')}}{3}\,\mathbb{J}
# $$
#
# or equivalently, in the isotropic Hill tensor block form, $[\mathbf{P}_{sphere}(\mathbf{R}^0)]_{ij}$ is the elastic Hill tensor for a sphere with bulk modulus $[\mathbf{k}^0]_{ij}$ and shear modulus $[\boldsymbol{\mu}^0]_{ij}$.
#
# For **spheroids** and general **ellipsoids** in an isotropic matrix, explicit formulas for $\uuuu{U}^{\uu{A}}$ and $\uuuu{V}^{\uu{A}}$ in terms of elliptic integrals are given in @sec-hill_elas and [@barthelemyIJES2019, Appendix A].
