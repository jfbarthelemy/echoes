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
#         href: user_inclusion.py
#       - text: Jupyter notebook
#         icon: journal-code
#         href: user_inclusion.ipynb
# ---
#
# # User-defined inclusions {#sec-user_inclusion}
#
# ::: {.callout-important icon=false}
#
# ## {{< iconify pajamas issue-type-objective >}} Objectives
#
# This tutorial introduces the `user_inclusion` class, which allows defining an arbitrary inclusion type by providing the concentration tensors analytically (or numerically). It shows how to subclass `user_inclusion`, implement the `build_all()` method, and integrate the result into standard homogenization schemes.
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

np.set_printoptions(precision=8, suppress=True)


# -

# :::
#
# ## Motivation {#sec-user-motivation}
#
# The `ellipsoid` class presented in @sec-concentration_tensors covers ellipsoidal inclusions for which Eshelby's uniformity result holds: the four concentration tensors derive solely from the Hill polarization tensor $\uuuu{P}$, computed automatically by *Echoes* for a wide range of matrix symmetries. The `sphere_nlayers` class extends coverage to concentrically layered spheres through the analytical solution of [@herve1993].
#
# Beyond these two built-in types, one may need to handle:
#
# - An inclusion whose shape is **not ellipsoidal and not a concentric sphere** (e.g. a composite granule, a non-convex shape, a polyhedral grain).
# - A case where the concentration tensors have been **determined externally**, for instance by a finite-element simulation of the auxiliary problem, by a dedicated semi-analytical formula, or by a numerical integration.
# - An **approximate analytical formula** hand-crafted for a specific geometry or interface type.
#
# The `user_inclusion` class covers all these situations. The user subclasses it and overrides `build_all()` to supply the four concentration tensors by any means. The resulting object is a first-class citizen in any *Echoes* homogenization scheme.
#
# ## Class interface {#sec-user-interface}
#
# ### Defining a subclass
#
# A `user_inclusion` is created by inheriting from it, registering properties in the constructor, and implementing `build_all()`:
#
# ```python
# class MyInclusion(user_inclusion):
#     def __init__(self, prop, fraction=0., symmetrize=[]):
#         super().__init__(fraction, symmetrize)
#         for key, val in prop.items():
#             user_inclusion.set_prop(self, key, val)   # note: explicit self
#
#     def build_all(self):
#         # compute and return the four concentration tensors
#         ...
#         return {"eE": eE, "eS": eS, "sE": sE, "sS": sS}
# ```
#
# The constructor arguments mirror those of `ellipsoid`:
#
# | Argument | Description |
# |---|---|
# | `fraction` | Volume fraction in the RVE (can also be set later via `.fraction`) |
# | `symmetrize` | List of symmetry classes to project the concentration tensors onto (e.g. `[ISO]`), or empty list |
#
# Properties are registered with `user_inclusion.set_prop(self, key, value)`. The explicit `self` is necessary because of Python's method-resolution order when calling a parent method from within a subclass.
#
# ### Attributes available inside `build_all()`
#
# When the homogenization scheme calls `build_all()`, the following attributes are set on `self`:
#
# | Attribute | Type | Description |
# |---|---|---|
# | `self.ref` | `tensor_param` | Reference medium (the embedding matrix) |
# | `self.refname` | `str` | Property name (e.g. `"C"`) |
# | `self.prop(self.refname)` | `tensor_param` | Property tensor of this inclusion |
#
# Both `self.ref` and `self.prop(key)` are `tensor_param` objects exposing the same fields as any other property tensor in *Echoes* (`.array`, `.param`, `.kmu`, …).
#
# ### Required return value
#
# `build_all()` must return a dictionary with the four $6\times6$ concentration matrices (in Kelvin-Mandel representation):
#
# | Key | Tensor | Relation |
# |---|---|---|
# | `"eE"` | $\uuuu{A}^E$ | $\langle\eps\rangle = \uuuu{A}^E:\E$ |
# | `"eS"` | $\uuuu{A}^S$ | $\langle\eps\rangle = \uuuu{A}^S:\Sig$ |
# | `"sE"` | $\uuuu{B}^E$ | $\langle\sig\rangle = \uuuu{B}^E:\E$ |
# | `"sS"` | $\uuuu{B}^S$ | $\langle\sig\rangle = \uuuu{B}^S:\Sig$ |
#
# These follow the conventions of @sec-four-concentration:
#
# $$
# \uuuu{A}^S = \uuuu{A}^E:\uuuu{C}_0^{-1}, \qquad
# \uuuu{B}^E = \uuuu{C}^\mathcal{I}:\uuuu{A}^E, \qquad
# \uuuu{B}^S = \uuuu{C}^\mathcal{I}:\uuuu{A}^S
# $${#eq-conc-relations}
#
# ## Example: sphere in an isotropic matrix {#sec-user-sphere}
#
# As a concrete worked example, we re-implement the standard spherical inhomogeneity by hand, using the analytical formula for the Hill polarization tensor.
#
# ### Hill polarization tensor for a sphere
#
# For a **spherical** inclusion embedded in an **isotropic** matrix $\uuuu{C}_0 = 3k_0\,\mathbb{J}+2\mu_0\,\mathbb{K}$, the Hill polarization tensor reduces to an isotropic tensor with closed-form scalar coefficients [@eshelby1957; @hill1965]:
#
# $$
# \uuuu{P} = \frac{1}{3k_0+4\mu_0}
# \left(\mathbb{J} + \frac{3(k_0+2\mu_0)}{5\mu_0}\,\mathbb{K}\right)
# $${#eq-P-sphere-iso}
#
# or equivalently, using the notation $p_J = \frac{1}{3(3k_0+4\mu_0)}$ for the volumetric coefficient and $p_K = \frac{3(k_0+2\mu_0)}{5\mu_0(3k_0+4\mu_0)}$ for the deviatoric one.
#
# Once $\uuuu{P}$ is known, the four concentration tensors follow directly from @eq-ell-concentration and @eq-conc-relations with $\delta\uuuu{C} = \uuuu{C}^\mathcal{I}-\uuuu{C}_0$:
#
# $$
# \uuuu{A}^E = \left(\uuuu{I}+\uuuu{P}:\delta\uuuu{C}\right)^{-1}
# $${#eq-Ae-user}
#
# ### Implementation
#
# The `J4`, `K4`, and `Id4` constants exported by *Echoes* are the $6\times6$ Kelvin-Mandel representations of the volumetric projector $\mathbb{J}$, deviatoric projector $\mathbb{K}$, and symmetric identity $\mathbb{I}$, respectively.

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

class SphereInISO(user_inclusion):
    """Spherical inhomogeneity in an isotropic matrix.
    Concentration tensors computed analytically via the Hill
    polarization tensor formula (Eq. eq-P-sphere-iso)."""

    def __init__(self, prop, fraction=0., symmetrize=[]):
        super().__init__(fraction, symmetrize)
        for key, val in prop.items():
            user_inclusion.set_prop(self, key, val)

    def build_all(self):
        # --- reference medium ---
        C0      = self.ref.array
        S0      = np.linalg.inv(C0)
        k0, mu0 = self.ref.kmu
        # --- inclusion stiffness ---
        Ci = self.prop(self.refname).array
        # --- Hill tensor for sphere in isotropic matrix (@eq-P-sphere-iso) ---
        P  = 1. / (3*k0 + 4*mu0) * (J4 + 3.*(k0 + 2*mu0) / (5*mu0) * K4)
        # --- concentration tensors ---
        eE = np.linalg.inv(Id4 + P.dot(Ci - C0))
        eS = eE.dot(S0)
        sE = Ci.dot(eE)
        sS = Ci.dot(eS)
        return {"eE": eE, "eS": eS, "sE": sE, "sS": sS}


# -

# ### Verification against `ellipsoid`
#
# For a spherical inclusion, `SphereInISO` must reproduce the same concentration tensors as `ellipsoid(shape=spherical)`, whose Hill tensor is computed automatically by *Echoes*:

# +
#| error: false
#| warning: false
#| code-fold: false
#| include: true

Ci = stiff_Enu(5., 0.25)    # inclusion: E=5, nu=0.25
C0 = stiff_kmu(3., 2.)      # matrix: k=3, mu=2

# Built-in ellipsoid (Echoes computes P internally)
ell = ellipsoid(shape=spherical, prop={"C": Ci})
ell.set_ref("C", C0)

# User-defined sphere (P computed by hand in build_all)
usr = SphereInISO(prop={"C": Ci})
usr.set_ref("C", C0)

for name, ref, usr_val in [
        ("eE", ell.eE, usr.eE),
        ("eS", ell.eS, usr.eS),
        ("sE", ell.sE, usr.sE),
        ("sS", ell.sS, usr.sS)]:
    err = np.max(np.abs(ref - usr_val))
    print(f"Max |ell.{name} - usr.{name}| = {err:.2e}")
# -

# $\,$
