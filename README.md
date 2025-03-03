# echoes

<p align="center">
<img src="./img/cover.png" alt="Description" width="250">
</p>

Manual of `echoes` (**E**xtended **C**alculator of **HO**mog**E**nization **S**chemes) library available [here](https://jfbarthelemy.github.io/echoes).

The library `echoes` allows to implement various mean-field homogenization schemes of random media involving different types of heterogeneities in the framework of elasticity, conductivity, viscoelasticity as well as nonlinear homogenization.

This book gathers tutorials presenting the main features of the library:

- elements of tensor calculus,
- Hill and Eshelby tensors and their derivatives with respect to reference medium moduli,
- concentration problems,
- RVEs and schemes in linear homogenization,
- extension to nonlinear homogenization,
- extension to linear time-dependent behaviors.

## Download

The core of `echoes` has been developed in C++ and wrapped by a Python interface. Hence its use requires first the installation of a Python environment including `pip` executable (for instance [Anaconda](https://www.anaconda.com/products/distribution)).

Wheel packages can be downloaded for various versions of Python under Windows or Linux by choosing the appropriate file for your configuration under the link [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14959866.svg)](https://doi.org/10.5281/zenodo.14959866)

Once in possession of the relevant `.whl` file, the package can be installed in a console (Anaconda console or any console allowing to run `pip`) by

```python
pip install -U echoes-XYZ.whl
# replacing echoes-XYZ.whl by the correct path to the whl file
```

## Citation

If you use `echoes`, please cite it as

```
Barthélémy, Jean-François, 2022. Echoes: Extended Calculator of HOmogEnization Schemes. https://doi.org/10.5281/ZENODO.14959866
```

or in `bibtex` style

```tex
@software{echoes,
  title = {Echoes: {{Extended Calculator}} of {{HOmogEnization Schemes}}},
  shorttitle = {Echoes},
  author = {Barthélémy, Jean-François},
  date = {2022-11-22},
  doi = {10.5281/ZENODO.14959866},
  url = {https://zenodo.org/record/14959866},
  organization = {Zenodo},
  version = {v1.0.0},
}
```

----------------
Powered by [Quarto](https://quarto.org/).
