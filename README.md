# echoes

Manual of **ECHOES** (**E**xtended **C**alculator of **HO**mog**E**nization **S**chemes) library available [here](https://jfbarthelemy.github.io/echoes).

The library **ECHOES** allows to implement various homogenization schemes involving different types of heterogeneities in the framework of elasticity, conductivity, viscoelasticity as well as tools to properly calculate the derivatives of macroscopic stiffness with respect to lower scale moduli (fundamental tool of the modified secant method in nonlinear homogenization).

This manual aims at recalling some fundamental aspects of the theory of homogenization of random media along with a presentation of the main features of the library **ECHOES** as well as code examples.

## Download

The core of **Echoes** has been developed in C++ and wrapped by a Python interface. Hence its use requires first the installation of a Python environment with `pip` executable (for instance [Anaconda](https://www.anaconda.com/products/distribution)).

Wheel packages can be downloaded for Python 3.7, 3.8, 3.9, 3.10 and 3.11 under Windows or Linux by choosing the appropriate file for your configuration under the link [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10559657.svg)](https://doi.org/10.5281/zenodo.10559657)

Once in possession of the relevant `.whl` file, the package can be installed in a console (Anaconda console or any console allowing to run `pip`) by

```python
pip install -U echoes-XYZ.whl # replacing echoes-XYZ.whl by the correct path to the whl file
```

## Citation

If you use **Echoes**, please cite it as

```
Barthélémy, Jean-François, 2022. Echoes: Extended Calculator of HOmogEnization Schemes. https://doi.org/10.5281/ZENODO.10559657
```

or in `bibtex` style

```
@misc{Echoes,
 author  = {Barthélémy, Jean-François},
 title   = {Echoes library (Extended Calculator of HOmogEnization Schemes)},
 url     = {https://jfbarthelemy.github.io/echoes/},
 doi     = {10.5281/zenodo.10559657}
 version = {v1.0.0},
 year    = {2022},
 month   = {11}
}
```

----------------
Created thanks to [Quarto](https://quarto.org/).
