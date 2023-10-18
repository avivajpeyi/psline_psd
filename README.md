# Slipper
[![pypi](https://img.shields.io/pypi/v/pyslipper.svg)](https://pypi.org/project/pyslipper/)
[![python](https://img.shields.io/pypi/pyversions/pyslipper.svg)](https://pypi.org/project/pyslipper/)
[![Build Status](https://github.com/avivajpeyi/pyslipper/actions/workflows/dev.yml/badge.svg)](https://github.com/avivajpeyi/pyslipper/actions/workflows/dev.yml)
[![Coverage Status](https://coveralls.io/repos/github/avivajpeyi/pyslipper/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/avivajpeyi/pyslipper?branch=main)


A python package to fit data with P-splines.


This implementation is built off Patricio Maturana-Russel (@pmat747)'s P-spline PSD
https://github.com/pmat747/psplinePsd

![](docs/static/logo.png)


* Documentation: <https://avivajpeyi.github.io/pyslipper>
* GitHub: <https://github.com/avivajpeyi/pyslipper>
* PyPI: <https://pypi.org/project/pyslipper/>
* Free software: MIT


## Developer setup

```
pip install -e .[dev]
pre-commit install
```

Push to branches -- ensure the workflow passes, and then merge to main.


To make a release:
```bash
git tag -a vX.Y.Z -m "Version X.Y.Z release: <description>"
git push origin --tags
```
