# Matrix operations and linear algebra (mola) library for core Python

- [Introduction](#introduction)
- [Getting started](#getting-started)
- [Prerequisites](#prerequisites)
- [Classes](#classes)
  * [Matrix](#matrix)
- [License](#license)
<!-- toc -->

## Introduction

**mola** is a Python library for doing algebra with matrices. It covers the basic operations such as matrix addition and multiplication, transposes, and inverse. Additionally, it has some miscellaneous data analytical tools for regression, clustering etc. It is written without any external Python libraries.

I wrote **mola** as a hobby project to remind myself of the linear algebra I studied in uni and to practice my Python programming. Particularly, this is an exercise in publishing my first Python library.

## Getting started

Install with "pip install mola" or download the GitHub repository for a more recent version.

See main.py for examples and read the documentation.

## Prerequisites

- Python 3.x (written on Python 3.9, so that's sure to work)

## Current features

- basic matrix operations implemented (addition, multiplication, transpose, inverse)
- matrices with labeled rows and columns
- user-friendly wrappers for function fitting (including linear least squares regression, Tikhonov regularization, Gauss-Newton iteration for nonlinear fitting)
- some basic matrix decompositions (QR decomposition and eigendecomposition)
- clustering with hard k-means and fuzzy c-means and density-based clustering (see examples/visualize_clustering.py for a demonstration)


![visualization of regression algorithms](https://github.com/jerela/mola/blob/master/examples/visualize_regression.png)
![visualization of clustering algorithms](https://github.com/jerela/mola/blob/master/examples/visualize_clustering.png)

## Classes

### Matrix

**Matrix** is the main class of **mola**. In practice, the elements of the matrix are implemented with lists. Most of its functionality involves calling methods from this class.

### LabeledMatrix

**LabeledMatrix** inherits **Matrix** and allows labeling of rows and columns, as well as overriding certain setter and getter functions to use those labels.

## TODO:
- checks to see if matrix is positive/negative definite/semidefinite
- different matrix norms (only Frobenius and Euclidean norm implemented right now)
- more decompositions (SVD)
- user-friendly wrapper for logistic regression
- preprocessing functions for matrix data (center and scale, e.g., z-scores)
- example data analysis project to showcase existing features

## License

```
MIT License

Copyright (c) 2023 Jere Lavikainen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
