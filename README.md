# Matrix operations and linear algebra (mola) library for vanilla Python

- [Introduction](#introduction)
- [Getting started](#getting-started)
- [Prerequisites](#prerequisites)
- [Classes](#classes)
  * [Matrix](#matrix)
- [License](#license)
<!-- toc -->

## Introduction

**mola** is a Python library for doing algebra with matrices. It covers the basic operations such as matrix addition and multiplication, transposes, and inverse. It is written without any external Python libraries.

I wrote **mola** as a hobby project to remind myself of the linear algebra I studied in uni and to practice my Python programming. Particularly, this is an exercise in publishing my first Python library.

## Getting started

Install with "pip install mola" or the GitHub repository for a more recent version.

See main.py for examples and read the documentation.

## Prerequisites

- Python 3.x (written on Python 3.9, so that's sure to work)

## Classes

### Matrix

**Matrix** is the main class of **mola**. In practice, the elements of the matrix are implemented with lists. Most of its functionality involves calling methods from this class. However, for a more user-friendly approach, I plan on implementing some wrapper functions you can call directly (e.g., "linear_least_squares()" to perform linear regression).

## TODO:
- checks to see if matrix is positive/negative definite/semidefinite
- different matrix norms (only Frobenius and Euclidean norm implemented right now)
- Gauss-Newton iteration
- Levenberg-Marquardt algorithm
- regularized least squares
- more decompositions (only QR implemented so far)

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
