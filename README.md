# Matrix operations and linear algebra library for vanilla Python (molalib)

- [Introduction](#introduction)
- [Getting started](#getting-started)
- [Prerequisites](#prerequisites)
- [Classes](#classes)
  * [Matrix](#matrix)
- [License](#license)
<!-- toc -->

## Introduction

**molalib** is a Python library for doing algebra with matrices. It covers the basic operations such as matrix addition and multiplication, transposes, inverse, different norms and some decompositions. It is built from scratch without using any external libraries.

I wrote **molalib** as a hobby project to remind myself of the linear algebra methods I studied in uni and to practice my Python programming.

## Getting started

WIP

## Prerequisites

Python 3.9

## Classes

### Matrix

Represents a standard mathematical matrix with any number of rows and columns.

**Matrix**(rows,columns,value)
- *rows*: the number of rows (height) in the matrix
- *columns*: the number of columns (width) in the matrix
- *value*: the initial numeric value that is assigned to all elements in the matrix (default: 0)


**set**(i,j,value)
- *i*: index of row (0 being the first row)
- *j*: index of column (0 being the first column)
- *value*: the numeric value that is assigned to the element at the given indices


## TODO:
- checks to see if matrix is positive/negative definite/semidefinite
- matrix norms
- class wrapping linear least squares in an approachable interface
- generalized least squares
- Gauss-Markov
- Levenberg-Marquardt
- regularized least squares
- simple decompositions
- support for complex numbers

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
