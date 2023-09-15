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
   Copyright 2023 Jere Lavikainen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

```
