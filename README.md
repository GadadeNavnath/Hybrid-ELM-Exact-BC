# A-hybrid-ELM-framework-with-exact-boundary-enforcement-for-irregular-domains
This repository contains reproducibility materials for the manuscript:
Title: “A Hybrid Extreme Learning Machine Framework with Exact
Boundary Enforcement for Irregular Domains”

Authors: Gadade Navnath Ankush and Sivaram Ambikasaran
This repository provides Python implementations for solving boundary value problems with exact enforcement of boundary conditions. The code is organized into four benchmark problems, covering linear and nonlinear differential equations with reguale and irregualar domains, including a three-dimensional example.

Requirements
The code is implemented in Python and requires the following packages:

Python 3.12 or later

numpy

matplotlib

scipy

shapely

Problems 3 additionally require:

geopandas

Install the required packages using:

pip install numpy matplotlib shapely geopandas scipy

## Problem Overview

| Problem   | Case   | BC Type   | Equation Class | Domain    |
| --------- | ------ | --------- | -------------- | --------- |
| Problem 1 | Case 1 | Dirichlet | Nonlinear      | Regular   |
|           | Case 2 | Mixed     | Nonlinear      | Regular   |
| Problem 2 | Case 1 | Dirichlet | Nonlinear      | Regular   |
|           | Case 2 | Neumann   | Nonlinear      | Regular   |
|           | Case 3 | Mixed     | Nonlinear      | Regular   |
| Problem 3 | Case 1 | Dirichlet | Linear         | Irregular |
|           | Case 2 | Neumann   | Linear         | Irregular |
|           | Case 3 | Mixed     | Linear         | Irregular |
| Problem 4 | Case 1 | Dirichlet | Nonlinear      | Irregular |
|           | Case 2 | Neumann   | Nonlinear      | Irregular |
|           | Case 3 | Mixed     | Nonlinear      | Irregular |
