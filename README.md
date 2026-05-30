### A-hybrid-ELM-framework-with-exact-boundary-enforcement-for-irregular-domains
This repository contains reproducibility materials for the manuscript:

**Title:** “A Hybrid Extreme Learning Machine Framework with Exact Boundary Enforcement for Irregular Domains”

**Authors:** Gadade Navnath Ankush and Sivaram Ambikasaran

This repository provides Python implementations for solving boundary value problems with exact enforcement of boundary conditions. The code is organized into four benchmark problems, covering linear and nonlinear partial differential equations with regular and irregualar domains, including a three-dimensional example.

---
### Requirements
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

---

### Problem Overview

| Problem   | Case   | BC Type   | Equation Class | Domain    |
| --------- | ------ | --------- | -------------- | --------- |
| Problem 1 | Case 1 | Dirichlet | Linear         | Regular   |
|           | Case 2 | Mixed     | Linear         | Regular   |
| Problem 2 | Case 1 | Dirichlet | Nonlinear      | Irregular |
|           | Case 2 | Neumann   | Nonlinear      | Irregular |
|           | Case 3 | Mixed     | Nonlinear      | Irregular |
| Problem 3 | Case 1 | Dirichlet | Nonlinear      | Irregular |
|           | Case 2 | Neumann   | Nonlinear      | Irregular |
|           | Case 3 | Mixed     | Nonlinear      | Irregular |
| Problem 4 | Case 1 | Dirichlet | Nonlinear      | Irregular |
|           | Case 2 | Neumann   | Nonlinear      | Irregular |
|           | Case 3 | Mixed     | Nonlinear      | Irregular |

**BC Type** denotes the type of boundary conditions imposed on the problem:

- **Dirichlet**: all boundary conditions are of Dirichlet type.
  
- **Neumann**: all boundary conditions are of Neumann type.
  
- **Mixed**: a combination of Dirichlet and Neumann boundary conditions.

---
### Methodology

The solution of the differential equations is obtained using a two-step procedure based on the Legendre-IELM framework combined with Gauss–Newton iterations. In the first step, an initial approximation is constructed by forming an overdetermined linear system using collocation points and basis functions, and solving it in the least-squares sense to obtain the initial coefficient vector. In the second step, the nonlinear problem is solved iteratively using a Gauss–Newton method: at each iteration, the residual and its Jacobian with respect to the coefficients are evaluated, a linear least-squares problem is solved to compute the update, and the coefficient vector is refined until the residual norm satisfies a prescribed tolerance or the maximum number of iterations is reached. For Problem 4 (linear PDE), the solution is obtained directly from the initial least-squares formulation, and no Gauss–Newton iteration is required.

---

### Data for Irregular Domains

For Problem 3, the physical domain corresponds to **Sweden**.

The repository includes the preprocessing scripts used to generate the geometry dataset and collocation points used in the computations. Generated datasets are stored in the `data/` folder.

Preprocessed and rescaled geometry information is based on the original dataset obtained from the GADM database:

https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SWE_0.json

---

### Repository Structure

The repository is organized using a main folder named `problems`, which contains separate folders for each benchmark problem. Within each problem folder, subfolders corresponding to the boundary-condition configurations are provided.

```text
problems/
├── problem1/
│   ├── Dirichlet_BCs/
│   └── Mixed_BCs/
├── problem2/
│   ├── Dirichlet_BCs/
│   ├── Neumann_BCs/
│   └── Mixed_BCs/
├── problem3/
│   ├── Dirichlet_BCs/
│   ├── Neumann_BCs/
│   └── Mixed_BCs/
└── problem4/
    ├── Dirichlet_BCs/
    ├── Neumann_BCs/
    └── Mixed_BCs/
```

Each boundary-condition folder includes:

* Python scripts for implementation
* A `data/` folder in which datasets are automatically generated and stored upon execution of the corresponding scripts
* A dedicated README file describing the mathematical problem, required execution order, dataset generation, and instructions for running the code

Figures used in the paper are provided separately in the `Figures/` folder.

---

### Reproducibility

This repository provides all code and data required to reproduce the numerical results presented in the paper.

Each problem folder contains the implementation scripts, dataset generation codes, and execution instructions associated with that problem. All parameter settings are defined directly within the scripts.

---

### License

This project is licensed under the MIT License.
