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

PyTorch

Problems 3 additionally require:

geopandas

Install the required packages using:

pip install numpy matplotlib shapely geopandas scipy torch

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

For nonlinear PDEs, the solution procedure involves two stages, which are described below. The proposed framework employs two Gauss–Newton procedures. The first is used in Stage 1 to obtain the IELM approximation, starting from the coefficients obtained from the corresponding linearized problem. The second is used in Stage 2 to enforce the boundary conditions exactly, taking the converged Stage 1 coefficients as the initial iterate. For linear PDEs, the framework simplifies to a single least-squares solve.

#### Stage 1: IELM Approximation

An initial solution is obtained using the Legendre-IELM framework. By collocating the governing equation and boundary conditions, an overdetermined nonlinear system is formed. The resulting nonlinear least-squares problem is solved using a Gauss–Newton procedure, initialized with the coefficients obtained from the corresponding linearized problem. The converged coefficients provide the Stage 1 approximation.

#### Stage 2: Exact Satisfaction of Boundary Conditions

Starting from the Stage 1 solution, the coefficients are further refined while enforcing the boundary conditions exactly. The boundary-enforcing coefficients are expressed as functions of the free coefficients, thereby reducing the problem to a nonlinear system involving only the free coefficients. A Gauss–Newton procedure is then applied to solve this reduced system. Since the boundary conditions are satisfied through the coefficient representation itself, they remain exactly satisfied throughout the iteration process. In practice, Stage 1 provides an excellent initial guess, and Stage 2 typically converges in one or a few Gauss–Newton iterations.


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

In addition, each problem folder contains a dedicated README file describing the mathematical problem, available boundary-condition configurations, dataset generation, required execution order, and instructions for running the code.

Figures used in the paper are provided separately in the `Figures/` folder.

### Reproducibility

This repository provides all code and data required to reproduce the numerical results presented in the paper.

Within each problem folder, the boundary-condition folders contain two scripts: main.py, which implements the numerical method, and domain.py, which generates the required datasets. All parameter settings are specified directly within these scripts.

---

### License

This project is licensed under the MIT License.
