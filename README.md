# Methods for Numerical Mathematics ( Methoden zur Numerischen Mathematik )
This is the final project work of Ilya Gutkin for the computational lab course. The project focuses on the numerical solution of partial differential equations (PDEs).  Its core objective is to solve convection-dominated linearized Navier–Stokes equations on space–time meshes and to verify whether the analytically derived stabilization methods withstand numerical testing.To handle problems where the spatial domain is three-dimensional, one must work with four-dimensional meshes (three spatial dimensions plus time). Since no existing implementation of such a solver was known to the author, this project aims to fill that gap and provide a PDE solver capable of handling fully four-dimensional problems.

## Project structure

The repository is organized into several main directories:

- [lectures](lectures/):  
  This folder contains all mathematical background material needed to understand the architecture of the code. 
  It is intended as a compact reference for the theoretical foundations used throughout the project.  
  For example, the file also contains a detailed description of the 4D [mesh construction](lectures/4d_mesh_construction.ipynb).
  In particular, the document [FEM Introduction](lectures/intro_fem.ipynb/) provides a concise summary of the finite element method and is highly recommended reading for anyone who wants to follow the implementation in detail.These lectures make no claim to completeness; they are intended only as a broad overview of the mathematical foundations relevant to the project.


- [demos](demo/):  
  This folder provides a collection of self-contained demonstration programs.  
  Each finite element space has its own dedicated demo, since the details of their implementation differ slightly.  
  The demos are designed to form the conceptual backbone for solving a wide range of PDEs:
  they typically start with a simple Poisson problem but can be freely modified and extended by the reader.  
  A solid understanding of these demos equips the user to set up and solve almost arbitrary PDEs using the library.

- [`src/`](src/methodsnm/):  
  This directory contains the full source code and all core building blocks of the solver, including:
  - a mesh constructor that stores the number of elements, the vertex data, and further geometric properties of the domain,  
  - a finite element space (FES) which takes a mesh as input and encodes the essential global information such as degrees of freedom and function spaces,  
  - a family of finite elements that the FES selects as its local building blocks.  

  Each finite element is defined on a reference element and is responsible for tasks such as:
  - evaluating basis functions and their derivatives,  
  - storing the local basis,  
  - providing all data specific to the reference simplex (for example, the reference triangle or tetrahedron).

  In addition, [src](src/methodsnm/) contains several helper modules that handle numerical integration, the solution of linear systems, and visualization.  
  Their usage is illustrated in the demos and they are intended to be reused directly by the user.

- `mesh function` classes:  
  A central component of the implementation is the class used to represent functions on the mesh.  
  In the finite element context, functions are not stored in their usual analytical form but via their degrees of freedom on the underlying mesh.  
  The mesh function class provides this representation and offers an interface for evaluation and post-processing.  
  A detailed explanation of this design can be found in the corresponding source file and is an important prerequisite for understanding how functions are handled internally in the FEM framework.

  In addition, two subdirectories within the `demo/` folder are of particular importance:

- [convection–diffusion:](demos/convection-diffusion/)  
  This directory contains examples of convection-dominated problems.  
  These demos clearly illustrate the limitations of a naive weak formulation and show where standard Galerkin methods begin to fail.  
  They motivate the need for stabilization techniques such as SUPG or other residual-based methods.  
  The examples here serve as a natural introduction to stabilization and provide practical insight into when and why these techniques become indispensable.

- [linearized-navierstokes:](linearized_Navier_Stokes)  
  This directory focuses on the linearized Navier–Stokes equations.  
  Similar stabilization issues arise here, especially when using equal-order finite element pairs such as the P1–P1 pairing.  
  The demos demonstrate how stabilized formulations must be incorporated to obtain reliable solutions for incompressible flow problems on space–time meshes.

Together, these two demo sections form the conceptual link between scalar and vector-valued PDEs and touch the essential problem to find stabilization techniques for Navier-Stokes equations.  
A further goal of the project is the development of a SUPG-like stabilization term for the linearized Navier–Stokes equations.  
The analytical derivation of this term has already been completed, and an implementation of the corresponding stabilized formulation will be added in the near future.  
This extension aims to provide a robust and fully functional stabilized solver for space–time Navier–Stokes problems, complementing the existing framework for time-dependent convection–diffusion and other convection-dominated PDEs.

- Finally, the project includes an extensive collection of Pytests that verify the correctness of the individual components.  These tests are intended not only to ensure reliability but also to provide the reader with confidence that every implemented part of the framework behaves as expected. All implemented modules currently pass the full test suite. Additional tests have been prepared for future extensions of the solver.

The test suite can be executed from the terminal using:

```bash
python3 -m pytest -vv
```

## Documentation

A complete API documentation has been generated using **pdoc** and is included in the `docs/` directory.  
It contains detailed descriptions of all modules, classes, and functions that make up the finite element framework.

To access the documentation, simply open this code:

```bash
xdg-open docs/index.html
```

### Configuring CI/CD in GWDG gitlab:
Once you have forked this repository your own GWDG Gitlab project should automatically have CI/CD and gitlab pages deployment, container registry, etc.. active in its project settings so it should be able to run the CI routines for enabling Jupyterlite. 

