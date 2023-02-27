## Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)

<center><img src="./TANN - Numerical Geolab/_images/abstract.png"  alt="centered image" width="50%" height="20%"></center>

- repository under construction ( * )

### 1. Micromechanical simulations using the Finite Element Method - ``` lattice ```


The class and scripts refer to the Finite Element (FE) code used in ([Masi, Stefanou, 2022](https://doi.org/10.1016/j.cma.2022.115190)) to generate data for training Thermdoynamics-based Artificial Neural Networks and their validation.

##### Usage

- The file ``` lattice_material.py ``` contains the classes for the constructor, assembly of lattice structures, and FE solver (Newton's method).
- ``` lattice_prescribed_path.py ``` contains the script for running the FE analysis of a lattice material unit cell, with periodic boundary conditions, given a prescribed strain increment path.
  - Constructor parameters: ```xmax, ymax, zmax``` are the total dimensions of the unit cell; ```nx, ny, nz``` are the number of nodes along each direction, and ```s``` is the magnitude of the perturbation (uniform spatial distribution) of the nodal coordinates
  - Boundary conditions: Dirichlet, Neumann, and periodic boundary conditions are implemented. The call is 
    ```sh
    BC = [nodal_degree,value,"type"]
    ```
    with ```nodal_degree``` being the degree of freedom of a particular node (i.e., node's index in ```node_coordinates``` times 3 plus 3), ```value``` the prescribed value, and ```type``` the type of boundary condition ```"DC"``` for Dirichlet, ```"NM"``` for Neumann, ```"PR"``` for periodic.
- ``` lattice_data_gen.py ``` contains the script for running the data generation, with periodic boundary conditions, given a prescribed strain increment path.
- ``` lattice_torsional.py ``` contains the script for running the FE analysis of a lattice structure with fixed bottom end and imposed torsional displacement (see [Masi, Stefanou, 2022](https://doi.org/10.1016/j.cma.2022.115190)).


### 2. Multiscale simulation with TANN - ``` TANN - Numerical Geolab ```

Hands-on: employ TANN as an user-material to perform Finite Element analyses [using Numerical Geolab, 2].
The application consists of a 3D model subjected to torsional deformations. The material used represents the volume average behavior of a lattice microstructure with bars displaying elasto-plastic rate-independent behavior, with von Mises yield criterion, and kinematic hardening. For more, we refer to [1,2].


<img src="./TANN - Numerical Geolab/_images/displacement_vertical_AI.png"  width="25%" height="20%">
         
         Torsional warping: vertical displacement field due to a torsional deformation. The displacement fields were exported with the help of the third party software Paraview.


( * ) For running part of the code Numerical Geolab [2] software is needed. The software is currently under review and will be uploaded online soon.
For more information, [contact me](mailto:filippo.masi@sydney.edu.au)



### References

If you use this code, please cite the related papers:

[1] F Masi, I Stefanou (2022). "[Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)](https://doi.org/10.1016/j.cma.2022.115190)". Computer Methods in Applied Mechanics and Engineering 398, 115190.

[2] Stathas, A. and Stefanou, I., 2022. Numerical Geolab, FEniCS for inelasticity. The FEniCS Conference.


    @article{masi2022multiscale,
    title={Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)},
    author={Masi, Filippo and Stefanou, Ioannis},
    journal={Computer Methods in Applied Mechanics and Engineering},
    volume={398},
    pages={115190},
    year={2022},
    publisher={Elsevier},
    doi={10.1016/j.cma.2022.115190}



