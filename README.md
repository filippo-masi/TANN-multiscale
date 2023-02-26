## Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)

- repository under construction ( * )

### 1. Micromechanical simulations using the Finite Element Method
``` lattice ```

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


### 2. Multiscale simulation with TANN
( * ) For running part of the code Numerical Geolab* software is needed. The software is currently under review and will be uploaded online soon.
- A Stathas and I Stefanou. Numerical Geolab, FEniCS for inelasticity. In The FEniCS Conference, 2022.

For more information, please [contact me](mailto:filippo.masi@sydney.edu.au)



### References

If you use this code, please cite the related papers:

F Masi, I Stefanou (2022). "[Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)](https://doi.org/10.1016/j.cma.2022.115190)". Computer Methods in Applied Mechanics and Engineering 398, 115190.


    @article{masi2022multiscale,
    title={Multiscale modeling of inelastic materials with Thermodynamics-based Artificial Neural Networks (TANN)},
    author={Masi, Filippo and Stefanou, Ioannis},
    journal={Computer Methods in Applied Mechanics and Engineering},
    volume={398},
    pages={115190},
    year={2022},
    publisher={Elsevier},
    doi={10.1016/j.cma.2022.115190}



