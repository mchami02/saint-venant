"""
Main entry point for 1D Saint-Venant (shallow water) equations solver.

This solver implements finite volume methods with various numerical fluxes
(Lax-Friedrichs, Rusanov, HLL) and time integration schemes (Explicit Euler, RK2).

Usage:
    python main.py <parameter_file> [options]
    
Examples:
    python main.py parameters.json
    python main.py parameters.json --output my_solution.h5
"""

import sys
from solver.data_file import DataFile
from solver.mesh import Mesh
from solver.physics import Physics
from solver.finite_volume import LaxFriedrichs, Rusanov, HLL
from solver.time_scheme import ExplicitEuler, RK2


def main():
    """Main function to run the shallow water solver."""
    
    # Verbosity level (0=silent, 1=normal, 2=verbose)
    VERBOSITY = 1
    
    # -------------------------------------------------------
    # Check command line arguments
    # -------------------------------------------------------
    if len(sys.argv) < 2:
        print("\033[91mPlease, enter the name of your JSON parameter file.\033[0m")
        print("Usage: python main.py <parameters.json> [options]")
        print("Options:")
        print("  --output FILE    Specify custom output filename (saved in results_dir)")
        print("\nExamples:")
        print("  python main.py parameters.json")
        print("  python main.py parameters.json --output my_simulation.h5")
        print("\nNote: Results are always saved in the directory specified in the parameter file")
        sys.exit(-1)
    
    # Parse optional output filename
    output_filename = None
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_filename = sys.argv[idx + 1]
    
    # -------------------------------------------------------
    # Read parameter file and initialize
    # -------------------------------------------------------
    data_file = DataFile(sys.argv[1])
    data_file.read_data_file(verbosity=0)  # Silent
    
    # -------------------------------------------------------
    # Print header and parameters
    # -------------------------------------------------------
    print("Solving 1D St-Venant equations with the following parameters")
    print('=' * 50)
    data_file.print_data()
    
    # -------------------------------------------------------
    # Build mesh
    # -------------------------------------------------------
    mesh = Mesh(data_file)
    mesh.initialize(verbosity=0)  # Silent
    
    # -------------------------------------------------------
    # Initialize physics (IC, BC, source terms)
    # -------------------------------------------------------
    physics = Physics(data_file, mesh)
    physics.initialize(verbosity=0)  # Silent
    
    # -------------------------------------------------------
    # Select numerical flux scheme
    # -------------------------------------------------------
    flux_name = data_file.numerical_flux
    
    if flux_name == "LaxFriedrichs":
        fin_vol = LaxFriedrichs(data_file, mesh, physics)
    elif flux_name == "Rusanov":
        fin_vol = Rusanov(data_file, mesh, physics)
    elif flux_name == "HLL":
        fin_vol = HLL(data_file, mesh, physics)
    else:
        print(f"\033[91mERROR::FINITEVOLUME : Case {flux_name} not implemented.\033[0m")
        sys.exit(-1)
    
    # -------------------------------------------------------
    # Select time integration scheme
    # -------------------------------------------------------
    time_scheme_name = data_file.time_scheme
    
    if time_scheme_name == "ExplicitEuler":
        time_scheme = ExplicitEuler(data_file, mesh, physics, fin_vol)
    elif time_scheme_name == "RK2":
        time_scheme = RK2(data_file, mesh, physics, fin_vol)
    else:
        print(f"\033[91mERROR::TIMESCHEME : Case {time_scheme_name} not implemented.\033[0m")
        sys.exit(-1)
    
    # -------------------------------------------------------
    # Solve
    # -------------------------------------------------------
    print('=' * 50)
    time_scheme.solve(verbosity=0, output_filename=output_filename)
    
    # -------------------------------------------------------
    # Success message
    # -------------------------------------------------------
    print("\033[92mSolved\033[0m")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
