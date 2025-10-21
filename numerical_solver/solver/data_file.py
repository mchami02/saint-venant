"""
DataFile class for reading and managing simulation parameters.

Reads configuration from a JSON file and provides access to all simulation parameters.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataFile:
    """Manages all simulation parameters from input file.
    
    Attributes:
        file_name: Path to the parameter file
        results_dir: Directory where results will be saved
        is_save_final_time_only: Save only final timestep
        save_frequency: Frequency of solution saves (in iterations)
        n_probes: Number of water height measurement probes
        probes_references: Reference indices for probes
        probes_positions: Physical positions of probes
        is_test_case: Whether this is a test case with known solution
        test_case: Name of the test case
        initial_condition: Type of initial condition
        init_file: File containing initial conditions
        initial_height: Initial water height
        initial_discharge: Initial discharge
        xmin: Minimum x coordinate
        xmax: Maximum x coordinate
        dx: Spatial step size
        Nx: Number of cells
        numerical_flux: Numerical flux scheme name
        scheme_order: Order of the scheme (1 or 2)
        time_scheme: Time integration scheme
        initial_time: Start time
        final_time: End time
        time_step: Time step size
        CFL: CFL number
        g: Gravity acceleration
        left_BC: Left boundary condition type
        right_BC: Right boundary condition type
        left_BC_data_file: Data file for left BC
        right_BC_data_file: Data file for right BC
        left_BC_imposed_height: Imposed height at left boundary
        left_BC_imposed_discharge: Imposed discharge at left boundary
        right_BC_imposed_height: Imposed height at right boundary
        right_BC_imposed_discharge: Imposed discharge at right boundary
        is_topography: Whether topography is present
        topography_type: Type of topography
        topography_file: File containing topography data
    """
    
    file_name: str = ""
    is_save_final_time_only: bool = False
    save_frequency: int = 1
    n_probes: int = 0
    probes_references: List[int] = field(default_factory=list)
    probes_positions: List[float] = field(default_factory=list)
    is_test_case: bool = False
    test_case: str = "None"
    initial_condition: str = "none"
    init_file: str = ""
    initial_height: float = 0.0
    initial_discharge: float = 0.0
    xmin: float = 0.0
    xmax: float = 1.0
    dx: float = 0.1
    Nx: int = 10
    numerical_flux: str = "LaxFriedrichs"
    scheme_order: int = 1
    time_scheme: str = "ExplicitEuler"
    initial_time: float = 0.0
    final_time: float = 1.0
    time_step: float = 0.01
    CFL: float = 0.9
    g: float = 9.81
    left_BC: str = "Neumann"
    right_BC: str = "Neumann"
    left_BC_data_file: str = ""
    right_BC_data_file: str = ""
    left_BC_imposed_height: float = 0.0
    left_BC_imposed_discharge: float = 0.0
    right_BC_imposed_height: float = 0.0
    right_BC_imposed_discharge: float = 0.0
    is_topography: bool = False
    topography_type: str = "FlatBottom"
    topography_file: str = ""
    
    def __init__(self, file_name: str = ""):
        """Initialize DataFile with optional parameter file name."""
        # Initialize all fields with their default values
        for field_name, field_obj in DataFile.__dataclass_fields__.items():
            if field_obj.default is not field_obj.default_factory:
                setattr(self, field_name, field_obj.default)
            else:
                setattr(self, field_name, field_obj.default_factory())
        
        if file_name:
            self.file_name = file_name
    
    def read_data_file(self, verbosity: int = 1):
        """Read and parse the JSON data file.
        
        Args:
            verbosity: Level of output verbosity (0=silent, 1=normal, 2=verbose)
        """
        if not os.path.exists(self.file_name):
            print(f"\033[91mERROR::DATAFILE : Unable to open file {self.file_name}\033[0m")
            exit(-1)
        
        if verbosity > 0:
            print(f"Reading JSON data file {self.file_name}")
        
        # Load JSON file
        with open(self.file_name, 'r') as f:
            data = json.load(f)
        
        # Parse parameters
        # results_dir is always "results" (hardcoded, not from JSON)
        self.results_dir = "results"
        self.is_save_final_time_only = data.get("save_final_time_only", self.is_save_final_time_only)
        self.save_frequency = data.get("save_frequency", self.save_frequency)
        
        # Probes
        if "probes" in data:
            probes = data["probes"]
            self.n_probes = len(probes)
            self.probes_references = [p["ref"] for p in probes]
            self.probes_positions = [p["position"] for p in probes]
        
        # Test case
        self.is_test_case = data.get("is_test_case", self.is_test_case)
        self.test_case = data.get("test_case", self.test_case)
        
        # Initial conditions
        self.initial_condition = data.get("initial_condition", self.initial_condition)
        self.initial_height = data.get("initial_height", self.initial_height)
        self.initial_discharge = data.get("initial_discharge", self.initial_discharge)
        self.init_file = data.get("init_file", self.init_file)
        
        # Mesh
        self.xmin = data.get("xmin", self.xmin)
        self.xmax = data.get("xmax", self.xmax)
        self.dx = data.get("dx", self.dx)
        
        # Numerical scheme
        self.numerical_flux = data.get("numerical_flux", self.numerical_flux)
        self.scheme_order = data.get("scheme_order", self.scheme_order)
        self.time_scheme = data.get("time_scheme", self.time_scheme)
        
        # Time parameters
        self.initial_time = data.get("initial_time", self.initial_time)
        self.final_time = data.get("final_time", self.final_time)
        self.time_step = data.get("time_step", self.time_step)
        self.CFL = data.get("CFL", self.CFL)
        
        # Physics
        self.g = data.get("gravity", self.g)
        
        # Boundary conditions
        self.left_BC = data.get("left_BC", self.left_BC)
        self.right_BC = data.get("right_BC", self.right_BC)
        self.left_BC_data_file = data.get("left_BC_data_file", self.left_BC_data_file)
        self.right_BC_data_file = data.get("right_BC_data_file", self.right_BC_data_file)
        self.left_BC_imposed_height = data.get("left_BC_imposed_height", self.left_BC_imposed_height)
        self.left_BC_imposed_discharge = data.get("left_BC_imposed_discharge", self.left_BC_imposed_discharge)
        self.right_BC_imposed_height = data.get("right_BC_imposed_height", self.right_BC_imposed_height)
        self.right_BC_imposed_discharge = data.get("right_BC_imposed_discharge", self.right_BC_imposed_discharge)
        
        # Topography
        self.is_topography = data.get("is_topography", self.is_topography)
        self.topography_type = data.get("topography_type", self.topography_type)
        self.topography_file = data.get("topography_file", self.topography_file)
        
        # Make temporary directory and copy init file
        os.makedirs("./temp", exist_ok=True)
        if self.init_file and os.path.exists(self.init_file):
            shutil.copy(self.init_file, "./temp/initial_condition.txt")
            self.init_file = "temp/initial_condition.txt"
        
        # Create and clean results directory
        if verbosity > 0:
            print("Creating the results directory...")
        
        os.makedirs(f"./{self.results_dir}", exist_ok=True)
        # Clean previous HDF5 results
        for f in os.listdir(f"./{self.results_dir}"):
            if f.endswith(".h5") or f.endswith(".hdf5"):
                os.remove(os.path.join(f"./{self.results_dir}", f))
        
        if verbosity > 0:
            print("\033[92mSUCCESS::DATAFILE : Results directory created successfully !\033[0m")
        
        # Set default topography if not specified
        if not self.is_topography:
            self.topography_type = "FlatBottom"
        
        # Set test case to None if not a test case
        if not self.is_test_case:
            self.test_case = "None"
        
        # Adjust dx to fit within spatial domain
        if verbosity > 0:
            print("\033[95mWARNING::DATAFILE : Adjusting dx to fit within the spatial domain.\033[0m")
        
        import math
        self.Nx = int(math.ceil((self.xmax - self.xmin) / self.dx))
        self.dx = (self.xmax - self.xmin) / self.Nx
        
        if verbosity > 0:
            print(f"\033[95mNew value : dx = {self.dx}\033[0m")
            print("\033[92mSUCCESS::DATAFILE : File read successfully\033[0m")
            print()
    
    def print_data(self):
        """Print all parameters to console."""
        print(f"Mesh                 = Generated")
        print(f"   |xmin             = {self.xmin}")
        print(f"   |xmax             = {self.xmax}")
        print(f"   |Nx               = {self.Nx}")
        print(f"   |dx               = {self.dx}")
        print(f"Is Test Case         = {self.is_test_case}")
        if self.is_test_case:
            print(f"Test Case            = {self.test_case}")
        print(f"InitialCondition     = {self.initial_condition}")
        if self.initial_condition == "UniformHeightAndDischarge":
            print(f"  |Initial Height    = {self.initial_height}")
            print(f"  |Initial Discharge = {self.initial_discharge}")
        if self.initial_condition == "InitFile":
            print(f"  |Init File         = {self.init_file}")
        print(f"Numerical Flux       = {self.numerical_flux}")
        print(f"Order                = {self.scheme_order}")
        print(f"Time Scheme          = {self.time_scheme}")
        print(f"Initial time         = {self.initial_time}")
        print(f"Final time           = {self.final_time}")
        print(f"Time step            = {self.time_step}")
        print(f"Gravity              = {self.g}")
        print(f"SaveFinalTimeOnly    = {self.is_save_final_time_only}")
        if not self.is_save_final_time_only:
            print(f"Save Frequency       = {self.save_frequency}")
        print(f"Number of probes     = {self.n_probes}")
        for i in range(self.n_probes):
            print(f"   |Position probe {self.probes_references[i]} = {self.probes_positions[i]}")
        print(f"LeftBC               = {self.left_BC}")
        if self.left_BC == "DataFile":
            print(f"   |LeftBCFile       = {self.left_BC_data_file}")
        if self.left_BC == "ImposedConstantHeight":
            print(f"   |ImposedHeight    = {self.left_BC_imposed_height}")
            print(f"   |ImposedDischarge = {self.left_BC_imposed_discharge}  (if supercritical)")
        if self.left_BC == "ImposedConstantDischarge":
            print(f"   |ImposedHeight    = {self.left_BC_imposed_height} (if supercritical)")
            print(f"   |ImposedDischarge = {self.left_BC_imposed_discharge}")
        print(f"RightBC              = {self.right_BC}")
        if self.right_BC == "DataFile":
            print(f"   |RightBCFile      = {self.right_BC_data_file}")
        if self.right_BC == "ImposedConstantHeight":
            print(f"   |ImposedHeight    = {self.right_BC_imposed_height}")
            print(f"   |ImposedDischarge = {self.right_BC_imposed_discharge} (if supercritical)")
        if self.right_BC == "ImposedConstantDischarge":
            print(f"   |ImposedHeight    = {self.right_BC_imposed_height} (if supercritical)")
            print(f"   |ImposedDischarge = {self.right_BC_imposed_discharge}")
        print(f"Topography           = {self.topography_type}")
        if self.topography_type == "File":
            print(f"Topography file      = {self.topography_file}")

