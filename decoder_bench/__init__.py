"""
Decoder Bench - A benchmarking suite for decoders for quantum error correction.

This package provides tools for generating quantum error correction datasets for various 
codes including surface codes, QLDPC codes, color codes, and lattice surgery circuits.

Main Functions:
    - gen_surface_circuit: Generate surface code with circuit-level noise
    - gen_surface_phenom: Generate surface code with phenomenological noise  
    - gen_surface_code_capacity: Generate surface code with code capacity noise
    - gen_qldpc_circuit: Generate QLDPC code with circuit-level noise
    - gen_qldpc_phenom: Generate QLDPC code with phenomenological noise
    - gen_qldpc_code_capacity: Generate QLDPC code with code capacity noise
    - gen_color_circuit: Generate color code with circuit-level noise
    - gen_color_phenom: Generate color code with phenomenological noise
    - gen_color_code_capacity: Generate color code with code capacity noise
    - gen_lattice_surgery_circuit: Generate lattice surgery circuit dataset

Key Classes:
    - DatasetGen: Dataset generation and syndrome collection
    - NoiseModel: Various noise models for quantum circuits
    - LatticeSurgeryCircuit: Lattice surgery circuit generation (available as circuit class)
    - ColorCode: Color code generation
    - Sampler: Samples the generated decoder traces
"""

# Import main generator functions
from .generator import (
    gen_surface_circuit,
    gen_surface_phenom, 
    gen_surface_code_capacity,
    gen_qldpc_circuit,
    gen_qldpc_phenom,
    gen_qldpc_code_capacity,
    gen_color_circuit,
    gen_color_phenom,
    gen_color_code_capacity,
    gen_lattice_surgery_circuit
)

# Import key classes
from .dataset_gen import DatasetGen
from .common.noise import NoiseModel, noisy_circuit_perfect_measurements
from .common.color_code_gen import ColorCode
from .common.build_circuit import build_circuit, get_qldpc_data_qubits
from .common.codes_q import create_bivariate_bicycle_codes
from .sampler import Decoder, DecoderRegistry, DecoderState, Sampler

# Import lattice surgery circuit class
from .ls_ckt_gen.circuit_zzxx import circuit as LatticeSurgeryCircuit

# Expose utility functions
from .common.utils import *
from .common.data_management import *

# Version info
__version__ = "0.1.0"
__author__ = "Decoder Bench Contributors"

# Define what gets imported with "from decoder_bench import *"
__all__ = [
    # Generator functions
    'gen_surface_circuit',
    'gen_surface_phenom', 
    'gen_surface_code_capacity',
    'gen_qldpc_circuit',
    'gen_qldpc_phenom',
    'gen_qldpc_code_capacity',
    'gen_color_circuit',
    'gen_color_phenom',
    'gen_color_code_capacity',
    'gen_lattice_surgery_circuit',
    
    # Key classes
    'DatasetGen',
    'NoiseModel',
    'LatticeSurgeryCircuit',
    'ColorCode',
    
    # Utility functions
    'build_circuit',
    'get_qldpc_data_qubits',
    'create_bivariate_bicycle_codes',
    'noisy_circuit_perfect_measurements',
    'Decoder',
    'DecoderRegistry',
    'DecoderState',
    'Sampler'
]
