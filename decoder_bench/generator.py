try:
    from .dataset_gen import DatasetGen
    from .common.build_circuit import build_circuit, get_qldpc_data_qubits
    from .common.noise import NoiseModel, noisy_circuit_perfect_measurements
    from .common.codes_q import create_bivariate_bicycle_codes
    from .common.color_code_gen import ColorCode
    from .ls_ckt_gen.circuit_zzxx import circuit as circuit_logical_measurement
except ImportError:
    from dataset_gen import DatasetGen
    from common.build_circuit import build_circuit, get_qldpc_data_qubits
    from common.noise import NoiseModel, noisy_circuit_perfect_measurements
    from common.codes_q import create_bivariate_bicycle_codes
    from common.color_code_gen import ColorCode
    from ls_ckt_gen.circuit_zzxx import circuit as circuit_logical_measurement

import stim
import argparse
import os
from typing import Tuple

# max_iterations = 100
# num_records = 50_000_000
# output_dir = '.'

def generate(circuit:stim.Circuit, name:str, max_iterations:int=100, num_records:int=50_000_000, store_unique:bool=True) -> None:
    """
    Generate syndromes for a given circuit and save to a file.
    
    Args:
        circuit (stim.Circuit): The quantum circuit to generate syndromes for.
        name (str): The name of the output file.
        max_iterations (int): Maximum iterations for syndrome generation.
        num_records (int): Number of records to generate.
    """
    generator = DatasetGen(circuit, name=name)
    generator.gen_syndromes(max_iterations=max_iterations, num_records=num_records, store_unique=store_unique)
    return

# Circuit generators
def gen_color_circuit(input:Tuple[int, float]) -> stim.Circuit:
    d, p = input
    circuit = ColorCode(d=d, rounds=d, p_bitflip=p).circuit
    circuit = circuit.without_noise()
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def gen_color_phenom(input:Tuple[int, float]) -> stim.Circuit:
    d, p = input
    circuit = ColorCode(d=d, rounds=d, p_bitflip=p, p_meas=p).circuit
    return circuit

def gen_color_code_capacity(input:Tuple[int, float]) -> stim.Circuit:
    d, p = input
    circuit = ColorCode(d=d, rounds=1, p_bitflip=p).circuit
    return circuit

def gen_qldpc_circuit(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    if d == 6:
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
    elif d == 10:
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
    elif d == 12:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
    else:
        raise ValueError('Invalid distance. Only 6, 10, and 12 are supported.')
    circuit = build_circuit(code, A_list, B_list, p=0,
                            num_repeat=d, use_both=False, 
                            z_basis=(basis == 'z'))
    circuit = circuit.without_noise()
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def gen_qldpc_phenom(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    if d == 6:
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
    elif d == 10:
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
    elif d == 12:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
    else:
        raise ValueError('Invalid distance. Only 6, 10, and 12 are supported.')
    circuit = build_circuit(code, A_list, B_list, p=p,
                            num_repeat=d, use_both=False, 
                            z_basis=(basis == 'z'), phenom=True)
    return circuit

def gen_qldpc_code_capacity(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    if d == 6:
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
    elif d == 10:
        code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
    elif d == 12:
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
    else:
        raise ValueError('Invalid distance. Only 6, 10, and 12 are supported.')
    circuit = build_circuit(code, A_list, B_list, p=0,
                            num_repeat=1, use_both=False, 
                            z_basis=(basis == 'z'))
    dq = get_qldpc_data_qubits(code)
    circuit = circuit.without_noise()
    circuit = noisy_circuit_perfect_measurements(circuit, p, dq)
    return circuit

def gen_surface_circuit(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    circuit = stim.Circuit.generated(f'surface_code:rotated_memory_{basis}',
                                     rounds=d,
                                     distance=d)
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def gen_surface_phenom(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    circuit = stim.Circuit.generated(f'surface_code:rotated_memory_{basis}',
                                     before_measure_flip_probability=p,
                                     before_round_data_depolarization=p,
                                     rounds=d,
                                     distance=d)
    return circuit

def gen_surface_code_capacity(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    circuit = stim.Circuit.generated(f'surface_code:rotated_memory_{basis}',
                                     rounds=1,
                                     distance=d)
    circuit = noisy_circuit_perfect_measurements(circuit, p)
    return circuit

def gen_lattice_surgery_circuit(input:Tuple[int, float, str]) -> stim.Circuit:
    d, p, basis = input
    ls_basis = basis.upper()
    basis = basis.upper()
    sim = circuit_logical_measurement(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=25, fixed_t2=40, fixed_cnot_latency=50, fixed_measure_latency=600, fixed_cnot_noise=p, fixed_measure_noise=p, rounds_per_op=d, idle_multiplier=1, basis=basis, ls_basis=ls_basis, merge=True).from_string('qreg q[2];')
    ckt = sim.ckt
    if type(ckt) is not stim.Circuit:
        ckt = stim.Circuit(ckt)
    circuit = ckt.without_noise()
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    return circuit

def main():
    """Main entry point for the decoder-bench-generate command."""
    parser = argparse.ArgumentParser(description='Generate quantum error correction datasets')
    
    # Required arguments
    parser.add_argument('--code', type=str, required=True, choices=['surface', 'qldpc', 'color', 'ls'],
                        help='Type of quantum code; use "ls" for a Lattice Surgery dataset')
    parser.add_argument('--basis', type=str, required=True, choices=['x', 'z'],
                        help='Measurement basis (ignored for the color code); Used as the Lattice Surgery basis if code == "ls"')
    parser.add_argument('--noise', type=str, required=True, 
                        choices=['code_capacity', 'phenom', 'circuit'],
                        help='Noise model to use; Lattice Surgery only uses circuit-level noise')
    parser.add_argument('--distance', '-d', type=int, required=True,
                        help='Code distance')
    parser.add_argument('--depolarization', '-p', type=float, required=True,
                        help='Depolarization/error probability')
    
    # Optional arguments
    parser.add_argument('--iterations', type=int, default=100,
                        help='Maximum iterations (default: 100)')
    parser.add_argument('--records', type=int, default=50_000_000,
                        help='Number of records to generate (default: 50,000,000)')
    parser.add_argument('--output-dir', type=str, default='../datasets',
                        help='Directory to save output files')
    parser.add_argument('--store_unique', type=bool, default=True, help='Store only unique syndromes (default: True)')
    
    args = parser.parse_args()
    
    code = args.code.lower()
    basis = args.basis.lower()
    noise_model = args.noise.lower()
    distance = args.distance
    depolarization = args.depolarization
    global max_iterations
    max_iterations = args.iterations
    global num_records
    num_records = args.records
    global output_dir
    output_dir = args.output_dir
    global store_unique
    store_unique = args.store_unique

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if code == 'surface':
        if noise_model == 'circuit':
            circuit = gen_surface_circuit((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/surface_circuit_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
        elif noise_model == 'phenom':
            circuit = gen_surface_phenom((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/surface_phenom_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
        elif noise_model == 'code_capacity':
            circuit = gen_surface_code_capacity((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/surface_code_capacity_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
    elif code == 'qldpc':
        if noise_model == 'circuit':
            circuit = gen_qldpc_circuit((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/qldpc_circuit_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
        elif noise_model == 'phenom':
            circuit = gen_qldpc_phenom((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/qldpc_phenom_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
        elif noise_model == 'code_capacity':
            circuit = gen_qldpc_code_capacity((distance, depolarization, basis))
            generate(circuit, f'{output_dir}/qldpc_code_capacity_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
    elif code == 'color':
        if noise_model == 'circuit':
            circuit = gen_color_circuit((distance, depolarization))
            generate(circuit, f'{output_dir}/color_circuit_{distance}_{depolarization}', max_iterations, num_records, store_unique)
        elif noise_model == 'phenom':
            circuit = gen_color_phenom((distance, depolarization))
            generate(circuit, f'{output_dir}/color_phenom_{distance}_{depolarization}', max_iterations, num_records, store_unique)
        elif noise_model == 'code_capacity':
            circuit = gen_color_code_capacity((distance, depolarization))
            generate(circuit, f'{output_dir}/color_code_capacity_{distance}_{depolarization}', max_iterations, num_records, store_unique)
    elif code == 'ls':
        circuit = gen_lattice_surgery_circuit((distance, depolarization, basis))
        generate(circuit, f'{output_dir}/ls_circuit_{distance}_{depolarization}_{basis}', max_iterations, num_records, store_unique)
    else:
        raise ValueError('How did you get here?')

if __name__ == "__main__":
    main()