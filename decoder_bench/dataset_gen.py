import numpy as np
import h5py
from itertools import product
import stim
import os

try:
    from .common.build_circuit import dem_to_check_matrices
    from .common.dem_matrices import detector_error_model_to_check_matrices
    from .common.data_management import hash_bool_array
    from .common.noise import NoiseModel
except ImportError:
    from common.build_circuit import dem_to_check_matrices
    from common.dem_matrices import detector_error_model_to_check_matrices
    from common.data_management import hash_bool_array
    from common.noise import NoiseModel

class CompressedBinaryArrayStore:
    """Helper class to track unique binary arrays.
    """
    def __init__(self):
        self.array_hashes = set()
        self.compressed_arrays = []
        self.original_lengths = []
        
    def add_array(self, bool_array):
        """
        Add a binary array if it hasn't been seen before.
        Returns True if array was added, False if it was already present.
        """
        array_hash = hash_bool_array(bool_array)
        
        if array_hash in self.array_hashes:
            return False
            
        self.array_hashes.add(array_hash)
        return True
    
    def __len__(self):
        return len(self.compressed_arrays)

class DatasetGen(CompressedBinaryArrayStore):
    def __init__(self, circuit:stim.Circuit,
                 matchable:bool=False,
                 name:str='dataset') -> None:
        self.circuit = circuit
        assert type(circuit) is stim.Circuit, 'Invalid circuit object, stim.Circuit expected'
        chk, obs, priors, _ = dem_to_check_matrices(circuit.detector_error_model(), 
                                                    return_col_dict=True)
        # if matchable:
        #     obj = detector_error_model_to_check_matrices(circuit.detector_error_model(decompose_errors=True))
        #     self.H = obj.edge_check_matrix.toarray()
        #     self.obs = obj.edge_observables_matrix.toarray()
        #     self.priors = obj.priors
        #     self.priors = obj.hyperedge_to_edge_matrix @ self.priors
        # else:
        self.H = chk.toarray()
        self.priors = priors
        self.obs = obs.toarray()
        self.matchable = matchable
        self.name = name
        super().__init__()
        pass
    
    def get_binary_permutations(self, n: int) -> np.ndarray:
        """Generate binary permutations for a bitstring of length n"""
        return np.array(list(product([0, 1], repeat=n)), dtype=np.bool_)
    
    def update_circuit(self, circuit:stim.Circuit) -> None:
        """Update the circuit and recompute the check matrices"""
        self.circuit = circuit
        chk, obs, priors, _ = dem_to_check_matrices(circuit.detector_error_model(), 
                                                    return_col_dict=True)
        if self.matchable:
            obj = detector_error_model_to_check_matrices(circuit.detector_error_model(decompose_errors=True))
            self.H = obj.edge_check_matrix.toarray()
            self.obs = obj.edge_observables_matrix.toarray()
            self.priors = obj.priors
            self.priors = obj.hyperedge_to_edge_matrix @ self.priors
        else:
            self.H = chk.toarray()
            self.priors = priors
            self.obs = obs.toarray()
        return

    def gen_syndromes(self, num_records:int=50_000_000, max_iterations:int=1000, store_unique:bool=True) -> None:
        """Generate syndromes and save them to a h5 file.
        Args:
            num_records (int): Number of records to generate.
            max_iterations (int): Maximum number of iterations to run. Every iteration samples 10M detectors.
            store_unique (bool): If True, only store unique syndromes (default: True).
        """
    
        print("Dataset Generation Configuration:")
        print(f"  Dataset name: {self.name}")
        print(f"  Check matrix dimensions: {self.H.shape}")
        print(f"  Target syndrome records: {num_records}")
        print(f"  Maximum iterations: {max_iterations}")
        print(f"  Matchable circuit: {self.matchable}")
        
        m, n = self.H.shape
        iterations = 0
        total_saved = 0
        all_syndromes = []
        while len(all_syndromes) + total_saved < num_records and iterations < max_iterations:
            print('')
            sampler = self.circuit.compile_detector_sampler(seed=iterations + 42)
            dets, obs = sampler.sample(shots=1_000_000, bit_packed=False, separate_observables=True)
            observables = []
            syndromes = []
            for i, det in enumerate(dets):
                print(f"Iterations: {iterations}, Collected syndromes: {len(self) + total_saved}, Target: {num_records}, Max. iterations: {max_iterations}, Collected this itr:{len(syndromes)},  idx:{i}", end='\r')
                success = self.add_array(det) # will store the array, or not
                if not store_unique:
                    success = True
                if success:
                    observables.append(obs[i])
                    syndromes.append(det)
                    if total_saved + len(syndromes) >= num_records:
                        if os.path.exists(f'{self.name}.h5'):
                            with h5py.File(f'{self.name}.h5', 'a') as file:
                                all_syndromes.extend(syndromes)
                                file['syndromes'].resize((len(file['syndromes']) + len(syndromes), m))
                                file['syndromes'][-len(syndromes):] = syndromes
                                file['observables'].resize((len(file['observables']) + len(observables), len(observables[0])))
                                file['observables'][-len(observables):] = observables
                                total_saved += len(syndromes)
                                self.compressed_arrays = []
                        else:
                            with h5py.File(f'{self.name}.h5', 'w') as file:
                                all_syndromes.extend(syndromes)
                                file.create_dataset('syndromes', 
                                                    data=syndromes, 
                                                    dtype=np.bool_, 
                                                    compression='gzip',
                                                    shape=(len(syndromes), m),
                                                    maxshape=(None, m))
                                file.create_dataset('observables', 
                                                    data=observables, 
                                                    dtype=np.bool_, 
                                                    compression='gzip',
                                                    shape=(len(observables), len(observables[0])),
                                                    maxshape=(None, len(observables[0])))
                                file.create_dataset('check_matrix', data=self.H.astype(np.uint8), dtype=np.uint8, compression='gzip')
                                file.create_dataset('obs_matrix', data=self.obs.astype(np.uint8), dtype=np.uint8, compression='gzip')
                                file.create_dataset('priors', data=self.priors, dtype=float, compression='gzip')
                                file.create_dataset('circuit', data=str(self.circuit), dtype=h5py.string_dtype(encoding='utf-8'))
                                total_saved += len(syndromes)
                                self.compressed_arrays = []
                            pass
                        return
                    pass
                pass
            if iterations == 0:
                with h5py.File(f'{self.name}.h5', 'w') as file:
                    all_syndromes.extend(syndromes)
                    file.create_dataset('syndromes', 
                                        data=syndromes, 
                                        dtype=np.bool_, 
                                        compression='gzip',
                                        shape=(len(syndromes), m),
                                        maxshape=(None, m))
                    file.create_dataset('observables', 
                                        data=observables, 
                                        dtype=np.bool_, 
                                        compression='gzip',
                                        shape=(len(observables), len(observables[0])),
                                        maxshape=(None, len(observables[0])))
                    file.create_dataset('check_matrix', data=self.H.astype(np.uint8), dtype=np.uint8, compression='gzip')
                    file.create_dataset('obs_matrix', data=self.obs.astype(np.uint8), dtype=np.uint8, compression='gzip')
                    file.create_dataset('priors', data=self.priors, dtype=float, compression='gzip')
                    file.create_dataset('circuit', data=str(self.circuit), dtype=h5py.string_dtype(encoding='utf-8'))
                    total_saved += len(syndromes)
                    self.compressed_arrays = []
            else:
                with h5py.File(f'{self.name}.h5', 'a') as file:
                    all_syndromes.extend(syndromes)
                    file['syndromes'].resize((len(file['syndromes']) + len(syndromes), m))
                    file['syndromes'][-len(syndromes):] = syndromes
                    file['observables'].resize((len(file['observables']) + len(observables), len(observables[0])))
                    file['observables'][-len(observables):] = observables
                    total_saved += len(syndromes)
                    self.compressed_arrays = []
            iterations += 1
            pass
        return
    
if __name__ == '__main__':
    d = 7
    p = 0.001
    circuit = stim.Circuit.generated('surface_code:rotated_memory_z',
                                     distance=d,
                                     rounds=d)
    circuit = NoiseModel.SI1000(p).noisy_circuit(circuit)
    dataset = DatasetGen(circuit, name=f'surface_d{d}_{p}_z')
    dataset.gen_syndromes(num_records=1_000_000)