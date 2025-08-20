import stim
import sinter
import numpy as np
from typing import *
import multiprocessing as mp
import os
# from beliefmatching import BeliefMatching
import pymatching

# Integrate Stim to detect errors and compute logical error rates

def get_measure_qubits_from_ckt(circuit:stim.Circuit) -> list:
        circuit = circuit.flattened()
        coords = circuit.get_final_qubit_coordinates()
        measure_qubits = []
        for qubit in coords.keys():
            if coords[qubit][0] % 2 == 0 and coords[qubit][1] % 2 == 0:
                measure_qubits.append(qubit)
        return measure_qubits

def gen_syndromes(circuit:stim.Circuit) -> list:
        sim = stim.TableauSimulator()
        sim.do_circuit(circuit)
        num_ancillas = len(get_measure_qubits_from_ckt(circuit))
        record = sim.current_measurement_record()
        return record[:num_ancillas]
    
def check(data:tuple) -> bool:
    lut, entries = data
    hits = np.sum([entry in lut for entry in entries])
    return hits

class stim_integ():
    def __init__(self, use_bm:bool=False) -> None:
        self.use_bm = use_bm # Belief matching
        pass

    # Copied from stim docs
    def count_logical_errors(self, circuit:str, num_shots:int) -> Tuple[int, np.ndarray]:
        circuit = stim.Circuit(circuit)
        # num_detectors = circuit.num_detectors
        num_observables = circuit.num_observables

        # Sample the circuit.
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        num_errors = None
        per_logical_errors = None

        if self.use_bm == False:
            # Extract decoder configuration data from the circuit.
            detector_error_model = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

            # Run the decoder.
            # predictions = sinter.predict_observables(
            #     dem=detector_error_model,
            #     dets=detection_events,
            #     decoder='pymatching',
            # )
            predictions = matcher.decode_batch(detection_events)

            # Count the mistakes.
            num_errors = 0
            per_logical_errors = np.zeros((1, num_observables))
            for actual_flip, predicted_flip in zip(observable_flips, predictions):
                if not np.array_equal(actual_flip, predicted_flip):
                    num_errors += 1
                per_logical_errors += np.array([i != j for i, j in zip(actual_flip, predicted_flip)])
        # else:
        #     bm = BeliefMatching(circuit, max_bp_iters=100)
        #     per_logical_errors = np.zeros((1, num_observables))
        #     predicted_observables = bm.decode_batch(detection_events) # Don't support individual patch LER with BeliefMatching yet
        #     num_errors = np.sum(np.any(predicted_observables != observable_flips, axis=1))
        return num_errors, per_logical_errors
    
    def get_syndromes(self, circuit:str, num_shots:int, cpus:int=int(os.cpu_count() / 2)) -> list:
        if type(circuit) == str:
            circuit = stim.Circuit(circuit)
        
        # this function uses multiprocessing inherently, do NOT use with external multiproc enabled
        
        pool = mp.Pool(cpus)
        syndromes = pool.map(gen_syndromes, [circuit for _ in range(num_shots)])
        # sampler = circuit.compile_sampler()
        # syndromes = sampler.sample(shots=num_shots)
        
        return syndromes
    
    def split_syndromes(self, syndromes:list, syndrome_length:int) -> np.ndarray:
        split = np.array([np.array_split(i, syndrome_length) for i in syndromes])
        return split
    
    def get_lut_hit_rate(self, circuit:stim.Circuit, 
                         num_shots:int, 
                         cpus:int=int(os.cpu_count() / 2)) -> float:
        # Circuit should be only for one round
        if type(circuit) == str:
            circuit = stim.Circuit(circuit)
            
        ratio = 0.9
            
        syndromes = self.get_syndromes(circuit, num_shots, cpus)
        lut_entries = syndromes[:int(num_shots * ratio)]      
        test_entries = syndromes[int(num_shots * ratio):]
        print('\t Completed syndrome generation')
        
        # Convert syndromes to addresses (hashing)
        hash = lambda arr: int(''.join(str(int(i)) for i in arr), 2)
        lut_entries = {hash(arr) for arr in lut_entries}
        test_entries = [hash(arr) for arr in test_entries]
        print('\t Completed hashing')
        
        split_tests = np.array_split(test_entries, cpus)
        
        data = tuple((lut_entries, split_tests[i]) for i in range(cpus))
        pool = mp.Pool(cpus)
        res = pool.map(check, data)
        # hits = np.sum([entry in lut_entries for entry in test_entries])
        hits = np.sum(res)
        
        return hits / len(test_entries)
    pass

if __name__ == '__main__':
    import pickle
    
    p_fast = 0.99
    num_shots = 10_000_000
    cycle_time = 1000
    latency_fast = 500
    latency_slow = 1000

    ch = stim_integ()
    files = {'Passive (d=3)':'d3_passive_ckt.pkl', 'Active (d=3)':'d3_active_ckt.pkl', 'Passive (d=5)':'d5_passive_ckt.pkl', 'Active (d=5)':'d5_active_ckt.pkl'}
    files = {'Passive (d=7)':'d7_passive_ckt.pkl', 'Active (d=7)':'d7_active_ckt.pkl'}#, 'Passive (d=9)':'d9_passive_ckt.pkl', 'Active (d=9)':'d9_active_ckt.pkl'}
    data = {}
    for file in files.keys():
        ckt = None
        print(file)
        with open(files[file], 'rb') as f:
            ckt = pickle.load(f)
            if type(ckt) == str:
                ckt = stim.Circuit(ckt)
        data[file] = np.mean([ch.get_lut_hit_rate(ckt, num_shots) for _ in range(1)])
        
    print(data)