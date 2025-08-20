import pandas as pd
import numpy as np
import pickle
import os

# Define gate latencies and noise for every physical qubit
# Can be extended to use different noise models/profiles or data from real qubits

# To be used as a base class for the circuit class, will support changing noise
# for specified qubits (defined by coords). Start by specifying the noise and 
# latency of gates for all physical qubits in the given lattice.

# Also incorporates idling errors.

class gate_lib():
    def __init__(self, distance, num_patches_x,
                 num_patches_y, spacing, seed=0,
                 fixed_measure_noise=None, fixed_measure_latency=None,
                 fixed_cnot_noise=None, fixed_cnot_latency=None,
                 cnot_latency_dist:str=None,
                 cnot_error_dist:str=None,
                 measure_latency_dist:str=None,
                 measure_error_dist:str=None,
                 cnot_latency_mean:float=None,
                 cnot_latency_std:float=None,
                 cnot_error_mean:float=None,
                 cnot_error_std:float=None,
                 measure_latency_mean:float=None,
                 measure_latency_std:float=None,
                 measure_error_mean:float=None,
                 measure_error_std:float=None,
                 idle_multiplier:int=1,
                 basis:str='Z',
                 error1Q=0.0001, latency1Q=30,
                 gates_1Q=['H', 'X', 'S', 'Z', 'Y'],
                 gates_2Q=['CX', 'CZ'],
                 measures=['M', 'MR', 'MX']) -> None:
        self.all_coordinates = [tuple((j, i)) 
                                for i in range((2 * distance + 1 + spacing) * num_patches_x) 
                                for j in range((2 * distance + 1 + spacing) * num_patches_y) 
                                if i % 2 == j % 2]
        """
        All possible physical qubit coordinates in the lattice
        """
        np.random.seed(seed)
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.gates_1Q = gates_1Q
        """
        All supported 1Q gates
        """
        self.gates_2Q = gates_2Q
        """
        All supported 2Q gates
        """
        self.measures = measures
        """
        All supported measurement gates
        """
        self.measure_latency = fixed_measure_latency
        self.measure_noise = fixed_measure_noise
        self.cnot_latency = fixed_cnot_latency
        self.cnot_noise = fixed_cnot_noise
        self.cnot_latency_dist = cnot_latency_dist
        self.cnot_error_dist = cnot_error_dist
        self.measure_latency_dist = measure_latency_dist
        self.measure_error_dist = measure_error_dist
        self.cnot_latency_mean   = cnot_latency_mean
        self.cnot_latency_std    = cnot_latency_std
        self.cnot_error_mean     = cnot_error_mean
        self.cnot_error_std      = cnot_error_std
        self.measure_latency_mean = measure_latency_mean
        self.measure_latency_std = measure_latency_std
        self.measure_error_mean = measure_error_mean
        self.measure_error_std  = measure_error_std
        self.idle_multiplier = idle_multiplier
        self.error1Q = error1Q
        self.basis = basis
        """
        1Q gate error rate
        """
        self.latency1Q = latency1Q
        """
        1Q gate latency
        """
        self.profile = {i:self.get_data() for i in self.all_coordinates}
        """
        gate_lib.profile[qubit_coord]['latency' or 'error'][gate ('H', 'CX' etc.)]
        """
        df = pd.read_csv(self.path + '/data/T1_T2.csv')
        self.T1 = np.array(df['T1'], dtype=float) * 10 ** (-6) # In microseconds
        """
        List of all T1 times
        """
        self.T2 = np.array(df['T2'], dtype=float) * 10 ** (-6) # In microseconds
        """
        List of all T2 times
        """
        self.t1_mapper = {}
        """
        Maps T1 times with all physical qubits
        """
        self.t2_mapper = {}
        """
        Maps T2 times with all physical qubits
        """
        pass

    def get_data(self) -> dict:
        measure_duration = self.sample_measure_latency() if self.measure_latency == None else self.measure_latency
        cnot_duration = self.sample_gate_latency() if self.cnot_latency == None else self.cnot_latency
        if self.cnot_latency_dist != None:
            if self.cnot_latency_dist != 'gaussian':
                raise ValueError('Only Gaussian distributions supported by error model')
            else:
                cnot_duration = np.random.normal(self.cnot_latency_mean, self.cnot_latency_std)
                pass
        if self.measure_latency_dist != None:
            if self.measure_latency_dist != 'gaussian':
                raise ValueError('Only Gaussian distributions supported by error model')
            else:
                measure_duration = np.random.normal(self.measure_latency_mean, self.measure_latency_std)
                pass
        measure_error = self.sample_measure_error(measure_duration) if self.measure_noise == None else self.measure_noise
        cnot_error = self.sample_gate_error(cnot_duration) if self.cnot_noise == None else self.cnot_noise
        if self.cnot_error_dist != None:
            if self.cnot_error_dist != 'gaussian':
                raise ValueError('Only Gaussian distributions supported by error model')
            else:
                cnot_error = np.random.normal(self.cnot_error_mean, self.cnot_error_std)
                pass
        if self.measure_error_dist != None:
            if self.measure_error_dist != 'gaussian':
                raise ValueError('Only Gaussian distributions supported by error model')
            else:
                measure_error = np.random.normal(self.measure_error_mean, self.measure_error_std)
                pass
        # '|' operator for dicts not supported for python versions < 3.9
        temp = {'latency': {**{i:self.latency1Q for i in self.gates_1Q}, 
                **{i:cnot_duration for i in self.gates_2Q},
                **{i:measure_duration for i in self.measures}},
                'error':{**{i:self.error1Q for i in self.gates_1Q},
                **{i:cnot_error for i in self.gates_2Q},
                **{i:measure_error for i in self.measures}}}
        return temp
    
    def set_cnot_latency(self, qubit:tuple, latency:float):
        for gate in self.gates_2Q:
            self.profile[qubit]['latency'][gate] = latency
        return
    
    def set_measure_latency(self, qubit:tuple, latency:float):
        for m in self.measures:
            self.profile[qubit]['latency'][m] = latency
        return
    
    def set_cnot_error(self, qubit:tuple, error:float):
        for gate in self.gates_2Q:
            self.profile[qubit]['error'][gate] = error
        return
    
    def set_measure_error(self, qubit:tuple, error:float):
        for m in self.measures:
            self.profile[qubit]['error'][m] = error
        return
    
    def sample_gate_latency(self) -> float:
        # Sample latencies from existing dataset
        cnot_durations = pd.read_csv('%s/data/cnot_data.csv'%(self.path))['CNOT Duration']
        return np.random.choice(cnot_durations)
    
    def sample_measure_latency(self) -> float:
        # Sample latencies from existing dataset
        measure_durations = pd.read_csv('%s/data/readout_data.csv'%(self.path))['Readout Duration']
        return np.random.choice(measure_durations)
    
    def sample_gate_error(self, duration:float) -> float:
        with open('%s/data/cnot_model.bin'%(self.path), 'rb') as file:
            model = pickle.load(file)
        return model.predict(np.array(duration).reshape(-1, 1))[0]
    
    def sample_measure_error(self, duration:float) -> float:
        with open('%s/data/readout_model.bin'%(self.path), 'rb') as file:
            model = pickle.load(file)
        return model.predict(np.array(duration).reshape(-1, 1))[0]
    
    def assign_T1(self, coord:tuple, val:float=None) -> None:
        if val == None:
            self.t1_mapper[coord] = np.random.choice(self.T1)
        else:
            self.t1_mapper[coord] = val * 10 ** (-6)
        return 
    
    def assign_T2(self, coord:tuple, val:float=None) -> None:
        if val == None:
            self.t2_mapper[coord] = np.random.choice(self.T2)
        else:
            self.t2_mapper[coord] = val * 10 ** (-6)
        return
    
    def idling_model(self, time:float, coord:tuple, idx:int) -> str:
        # Time: Time idle in ns
        if time < 10 ** (-6):
            time = 0
        px = (1 - np.exp(-time * 10 ** (-9) / self.t1_mapper[coord])) / 4
        py = px
        pz = (1 - np.exp(-time * 10 ** (-9) / self.t2_mapper[coord])) / 2 - px
        assert px >= 0, "Negative pauli error (Px), consider changing the T1/T2 times."
        assert py >= 0, "Negative pauli error (Py), consider changing the T1/T2 times."
        assert pz >= 0, "Negative pauli error (Pz), consider changing the T1/T2 times."
        # error = 'PAULI_CHANNEL_1(%f,%f,%f) %i'%((px) * self.idle_multiplier, 
        #                                         (py) * self.idle_multiplier, 
        #                                         (pz) * self.idle_multiplier, 
        #                                         idx)
        
        ######
        # self.idle_multiplier is used for lifting the idling errors for IBM systems. The reason
        # for this is that IBM systems that IBM systems have longer T1/T2 times at the cost of longer
        # measurement latencies. The effect of these long measurement latencies cannot be captured by simply
        # increasing the readout duration specified during circuit generation. To work around this, we lift the 
        # idling errors only for IBM systems by a factor of 3. For google systems, this factor remains 1.
        ######
        if self.basis == 'Z':
            # Only X/Y errors
            error = 'X_ERROR(%f) %i'%(1 - (1 - px * self.idle_multiplier) * (1 - py * self.idle_multiplier), idx)
        else:
            error = 'Z_ERROR(%f) %i'%(1 - ((1 - pz * self.idle_multiplier) * (1 - py * self.idle_multiplier)), idx)
        error
        return error
    pass

if __name__ == "__main__":
    obj = gate_lib(distance=3, num_patches_x=2, num_patches_y=1, spacing=1)
    print(obj.profile)
