#########################################################
#########################################################
# Two patches -> merge -> split -> measure observable
#########################################################
#########################################################


import numpy as np
import networkx as nx
import itertools
from typing import *
try:
    # Try relative imports first (when used as a package)
    from .circuit_parser import circuit_parser
    from .gate_lib import gate_lib
    from .stim_integ import stim_integ
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from circuit_parser import circuit_parser
    from gate_lib import gate_lib
    from stim_integ import stim_integ

# Work at an abstraction level of patches -- the input to this class is a circuit
# defining the logical qubits and the logical operations between them.
# Will need a way to parse the input circuit (from a file or input string).
# Once all logical qubits have been defined, logical operations using lattce surgery
# can be performed. 

# Some constraints to make implementation simpler:
# - Z boundaries fixed to be on the left and right side of the patches
# - X boundaries fixed to be on the top and bottom of the patches
# So a merge/split of two patches in the vertical direction will operate on the X boundary
# and ones in the horizontal direction will operate on the Z boundary.
class circuit(circuit_parser, gate_lib, stim_integ):
    def __init__(self, distance:int=3, disable_noise:bool=False, seed:int=0,
                 num_patches_x:int=20, num_patches_y:int=20, 
                 spacing:int=1, rounds_per_op:int=None,
                 init_error=None,
                 fixed_measure_noise:float=None, 
                 fixed_measure_latency:float=None,
                 fixed_cnot_noise:float=None,
                 fixed_cnot_latency:float=None,
                 fixed_t1:float=None, fixed_t2:float=None,
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
                 use_bm:bool=False, # Use Belief Matching decoder
                 basis:str='Z',
                 ls_basis:str='X',
                 merge:bool=True,
                 defective_qubits:int=0,
                 defective_ratio:float=0,
                 error1Q:float=0.0001, latency1Q:float=30,
                 gates_1Q:list=['H', 'X', 'S', 'Z', 'Y'],
                 gates_2Q:list=['CX', 'CZ'],
                 measures:list=['M', 'MR', 'MX']) -> None:
        np.random.seed(seed)
        self.d = distance
        self.disable_noise = disable_noise
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        assert spacing > 0, 'expected spacing between patches > 0, got %i'%(spacing)
        self.spacing = spacing
        self.init_error = init_error
        self.coords_index_mapper = {}
        self.index_coords_mapper = {}
        self.measurement_tracker = []
        self.distributed_idle = False
        self.passive_sync = False
        self.distributed_rounds = 1
        self.total_idle = 0
        self.fixed_t1 = fixed_t1
        self.fixed_t2 = fixed_t2
        self.defective_qubits = defective_qubits
        self.defective_ratio = defective_ratio
        self.gates = gates_1Q + gates_2Q + measures
        self.time = 0 # Tracks the time of the fastest qubit after every TICK
        self.decode_latency = 100 # In ns
        self.basis = basis
        self.ls_basis = ls_basis
        self.measures_logical = ['M', 'MX']
        self.merge_ls = merge
        self.use_bm = use_bm
        self.zobs = []
        self.xobs = []
        self.prev_datas = []
        self.prev_xmeasures = []
        self.prev_zmeasures = []
        self.new_datas = []
        if rounds_per_op == None:
            rounds_per_op = distance + 1
        self.rounds_per_op = rounds_per_op
        self.sync_round = self.rounds_per_op
        stim_integ.__init__(self, use_bm=use_bm)
        circuit_parser.__init__(self)
        gate_lib.__init__(self, distance, num_patches_x, num_patches_y,
                          spacing, seed=seed, fixed_cnot_latency=fixed_cnot_latency,
                          fixed_measure_noise=fixed_measure_noise,
                          fixed_cnot_noise=fixed_cnot_noise,
                          fixed_measure_latency=fixed_measure_latency,
                          cnot_latency_dist=cnot_latency_dist,
                          cnot_error_dist=cnot_error_dist,
                          measure_latency_dist=measure_latency_dist,
                          cnot_latency_mean=cnot_latency_mean,
                          cnot_latency_std=cnot_latency_std,
                          cnot_error_mean=cnot_error_mean,
                          cnot_error_std=cnot_error_std,
                          measure_error_mean=measure_error_mean,
                          measure_error_std=measure_error_std,
                          measure_latency_mean=measure_latency_mean,
                          measure_latency_std=measure_latency_std,
                          measure_error_dist=measure_error_dist,
                          idle_multiplier=idle_multiplier, basis=basis,
                          error1Q=error1Q, latency1Q=latency1Q,
                          gates_1Q=gates_1Q, gates_2Q=gates_2Q, measures=measures)
        self.assign_decoder_latency()
        pass
    
    def assign_decoder_latency(self):
        latencies = {3:70, 5:60, 7:60, 9:70, 11:110, 13:160, 15:250, 17:370} # From Riverlane
        self.decoder_latency = latencies[self.d]
        return
    
    def get_patch_layout(self) -> None:
        # Print the layout of all patches for the given code distance and grid
        # size.
        # Every logical qubit spans 2*d points on the grid
        # The spacing defines the number of data qubits between every patch
        for i in range(self.num_patches_x):
            for j in range(self.num_patches_y):
                space = '.' * self.spacing
                if j == self.num_patches_y - 1:
                    space = ''
                print('|%i|'%((i) * (self.num_patches_y) + (j + 1)) + space, end='')
                pass
            print('')
        return
    
    def noise_profile(self) -> None:
        # Define the noise affecting every qubit in the lattice
        # This will be added during the circuit synthesis phase.
        return
    
    def merge(self, coords:dict) -> dict:
        num_qubits = len(coords.keys())
        qubits = list(coords.keys())
        # Place data qubits
        data_coords = []
        # Place measurement qubits
        x_measure_coords = []
        z_measure_coords = []
        # Need to define which data qubits form a patch
        # Find row and column of the patch specified by qubit
        row = np.floor((qubits[0]) / self.num_patches_y)
        col = (qubits[0]) % self.num_patches_y
        if self.ls_basis == 'X':
            patch_start_x = row * (2 * self.d + self.spacing * 2)
            patch_start_y = col * (2 * self.d + self.spacing * 2)
            limit_x = 2 * self.d + 1
            limit_y = num_qubits * (2 * self.d + 1) + self.spacing
            loop_x = self.d + 1
            loop_y = num_qubits * (self.d + 1)
            lr_boundary = self.d
            tb_boundary = num_qubits * self.d + self.spacing
        else:
            patch_start_x = col * (2 * self.d + self.spacing * 2)
            patch_start_y = row * (2 * self.d + self.spacing * 2)
            limit_x = num_qubits * (2 * self.d + 1) + self.spacing
            limit_y = 2 * self.d + 1
            loop_x = num_qubits * (self.d + 1)
            loop_y = self.d + 1
            lr_boundary = num_qubits * self.d + self.spacing
            tb_boundary = self.d
        # Shift origin temporarily to treat this patch as a single patch
        mapper = {}
        for i in range(limit_x):
            for j in range(limit_y):
                if i % 2 == j % 2:
                    indices = ((int(patch_start_x + i), int(patch_start_y + j)))
                    mapper[tuple((i, j))] = indices
                pass
        # Define measure qubits for both basis
        for i in range(loop_x):
            for j in range(loop_y):
                coords = tuple((2 * i, 2 * j))
                left_right_boundary = (i == 0 or i == lr_boundary)
                top_bottom_boundary = (j == 0 or j == tb_boundary)
                parity = i % 2 != j % 2
                if left_right_boundary and parity:
                    continue
                if top_bottom_boundary and not parity:
                    continue
                if parity:
                    x_measure_coords.append(coords)
                else:
                    z_measure_coords.append(coords)
                pass
        # Define observables and data qubits
        x_observable = []
        z_observable = []
        for i in range(limit_x): # vertical
            for j in range(limit_y): # horizontal
                if i % 2 == 1 and j % 2 == 1:
                    data_coords.append(tuple((i, j)))
                    if i == 1:
                        x_observable.append(tuple((i, j)))
                    if j == 1:
                        z_observable.append(tuple((i, j)))
        # Shift origin back
        data_coords = [mapper[i] for i in data_coords]
        x_observable = [mapper[i] for i in x_observable]
        z_observable = [mapper[i] for i in z_observable]
        x_measure_coords = [mapper[i] for i in x_measure_coords]
        z_measure_coords = [mapper[i] for i in z_measure_coords]

        all_coords = data_coords + x_measure_coords + z_measure_coords
        self.new_datas = [i for i in data_coords if i not in self.prev_datas]
        self.measure_new_data_mapper = {}
        self.new_xmeasure = [i for i in x_measure_coords if i not in self.prev_xmeasures]
        self.new_zmeasure = [i for i in z_measure_coords if i not in self.prev_zmeasures]

        # Assign T1 and T2 times to each coord (physical qubit)
        _ = [self.assign_T1(coord, val=self.fixed_t1) for coord in all_coords]
        _ = [self.assign_T2(coord, val=self.fixed_t2) for coord in all_coords]
        if self.defective_qubits != 0:
            _ = [self.assign_T1(coord, val=self.fixed_t1 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
            _ = [self.assign_T2(coord, val=self.fixed_t2 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
        
        # Pass coords of this logical qubit to the final circuit synthesizer
        mappings = {}
        mappings['data_coords'] = data_coords
        mappings['x_observable'] = x_observable
        mappings['z_observable'] = z_observable
        mappings['x_measure_coords'] = x_measure_coords
        mappings['z_measure_coords'] = z_measure_coords
        return mappings

    def reset_merged_data_qubits(self, mappings, old_coords) -> str:
        data_coords = mappings['data_coords']
        data_idx = [self.coords_index_mapper[i] for i in data_coords]
        if self.ls_basis == 'Z':
            reset_data = 'R'
        else:
            reset_data = 'RX'
        old_data_coords = []
        for mapping in old_coords.values():
            old_data_coords += mapping['data_coords']
        old_data_idx = [self.coords_index_mapper[i] for i in old_data_coords]
        ckt_str = reset_data + ' ' + ' '.join(str(i) for i in data_idx if i not in old_data_idx) + '\n'
        return ckt_str
    
    def reset_merged_measure_qubits(self, mappings, old_coords) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
        z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
        measure_idxs = z_measure_idx + x_measure_idx
        reset_measure = 'R'
        old_measure_coords = []
        for mapping in old_coords.values():
            old_measure_coords += mapping['x_measure_coords']
            old_measure_coords += mapping['z_measure_coords']
        old_measure_idxs = [self.coords_index_mapper[i] for i in old_measure_coords]
        ckt_str += reset_measure + ' ' + ' '.join(str(i) for i in measure_idxs if i not in old_measure_idxs) + '\n'
        return ckt_str

    def map_qubit(self, qubit:int=0) -> dict:
        # Place data qubits
        data_coords = []
        # Place measurement qubits
        x_measure_coords = []
        z_measure_coords = []
        # Need to define which data qubits form a patch
        # Find row and column of the patch specified by qubit
        row = np.floor((qubit) / self.num_patches_y)
        col = (qubit) % self.num_patches_y
        if self.ls_basis == 'X':
            patch_start_x = row * (2 * self.d + self.spacing * 2)
            patch_start_y = col * (2 * self.d + self.spacing * 2)
        else:
            patch_start_x = col * (2 * self.d + self.spacing * 2)
            patch_start_y = row * (2 * self.d + self.spacing * 2)
        # Shift origin temporarily to treat this patch as a single patch
        mapper = {}
        for i in range(self.d * 2 + 1):
            for j in range(self.d * 2 + 1):
                if i % 2 == j % 2:
                    indices = ((int(patch_start_x + i), int(patch_start_y + j)))
                    mapper[tuple((i, j))] = indices
                pass
        # Define measure qubits for both basis
        for i in range(self.d + 1):
            for j in range(self.d + 1):
                coords = tuple((2 * i, 2 * j))
                left_right_boundary = (i == 0 or i == self.d)
                top_bottom_boundary = (j == 0 or j == self.d)
                parity = i % 2 != j % 2
                if left_right_boundary and parity:
                    continue
                if top_bottom_boundary and not parity:
                    continue
                if parity:
                    x_measure_coords.append(coords)
                else:
                    z_measure_coords.append(coords)
                pass
        # Define observables and data qubits
        x_observable = []
        z_observable = []
        for i in range(self.d * 2 + 1): # vertical
            for j in range(self.d * 2 + 1): # horizontal
                if i % 2 == 1 and j % 2 == 1:
                    data_coords.append(tuple((i, j)))
                    if i == self.d:
                        x_observable.append(tuple((i, j)))
                    if j == self.d:
                        z_observable.append(tuple((i, j)))
        # Shift origin back
        data_coords = [mapper[i] for i in data_coords]
        x_observable = [mapper[i] for i in x_observable]
        z_observable = [mapper[i] for i in z_observable]
        x_measure_coords = [mapper[i] for i in x_measure_coords]
        z_measure_coords = [mapper[i] for i in z_measure_coords]

        self.xobs.extend(x_observable)
        self.zobs.extend(z_observable)

        all_coords = data_coords + x_measure_coords + z_measure_coords
        self.prev_datas += data_coords
        self.prev_xmeasures += x_measure_coords
        self.prev_zmeasures += z_measure_coords
        self.num_measures = len(x_measure_coords + z_measure_coords)
        self.num_datas = len(data_coords)

        # Assign T1 and T2 times to each coord (physical qubit)
        _ = [self.assign_T1(coord, val=self.fixed_t1) for coord in all_coords]
        _ = [self.assign_T2(coord, val=self.fixed_t2) for coord in all_coords]
        # Add defects to the lattice
        if self.defective_qubits == 1: # X observables
            self.all_defective_qubits = x_observable
            _ = [self.assign_T1(coord, val=self.fixed_t1 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
            _ = [self.assign_T2(coord, val=self.fixed_t2 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
        elif self.defective_qubits == 2: # Z observables
            self.all_defective_qubits = z_observable
            _ = [self.assign_T1(coord, val=self.fixed_t1 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
            _ = [self.assign_T2(coord, val=self.fixed_t2 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
        elif self.defective_qubits == 3: # Random
            idxs = np.random.choice(range(len(all_coords)), self.d, replace=False)
            self.all_defective_qubits = [all_coords[i] for i in idxs]
            _ = [self.assign_T1(coord, val=self.fixed_t1 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]
            _ = [self.assign_T2(coord, val=self.fixed_t2 * (1 - self.defective_ratio)) for coord in self.all_defective_qubits]

        # Pass coords of this logical qubit to the final circuit synthesizer
        mappings = {}
        mappings['data_coords'] = data_coords
        mappings['x_observable'] = x_observable
        mappings['z_observable'] = z_observable
        mappings['x_measure_coords'] = x_measure_coords
        mappings['z_measure_coords'] = z_measure_coords
        return mappings
    
    def coords_to_index_mapper(self, coords:dict) -> None:
        coords_to_index = lambda coord: int(((coord[0] + coord[1]) * 
                                             (coord[0] + coord[1] + 1)) // 
                                             2 + coord[1]) # Cantor pairing
        coords_index_mapper = {}
        index_coords_mapper = {}
        measure_idxs = []
        data_idxs = []
        self.qubit_patch_mapper = {}
        self.patch_qubit_mapper = {}
        patch = 0
        if type(coords) != dict:
            raise ValueError('Expected a dictionary of coordinate dictionaries per logical qubit')
        # Proceed through every stage of the surface code cycle for every qubit all at once. 
        for mappings in coords.values():
            if type(mappings) != dict:
                raise ValueError('Expected a dictionary of coordinate dictionaries per logical qubit')
            data_coords = mappings['data_coords']
            x_measure_coords = mappings['x_measure_coords']
            z_measure_coords = mappings['z_measure_coords']
            all_coords = data_coords + x_measure_coords + z_measure_coords
            measure_idxs += [coords_to_index(c) for c in x_measure_coords + z_measure_coords]
            data_idxs += [coords_to_index(c) for c in data_coords]
            for c in all_coords:
                idx = coords_to_index(c)
                coords_index_mapper[c] = idx
                index_coords_mapper[idx] = c
                self.qubit_patch_mapper[idx] = patch
                self.patch_qubit_mapper[patch] = [idx] if patch not in self.patch_qubit_mapper.keys() \
                    else self.patch_qubit_mapper[patch] + [idx] 
            patch += 1
        self.coords_index_mapper = coords_index_mapper # Save for the future
        self.index_coords_mapper = index_coords_mapper
        self.measure_idxs = measure_idxs
        self.data_idxs = data_idxs
        self.add_skew()
        return

    def get_qubit_idxs(self) -> list:
        return list(self.index_coords_mapper.keys())
    
    def get_qubit_coords(self) -> list:
        return list(self.coords_index_mapper.keys())
    
    def get_measure_qubit_coords(self) -> list:
        return [self.index_coords_mapper[idx] for idx in self.measure_idxs]
    
    def get_patch_idxs(self) -> list:
        return list(self.patch_qubit_mapper.keys())
    
    # A tick represents the start of a new frame. 
    def tick(self, str: str) -> str:
        t = str + 'TICK\n'
        return t 
    
    def __X():
        return 'X'
    
    def __Z():
        return 'Z'
    
    def reset_data(self, mappings:dict) -> str:
        ckt_str = ''
        data_coords = mappings['data_coords']
        data_idx = [self.coords_index_mapper[i] for i in data_coords]
        if self.basis == 'Z': # replace with condition for X or Z basis experiment
            reset_data = 'R'
        else:
            reset_data = 'RX'
        ckt_str += reset_data + ' ' + ' '.join(str(i) for i in data_idx) + "\n"
        if self.init_error != None:
            if self.basis == 'Z':
                ckt_str += 'X_ERROR(%f) '%(self.init_error) + ' '.join(str(i) for i in data_idx) + "\n"
            else:
                ckt_str += 'Z_ERROR(%f) '%(self.init_error) + ' '.join(str(i) for i in data_idx) + "\n"
            pass
        return ckt_str
    
    def reset_ancilla(self, mappings:dict) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
        z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
        reset_measure = 'R'
        ckt_str += reset_measure + ' ' + ' '.join(str(i) for i in x_measure_idx + z_measure_idx) + "\n"
        if self.init_error != None:
            ckt_str += 'X_ERROR(%f) '%(self.init_error) + ' '.join(str(i) 
                                                                   for i in x_measure_idx + z_measure_idx) + "\n"
            pass
        return ckt_str
    
    def add_gate(self, gate:str, phys_qubit:int, target_qubit:int=None) -> str:
        # Return a string specifying the gate and target qubit, and the noise annotation
        # for that gate.
        # Lookup the gate error for this phys_qubit from the noise profile
        # 1Q gates:
        noise = '\nDEPOLARIZE1(%f) %i'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit) if self.disable_noise == False else ''
        str = gate + ' %i'%(phys_qubit) + noise + "\n"
        # 2Q gates:
        if target_qubit != None:
            noise = '\nDEPOLARIZE2(%f) %i %i'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit, target_qubit) if self.disable_noise == False else ''
            str = gate + ' %i %i'%(phys_qubit, target_qubit) + noise + "\n"
        if gate in self.measures:
            noise = 'X_ERROR(%f) %i\n'%(self.profile[self.index_coords_mapper[phys_qubit]]['error'][gate], phys_qubit) if self.disable_noise == False else ''
            str = noise + gate + ' %i\n'%(phys_qubit)
            pass
        return str
    
    def hadamard_stage(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        for mappings in coords.values():
            x_measure_coords = mappings['x_measure_coords']
            x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
            ckt_str += ''.join(self.add_gate('H', i) for i in x_measure_idx)
        return ckt_str
    
    def cnot_stage(self, coords:dict) -> str:
        interaction_order_z = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        interaction_order_x = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        cxs = []
        str = ''
        for i in range(len(interaction_order_x)):
            ckt_str = []
            for mappings in coords.values():
                data_coords = mappings['data_coords']
                x_measure_coords = mappings['x_measure_coords']
                z_measure_coords = mappings['z_measure_coords']
                for q in x_measure_coords:
                    c = tuple((q[0] + interaction_order_x[i][0], q[1] + interaction_order_x[i][1]))
                    if c in data_coords:
                        ckt_str.append(self.add_gate('CX', self.coords_index_mapper[q], 
                                                     self.coords_index_mapper[c]))
                        if c in self.new_datas:
                            self.measure_new_data_mapper[q] = [c] if q not in self.measure_new_data_mapper.keys() else self.measure_new_data_mapper[q] + [c]
                    pass
                for q in z_measure_coords:
                    c = tuple((q[0] + interaction_order_z[i][0], q[1] + interaction_order_z[i][1]))
                    if c in data_coords:
                        ckt_str.append(self.add_gate('CX', self.coords_index_mapper[c], 
                                                     self.coords_index_mapper[q]))
                        if c in self.new_datas:
                            self.measure_new_data_mapper[q] = [c] if q not in self.measure_new_data_mapper.keys() else self.measure_new_data_mapper[q] + [c]
                    pass
                pass
            cxs.append(ckt_str)
        for i in range(len(interaction_order_x)):
            str = self.tick(str)
            str += ''.join(cx for cx in cxs[i])
        return str
    
    def measure_ancilla(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        self.pre_merge_measurements = {}
        all_measurements = []
        for patch, mappings in enumerate(coords.values()):
            x_measure_coords = mappings['x_measure_coords']
            z_measure_coords = mappings['z_measure_coords']
            x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
            z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
            measure_qubits = x_measure_idx + z_measure_idx
            ckt_str += ''.join(self.add_gate('MR', i) for i in measure_qubits)
            # Track the measurement index
            self.measurement_tracker += x_measure_coords + z_measure_coords
            all_measurements += measure_qubits
        self.pre_merge_measurements = {self.index_coords_mapper[j]:i - len(all_measurements) for i, j in enumerate(all_measurements)}
        return ckt_str
    
    def measure_merged_ancilla(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        all_measurements = []
        for patch, mappings in enumerate(coords.values()):
            x_measure_coords = mappings['x_measure_coords']
            z_measure_coords = mappings['z_measure_coords']
            x_measure_idx = [self.coords_index_mapper[i] for i in x_measure_coords]
            z_measure_idx = [self.coords_index_mapper[i] for i in z_measure_coords]
            measure_qubits = x_measure_idx + z_measure_idx
            ckt_str += ''.join(self.add_gate('MR', i) for i in measure_qubits)
            # Track the measurement index
            self.measurement_tracker += x_measure_coords + z_measure_coords
            all_measurements += measure_qubits
        self.post_merge_measurements = {self.index_coords_mapper[j]:i - len(all_measurements) for i, j in enumerate(all_measurements)}
        return ckt_str
    
    def measure_merged_data(self) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        gate = 'M' if self.ls_basis == 'Z' else 'MX'
        data_coords = self.new_datas
        data_idx = [self.coords_index_mapper[i] for i in data_coords]
        # self.measurement_tracker += data_coords
        self.merged_data_measure_tracker = {i:j for j, i in enumerate(data_coords)}
        ckt_str += ''.join(self.add_gate(gate, i) for i in data_idx)
        self.measure_out = True
        return ckt_str
    
    def measure_data(self, coords:dict) -> str:
        ckt_str = ''
        ckt_str = self.tick(ckt_str)
        gate = 'M' if self.basis == 'Z' else 'MX'
        for mappings in coords.values():
            data_coords = mappings['data_coords']
            data_idx = [self.coords_index_mapper[i] for i in data_coords]
            self.measurement_tracker += data_coords
            ckt_str += ''.join(self.add_gate(gate, i) for i in data_idx)
        return ckt_str
    
    def detector_first_round(self, mappings:dict) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {}
        if self.basis == 'Z': # substitute with appropriate condition for x/z basis experiment
            indices = {i:self.measurement_tracker.index(i) - 
                       len(self.measurement_tracker) 
                       for i in z_measure_coords}
        else:
            indices = {i:self.measurement_tracker.index(i) - 
                       len(self.measurement_tracker) 
                       for i in x_measure_coords}
        ckt_str += '\n'.join('DETECTOR(%i, %i, 0) rec[%i]'%(\
            i[0], i[1], indices[i]) for i in indices.keys()) + '\n'
        return ckt_str
    
    def detector_first_round_post_merge(self, mappings:dict, logical_qubits:int) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {i:self.measurement_tracker.index(i) - 
                   len(self.measurement_tracker) 
                   for i in x_measure_coords + z_measure_coords}
        for i in indices.keys():
            if i in self.measure_new_data_mapper.keys():
                datas = self.measure_new_data_mapper[i]
                assert len(datas) == 2, "Shouldn't be more than 2 here"
                idx0 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[0]])
                idx1 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[1]])
                ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i] rec[%i] rec[%i]\n'%(\
                            i[0], i[1], indices[i], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i], idx0, idx1)
                pass
            else:
                ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i]\n'%(\
                            i[0], i[1], indices[i], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i])
        # # Stabilizers that were removed due to split
        # if self.measure_out:
        #     self.measure_out = False
        #     if self.ls_basis == 'Z':
        #         for i in self.new_xmeasure:
        #             # Len of datas can be 2/1
        #             datas = self.measure_new_data_mapper[i]
        #             if len(datas) == 2:
        #                 idx0 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[0]])
        #                 idx1 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[1]])
        #                 ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i] rec[%i]\n'%(\
        #                         i[0], i[1], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i], idx0, idx1)
        #                 pass
        #             else:
        #                 idx0 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[0]])
        #                 ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i]\n'%(\
        #                         i[0], i[1], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i], idx0)
        #                 pass
        #             pass
        #         pass
        #     else:
        #         for i in self.new_zmeasure:
        #             # Len of datas can be 2/1
        #             datas = self.measure_new_data_mapper[i]
        #             if len(datas) == 2:
        #                 idx0 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[0]])
        #                 idx1 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[1]])
        #                 ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i] rec[%i]\n'%(\
        #                         i[0], i[1], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i], idx0, idx1)
        #                 pass
        #             else:
        #                 idx0 = -len(self.measurement_tracker) - (len(list(self.merged_data_measure_tracker.keys())) - self.merged_data_measure_tracker[datas[0]])
        #                 ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i]\n'%(\
        #                         i[0], i[1], -len(self.measurement_tracker) - len(list(self.merged_data_measure_tracker.keys())) + self.post_merge_measurements[i], idx0)
        #                 pass
        #             pass
        #         pass
        return ckt_str
    
    def merged_detector_first_round(self, mappings:dict) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {}
        for i in z_measure_coords:
            indices[i] = self.measurement_tracker.index(i) - len(self.measurement_tracker)
            if i not in self.pre_merge_measurements:
                if self.ls_basis == 'Z':
                    ckt_str += 'DETECTOR(%i, %i, 0) rec[%i]\n'%(i[0], i[1], indices[i])
                pass
            else:
                ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i]\n'%(i[0], i[1], indices[i], self.pre_merge_measurements[i] - len(self.measurement_tracker))
                pass
            pass
        pass
        for i in x_measure_coords:
            indices[i] = self.measurement_tracker.index(i) - len(self.measurement_tracker)
            if i not in self.pre_merge_measurements:
                if self.ls_basis == 'X':
                    ckt_str += 'DETECTOR(%i, %i, 0) rec[%i]\n'%(i[0], i[1], indices[i])
                pass
            else:
                ckt_str += 'DETECTOR(%i, %i, 0) rec[%i] rec[%i]\n'%(i[0], i[1], indices[i], self.pre_merge_measurements[i] - len(self.measurement_tracker))
                pass
            pass
        ckt_str += '\n'
        return ckt_str
    
    def add_detectors(self, mappings:dict, logical_qubits:int) -> str:
        ckt_str = ''
        x_measure_coords = mappings['x_measure_coords']
        z_measure_coords = mappings['z_measure_coords']
        indices = {i:self.measurement_tracker.index(i) - 
                   len(self.measurement_tracker) 
                   for i in x_measure_coords + z_measure_coords}
        ckt_str += '\n'.join('DETECTOR(%i, %i, 0) rec[%i] rec[%i]'%(\
            i[0], i[1], indices[i], indices[i] - logical_qubits * \
                len(x_measure_coords + z_measure_coords)) for i in indices.keys()) \
                    + '\n'
        return ckt_str
    
    def final_detectors(self, mappings:dict) -> str:
        interaction_order_z = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        interaction_order_x = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        data_coords = mappings['data_coords']
        z_measure_coords = mappings['z_measure_coords']
        x_measure_coords = mappings['x_measure_coords']
        # Last len(data_coords) measurements in measurement_tracker correspond to the data qubits
        if self.basis == 'Z':
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in data_coords + z_measure_coords}
            ckt_str = ''
            for coord in z_measure_coords:
                rec = ''
                for i in interaction_order_z:
                    c = tuple((coord[0] + i[0], coord[1] + i[1]))
                    if c in data_coords:
                        rec += 'rec[%i] '%(indices[c])
                    pass
                rec += 'rec[%i]\n'%(indices[coord])
                ckt_str += 'DETECTOR(%i, %i, 1) '%(coord[0], coord[1]) + rec
        else:
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in data_coords + x_measure_coords}
            ckt_str = ''
            for coord in x_measure_coords:
                rec = ''
                for i in interaction_order_x:
                    c = tuple((coord[0] + i[0], coord[1] + i[1]))
                    if c in data_coords:
                        rec += 'rec[%i] '%(indices[c])
                    pass
                rec += 'rec[%i]\n'%(indices[coord])
                ckt_str += 'DETECTOR(%i, %i, 1) '%(coord[0], coord[1]) + rec
        return ckt_str
    
    def add_observables(self, mappings:dict, qubit:int) -> str:
        z_observables = mappings['z_observable']
        x_observables = mappings['x_observable']
        ckt_str = ''
        if self.basis == 'Z':
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in z_observables}
        else:
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in x_observables}
        ckt_str += 'OBSERVABLE_INCLUDE(%i) '%(qubit)
        if self.basis == 'Z':
            for i in z_observables:
                ckt_str += 'rec[%i] '%(indices[i])
        else:
            for i in x_observables:
                ckt_str += 'rec[%i] '%(indices[i])
        ckt_str += '\n'
        return ckt_str
    
    def add_observables_post_merge(self, coords:dict) -> str:
        ckt_str = 'OBSERVABLE_INCLUDE(%i) '%(0)
        for mappings in coords.values():
            z_observables = mappings['z_observable']
            x_observables = mappings['x_observable']
            if self.basis == 'Z':
                indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                        for i in z_observables}
            else:
                indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                        for i in x_observables}
            if self.basis == 'Z':
                for i in z_observables:
                    ckt_str += 'rec[%i] '%(indices[i])
            else:
                for i in x_observables:
                    ckt_str += 'rec[%i] '%(indices[i])
        ckt_str += 'rec[%i]'%(-int(len(list(self.merged_data_measure_tracker.keys()))/2) - 1 - 2 * self.num_datas - 2 * self.num_measures * self.rounds_per_op)
        ckt_str += '\n'
        return ckt_str
    
    def add_observables_merged(self, mappings:dict, qubit:int) -> str:
        z_observables = mappings['z_observable']
        x_observables = mappings['x_observable']
        ckt_str = ''
        if self.basis == 'Z':
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in self.zobs}
        else:
            indices = {i:self.measurement_tracker.index(i) - len(self.measurement_tracker) 
                    for i in self.xobs}
        ckt_str += 'OBSERVABLE_INCLUDE(%i) '%(qubit)
        if self.basis == 'Z':
            for i in self.zobs:
                ckt_str += 'rec[%i] '%(indices[i])
        else:
            for i in self.xobs:
                ckt_str += 'rec[%i] '%(indices[i])
        ckt_str += '\n'
        return ckt_str
    
    def setup_patches(self, coords:dict) -> str:
        # Take coords for all patches (logical qubits) and generate interactions
        # In case there are future qubit-qubit interactions, add the circuits for those later
        # Add only the first and d-1 rounds for the initial logical qubits. Save the 
        # last detector annotations in case there are no logical operations (memory experiment)
        self.coords_to_index_mapper(coords)

        merge = self.merge_ls
        merged_reset = ''
        merged_mapper = {}
        if True:
            mappings = self.merge(coords=coords)
            new_coords = {0:mappings}
            self.coords_to_index_mapper(new_coords)
            merged_mapper = self.coords_index_mapper
            merged_reset += self.reset_merged_data_qubits(mappings=mappings, old_coords=coords)
            merged_reset += self.reset_merged_measure_qubits(mappings=mappings, old_coords=coords)
        else:
            merged_mapper = self.coords_index_mapper

        # Specify qubit coordinates for Stim
        ckt_str = ''.join("QUBIT_COORDS(%i, %i) %i\n"%(\
            coord[0], coord[1], merged_mapper[coord]) 
                          for coord in merged_mapper.keys())
        rep_ckt = ''

        # Reset phase
        for mappings in coords.values():
            ckt_str += self.reset_data(mappings)
            ckt_str += self.reset_ancilla(mappings)
        
        if merge:
            ckt_str += merged_reset

        # stim can take gates with multiple targets, but to make noise
        # annotations simpler, specify gates for each qubit in a new line. 

        # Hadamard phase
        rep_ckt += self.hadamard_stage(coords)

        # CNOT phase
        rep_ckt += self.cnot_stage(coords)

        # Hadamard phase
        rep_ckt += self.hadamard_stage(coords)

        # Measurement phase
        rep_ckt += self.measure_ancilla(coords)

        # TODO: 2 things:
        # - What if there's a logical operation on a qubit while the other qubits are free? 
        #   - How to grow the circuit for this case? (And other generic cases)
        # - How to add idling error to the final circuit? (will be a full parse of the circuit)

        # For lattice surgery, the coords of a patch will change, and so the same functions
        # above can be called for the new set of coords (Think in terms of split and merge: 
        # A split will result in a new logical with it's own set of coords, a merge will 
        # result in two logicals being fused. In the case of a merge, the first detector 
        # annotations will be slightly different since the qubits in the gap between the 
        # two logicals will be measured.)

        # Detector phase
        # For the first round, the detector just specifies the measurements that are deterministic
        ckt_str += rep_ckt
        for mappings in coords.values():
            ckt_str += self.detector_first_round(mappings)
        # ckt_str += 'REPEAT %i {\n'%(self.rounds_per_op) # REPEAT does not help wrt time -- unrolled is better
        rep_ckt += 'SHIFT_COORDS(0, 0, 1)\n'
        for mappings in coords.values():
            rep_ckt += self.add_detectors(mappings, len(list(coords.values())))
            pass
        for _ in range(self.rounds_per_op - 1): # Unroll the loop for all rounds
            ckt_str += rep_ckt
        # ckt_str += '}\n'

        rep_ckt = ''
        if merge:
            self.merging = True
            mappings = self.merge(coords=coords)
            new_coords = {0:mappings}
            self.coords_to_index_mapper(new_coords)
            # ckt_str += self.reset_merged_data_qubits(mappings=mappings, old_coords=coords)
            # ckt_str += self.reset_merged_measure_qubits(mappings=mappings, old_coords=coords)
            # Hadamard phase
            rep_ckt += self.hadamard_stage(new_coords)

            # CNOT phase
            rep_ckt += self.cnot_stage(new_coords)

            # Hadamard phase
            rep_ckt += self.hadamard_stage(new_coords)

            self.measurement_tracker = []
            # Measurement phase
            rep_ckt += self.measure_merged_ancilla(new_coords)
            ckt_str += rep_ckt

            ckt_str += 'SHIFT_COORDS(0, 0, 1)\n'
            ckt_str += self.merged_detector_first_round(mappings)
            rep_ckt += 'SHIFT_COORDS(0, 0, 1)\n'
            rep_ckt += self.add_detectors(mappings=mappings, logical_qubits=len(list(new_coords.values())))
            for _ in range(self.rounds_per_op - 1):
                ckt_str += rep_ckt
            # coords = new_coords
            self.merging = False
                
            # Now split the merged patch back into original patches
            self.measurement_tracker = []
            ckt_str += self.measure_merged_data()
            rep_ckt = self.hadamard_stage(coords)
            rep_ckt += self.cnot_stage(coords)
            rep_ckt += self.hadamard_stage(coords)
            rep_ckt += self.measure_ancilla(coords)
            ckt_str += rep_ckt
            ckt_str += 'SHIFT_COORDS(0, 0, 1)\n'
            for mappings in coords.values():
                ckt_str += self.detector_first_round_post_merge(mappings, len(list(new_coords.values())))
            rep_ckt += 'SHIFT_COORDS(0, 0, 1)\n'
            for mappings in coords.values():
                rep_ckt += self.add_detectors(mappings, len(list(coords.values())))
                pass
            for _ in range(self.rounds_per_op - 1): # Unroll the loop for all rounds
                ckt_str += rep_ckt
            
        # Final measurement of data qubits and detector annotations
        epilogue = ''
        epilogue += self.measure_data(coords)
        for mappings in coords.values():
            epilogue += self.final_detectors(mappings)
        
        # Add observables
        for q in coords.keys():
            # These are the default observables without any logical interactions
            obs = self.add_observables(coords[q], q)
            pass
        
        # Add observables
        if merge == False:
            for q in coords.keys():
                # These are the default observables without any logical interactions
                epilogue += obs
                pass
        else:
            if self.ls_basis == self.basis:
                epilogue += self.add_observables_post_merge(coords=coords)
            else:
                epilogue += obs
        epilogue = self.tick(epilogue)
        self.default_epilogue = epilogue
        self.init_ckt = ckt_str

        ckt_str += epilogue

        ckt_str = self.fenced_sched(ckt_str)
        return ckt_str

    def __lattice_merge(self, coords:dict) -> None:
        return
    
    def __lattice_split(self, coords:dict) -> None:
        return
    
    def add_skew(self) -> None:
        # Add skew between qubits for all gates.
        # Skew is wrt a controllers that can be considered the master
        # So some qubits will have no skew, others will have some predefined skew
        # The skew could be constant in time (not very realistic?) or vary with
        # time. Skew could also be +ve or -ve.
        # This skew would be automatically added to the idle period before a gate
        # is executed for a qubit. 
        self.skew_map = {i:0 for i in self.index_coords_mapper.keys()}
        # # Skew for staggering instructions between logical patches
        # stagger_time = 5
        # self.stagger_patches = {i:stagger_time * self.qubit_patch_mapper[i] for i in self.index_coords_mapper.keys()}
        # self.skew_map = self.stagger_patches
        # Skew between controllers
        per_logical_controller = False
        phys_qubits_per_controller = 100
        # In case a fixed number of qubits are controlled by a single controllers,
        # the comtrollers will control different square patches of qubits (rather than linear chains)
        # for coords in self.coords_index_mapper.keys():
        #     pass
        # skew introduced because of variable decoding latency
        return
    
    def fenced_sched(self, ckt:str) -> str:
        # Convert to array
        ckt = np.array([line for line in ckt.split('\n')], dtype='object')
        new_ckt = ckt
        ticks = np.where(ckt == 'TICK')[0] # Find indices of all TICKS
        # Find sub-ckt between every 2 TICKS
        intervals = np.lib.stride_tricks.sliding_window_view(ticks, 2)
        inst_cnt = {}
        idx_offset = 0
        round_counter = 0
        hcounter = 0
        frame_start = 0
        sync_counter = 0
        for interval in intervals:
            timeline = {}
            insts_debug = {}
            snip = ckt[interval[0] + 1:interval[1]]
            # This is the sub-circuit executed for all logical qubits in parallel
            max_latency = 0
            qubits = []
            skew_preamble = {}
            last_gate = None
            h = (hcounter % 2)
            if 'H' in ckt[interval[0] + 1]:
                if h == 0:
                    frame_start = interval[0]
                hcounter += 1
            for inst in snip:
                split = inst.split(' ')
                gate = split[0]
                if len(split) < 2 or gate not in self.gates:
                    # Nothing useful
                    continue
                last_gate = gate
                qubit = int(split[1])
                latency = self.profile[self.index_coords_mapper[qubit]]['latency'][gate]
                skew = self.skew_map[qubit]
                # This max latency determines the duration of this TICK
                max_latency = max(max_latency, latency + skew)
                qubits.append(qubit)
                # Construct timeline for all measure qubits in the lattice
                # Will need to account for skew here
                target = None
                timeline[qubit] = [tuple((self.time, self.time + latency + skew))] \
                    if qubit not in timeline.keys() \
                    else timeline[qubit] + [tuple((self.time, self.time + latency + skew))]
                insts_debug[self.time] = [gate] if self.time not in insts_debug.keys() else insts_debug[self.time] + [gate]
                if gate in self.gates_2Q:
                    target = int(split[2])
                    qubits.append(target)
                    timeline[target] = [tuple((self.time, self.time + latency + skew))] \
                        if target not in timeline.keys() \
                        else timeline[target] + [tuple((self.time, self.time + latency + skew))]
                    pass
                # Skew contributes to idling errors before the gate, slack 
                # contributes to idling errors after a gate. 
                skew_preamble[qubit] = skew
                if target != None:
                    skew_preamble[target] = self.skew_map[target]
                pass
            if last_gate in self.measures and last_gate not in self.measures_logical:
                round_counter += 1
            addl_latency = 0 # self.decode_latency if round_counter % self.sync_round == 0 and round_counter > 0 and last_gate != 'M' else 0
            self.time += max_latency + addl_latency
            insts_conc = {time:len(insts_debug[time]) 
                          for time in insts_debug.keys()}
            inst_cnt = {**inst_cnt, **insts_conc}
            # The qubits not included above will incur an idling error equal to
            # the total time of the tick
            post_time = {qubit:self.time - timeline[qubit][-1][1]
                         for qubit in qubits}
            rem = {qubit:max_latency 
                   for qubit in np.setdiff1d(self.measure_idxs + self.data_idxs,
                                             qubits)}
            post_time = {**post_time, **rem}
            self.total_idle += np.sum(list(post_time.values())) + np.sum(list(skew_preamble.values()))
            post_noise = '\n'.join(self.idling_model(post_time[qubit], 
                                                     self.index_coords_mapper[qubit], qubit) 
                                                     for qubit in post_time.keys())
            if self.disable_noise == False:
                new_ckt = np.insert(new_ckt, 
                                    interval[0] + 1 + idx_offset, 
                                    '\n'.join(self.idling_model(skew_preamble[qubit], 
                                                                self.index_coords_mapper[qubit], qubit) 
                                                                for qubit in skew_preamble.keys()))
                idx_offset += 1
                new_ckt = np.insert(new_ckt, 
                                    interval[1] + idx_offset, 
                                    post_noise)
                idx_offset += 1
                pass
            pass
        new_ckt = '\n'.join(i for i in new_ckt)
        self.concurrent_insts = inst_cnt
        return new_ckt
    
    def scheduler(self, ckt:str) -> str:
        new_ckt = self.fenced_sched(ckt=ckt)
        return new_ckt

    def synthesize(self, cmds:dict) -> None:
        # Main synthesis function. Will call other functions based on the input 
        # circuit
        # Generate initial circuit for all logical qubits.
        coords = {i:self.map_qubit(i) for i in cmds.keys()}
        ckt = self.setup_patches(coords)
        self.ckt = ckt
        # print(ckt + self.default_epilogue)
        # Iterate through logical operations.
        return ckt

    def get_error_rate(self, ckt:str, num_shots:int=1_000_000) -> Tuple[float, np.ndarray]:
        num_errors, errors_per_logical = self.count_logical_errors(ckt, num_shots)
        return num_errors / num_shots, errors_per_logical / num_shots
    
    def from_string(self, ckt:str) -> None:
        cmds = circuit_parser.from_string(self, ckt)
        self.measurement_tracker = []
        self.synthesize(cmds)
        return self

    def from_file(self, ckt:str) -> None:
        cmds = circuit_parser.from_file(self, ckt)
        self.measurement_tracker = []
        self.synthesize(cmds)
        return self
    
    pass

if __name__ == "__main__":
    d = 3
    err = 0.001
    ls_basis = 'X'
    basis = 'Z'
    slack = 100
    sim = circuit(distance=d, num_patches_y=20, num_patches_x=20, spacing=1, disable_noise=False, fixed_t1=250, fixed_t2=150, fixed_cnot_latency=200, fixed_measure_latency=500, fixed_cnot_noise=err, fixed_measure_noise=err, rounds_per_op=d+1, idle_multiplier=3, basis=basis, ls_basis=ls_basis, merge=True).from_string('qreg q[2];')
    
    e, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=100_000)
    # print(new_ckt)
    # print(sim.skew_map)
    print(sim.time, sim.total_idle, e)
    