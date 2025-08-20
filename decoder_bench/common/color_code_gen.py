# Derived from seokhyung-lee/color-code-stim

import math
from typing import Union, Optional, Iterable

import igraph as ig
import stim

class ColorCode:
    tanner_graph: ig.Graph
    circuit: stim.Circuit
    d: int
    rounds: int
    qubit_groups: dict

    def __init__(self,
                 *,
                 d: int,
                 rounds: int,
                 shape: str = 'triangle',
                 cnot_schedule: Union[str, Iterable[int]] = 'BKS',
                 p_bitflip: float = 0.,
                 p_reset: float = 0.,
                 p_meas: float = 0.,
                 p_cnot: float = 0.,
                 p_idle: float = 0.,
                 p_circuit: Optional[float] = None,
                 # custom_noise_channel: Optional[Tuple[str, object]] = None,
                 # dem_decomposed: Optional[Dict[str, Tuple[
                 #     stim.DetectorErrorModel, stim.DetectorErrorModel]]] = None,
                 benchmarking: bool = False,
                 ):
        """
        Class for constructing a color code circuit and simulating the
        concatenated MWPM decoder.

        Parameters
        ----------
        d : int >= 3
            Code distance. Should be an odd number of 3 or more.
        rounds : int >= 1
            Number of syndrome extraction rounds.
        shape : {'triangle'}, default 'triangle'
            Shape of the color code patch. Currently support only triangular patch.
        cnot_schedule : {12-tuple of integers, 'LLB', 'BKS'}, default 'LLB'
            CNOT schedule.
            If this is a 12-tuple of integers, it indicate (a, b, ... l)
            specifying the CNOT schedule.
            If this is 'LLB', it is (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2),
            which is the optimal schedule from our paper.
            If this is 'LLB_reversed', it is (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1),
            which has the X- and Z-part reversed from 'LLB'.
            If this is 'BKS', it is (4, 1, 2, 3, 6, 5, 3, 2, 5, 6, 7, 4),
            which is the optimal schedule from [Beverland et al.,
            PRXQuantum.2.020341].
        p_bitflip : float, default 0
            Bit-flip probability at the start of each round.
        p_reset : float, default 0
            Probability of a wrong qubit reset (i.e., producing an
            orthogonal state).
        p_meas : float, default 0
            Probability of a flipped measurement outcome.
        p_cnot : float, default 0
            Strength of a two-qubit depolarizing noise channel following
            each CNOT gate.
        p_idle : float, default 0
            Strength of a single-qubit depolarizing noise channel following
            each idle gate.
        p_circuit : float, optional
            If given, p_reset = p_meas = p_cnot = p_idle = p_circuit.
        benchmarking : bool, default False
            Whether to measure execution time of each step.
        """
        if isinstance(cnot_schedule, str):
            if cnot_schedule == 'BKS':
                cnot_schedule = (4, 1, 2, 3, 6, 5, 3, 2, 5, 6, 7, 4)
            elif cnot_schedule == 'LLB':
                cnot_schedule = (2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2)
            elif cnot_schedule == 'LLB_reversed':
                cnot_schedule = (3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1)
            else:
                raise ValueError
        else:
            assert len(cnot_schedule) == 12

        assert d >= 3 and rounds >= 1
        assert shape in ['triangle']

        if p_circuit is not None:
            p_reset = p_meas = p_cnot = p_idle = p_circuit

        self.d = d
        self.rounds = rounds
        self.shape = shape
        self.cnot_schedule = cnot_schedule
        self.probs = {
            'bitflip': p_bitflip,
            'reset': p_reset,
            'meas': p_meas,
            'cnot': p_cnot,
            'idle': p_idle
        }
        # self.custom_noise_channel = custom_noise_channel

        self.tanner_graph = ig.Graph()
        self.circuit = stim.Circuit()

        self.benchmarking = benchmarking

        # Mapping between detector ids and ancillary qubits
        # self.detectors[detector_id] = (anc_qubit, time_coord)
        self.detectors = []

        # Detector ids grouped by colors
        self.detector_ids = {'r': [], 'g': [], 'b': []}

        # Various qubit groups
        self.qubit_groups = {}

        # Decomposed detector error models
        # It is generated when required.
        # if dem_decomposed is None:
        #     dem_decomposed = {}
        self.dems_decomposed = {}

        tanner_graph = self.tanner_graph

        self._add_tanner_graph_vertices()

        data_qubits = tanner_graph.vs.select(pauli=None)
        anc_qubits = tanner_graph.vs.select(pauli_ne=None)
        anc_Z_qubits = anc_qubits.select(pauli='Z')
        anc_X_qubits = anc_qubits.select(pauli='X')
        anc_red_qubits = anc_qubits.select(color='r')
        anc_green_qubits = anc_qubits.select(color='g')
        anc_blue_qubits = anc_qubits.select(color='b')

        self.qubit_groups.update({
            'data': data_qubits,
            'anc': anc_qubits,
            'anc_Z': anc_Z_qubits,
            'anc_X': anc_X_qubits,
            'anc_red': anc_red_qubits,
            'anc_green': anc_green_qubits,
            'anc_blue': anc_blue_qubits
        })

        self._generate_circuit()

        # Get detector list
        detector_coords_dict = self.circuit.get_detector_coordinates()
        for detector_id in range(self.circuit.num_detectors):
            coords = detector_coords_dict[detector_id]
            x = math.floor(coords[0])
            y = round(coords[1])
            t = round(coords[2])
            try:
                name = f'{x}-{y}-X'
                qubit = tanner_graph.vs.find(name=name)
            except ValueError:
                name = f'{x + 1}-{y}-Z'
                qubit = tanner_graph.vs.find(name=name)
            self.detectors.append((qubit, t))
            self.detector_ids[qubit['color']].append(detector_id)

    def _add_tanner_graph_vertices(self):
        shape = self.shape
        d = self.d
        tanner_graph = self.tanner_graph

        if shape in {'triangle'}:
            assert d % 2 == 1

            detid = 0
            L = round(3 * (d - 1) / 2)
            for y in range(L + 1):
                if y % 3 == 0:
                    anc_qubit_color = 'g'
                    anc_qubit_pos = 2
                elif y % 3 == 1:
                    anc_qubit_color = 'b'
                    anc_qubit_pos = 0
                else:
                    anc_qubit_color = 'r'
                    anc_qubit_pos = 1

                for x in range(y, 2 * L - y + 1, 2):
                    boundary = []
                    if y == 0:
                        boundary.append('r')
                    if x == y:
                        boundary.append('g')
                    if x == 2 * L - y:
                        boundary.append('b')
                    boundary = ''.join(boundary)
                    if not boundary:
                        boundary = None

                    if round((x - y) / 2) % 3 != anc_qubit_pos:
                        tanner_graph.add_vertex(name=f"{x}-{y}",
                                                x=x,
                                                y=y,
                                                qid=tanner_graph.vcount(),
                                                pauli=None,
                                                color=None,
                                                boundary=boundary)
                    else:
                        for pauli in ['Z', 'X']:
                            tanner_graph.add_vertex(name=f"{x}-{y}-{pauli}",
                                                    x=x,
                                                    y=y,
                                                    qid=tanner_graph.vcount(),
                                                    pauli=pauli,
                                                    color=anc_qubit_color,
                                                    boundary=boundary)
                            detid += 1

        else:
            raise ValueError

    def _generate_circuit(self):
        qubit_groups = self.qubit_groups
        cnot_schedule = self.cnot_schedule
        tanner_graph = self.tanner_graph
        circuit = self.circuit
        rounds = self.rounds

        probs = self.probs
        p_bitflip = probs['bitflip']
        p_reset = probs['reset']
        p_meas = probs['meas']
        p_cnot = probs['cnot']
        p_idle = probs['idle']

        # custom_noise_channel = self.custom_noise_channel

        data_qubits = qubit_groups['data']
        # anc_qubits = qubit_groups['anc']
        anc_Z_qubits = qubit_groups['anc_Z']
        anc_X_qubits = qubit_groups['anc_X']

        data_qids = data_qubits['qid']
        # anc_qids = anc_qubits['qid']
        anc_Z_qids = anc_Z_qubits['qid']
        anc_X_qids = anc_X_qubits['qid']

        num_data_qubits = len(data_qids)
        num_anc_Z_qubits = len(anc_Z_qubits)
        num_anc_X_qubits = len(anc_X_qubits)
        num_anc_qubits = num_anc_X_qubits + num_anc_Z_qubits

        num_qubits = tanner_graph.vcount()
        all_qids = list(range(num_qubits))
        all_qids_set = set(all_qids)

        # Syndrome extraction circuit without SPAM
        synd_extr_circuit_without_spam = stim.Circuit()
        for timeslice in range(1, max(cnot_schedule) + 1):
            targets = [i for i, val in enumerate(cnot_schedule)
                       if val == timeslice]
            operated_qids = set()
            for target in targets:
                if target in {0, 6}:
                    offset = (-1, 1)
                elif target in {1, 7}:
                    offset = (1, 1)
                elif target in {2, 8}:
                    offset = (2, 0)
                elif target in {3, 9}:
                    offset = (1, -1)
                elif target in {4, 10}:
                    offset = (-1, -1)
                else:
                    offset = (-2, 0)

                target_anc_qubits \
                    = anc_Z_qubits if target < 6 else anc_X_qubits
                for anc_qubit in target_anc_qubits:
                    data_qubit_x = anc_qubit['x'] + offset[0]
                    data_qubit_y = anc_qubit['y'] + offset[1]
                    data_qubit_name = f"{data_qubit_x}-{data_qubit_y}"
                    try:
                        data_qubit = tanner_graph.vs.find(name=data_qubit_name)
                    except ValueError:
                        continue
                    anc_qid = anc_qubit.index
                    data_qid = data_qubit.index
                    operated_qids.update({anc_qid, data_qid})

                    tanner_graph.add_edge(anc_qid, data_qid)
                    CX_target = [data_qid, anc_qid] if target < 6 \
                        else [anc_qid, data_qid]
                    synd_extr_circuit_without_spam.append('CX', CX_target)
                    if p_cnot > 0:
                        synd_extr_circuit_without_spam.append('DEPOLARIZE2',
                                                              CX_target,
                                                              p_cnot)

            if p_idle > 0:
                idling_qids = list(all_qids_set - operated_qids)
                synd_extr_circuit_without_spam.append("DEPOLARIZE1",
                                                      idling_qids,
                                                      p_idle)

            synd_extr_circuit_without_spam.append("TICK")

        def get_qubit_coords(qubit: ig.Vertex):
            coords = [qubit['x'], qubit['y']]
            if qubit['pauli'] == 'Z':
                coords[0] -= 0.5
            elif qubit['pauli'] == 'X':
                coords[0] += 0.5

            return tuple(coords)

        # Syndrome extraction circuit with measurement & detector
        def get_synd_extr_circuit(first=False):
            synd_extr_circuit = synd_extr_circuit_without_spam.copy()

            synd_extr_circuit.append('MRZ', anc_Z_qids, p_meas)
            for j, anc_qubit in enumerate(anc_Z_qubits):
                lookback = -num_anc_Z_qubits + j
                coords = get_qubit_coords(anc_qubit)
                coords += (0,)
                if first:
                    target = stim.target_rec(lookback)
                else:
                    target = [stim.target_rec(lookback),
                              stim.target_rec(lookback - num_anc_qubits)]
                synd_extr_circuit.append('DETECTOR', target, coords)

            synd_extr_circuit.append('MRX', anc_X_qids, p_meas)
            if not first:
                for j, anc_qubit in enumerate(anc_X_qubits):
                    lookback = -num_anc_X_qubits + j
                    coords = get_qubit_coords(anc_qubit)
                    coords += (0,)
                    target = [stim.target_rec(lookback),
                              stim.target_rec(lookback - num_anc_qubits)]
                    synd_extr_circuit.append('DETECTOR', target, coords)

            if p_reset > 0:
                synd_extr_circuit.append("X_ERROR", anc_Z_qids, p_reset)
                synd_extr_circuit.append("Z_ERROR", anc_X_qids, p_reset)
            if p_idle > 0:
                synd_extr_circuit.append("DEPOLARIZE1",
                                         data_qids,
                                         p_idle)
            if p_bitflip > 0:
                synd_extr_circuit.append("X_ERROR", data_qids, p_bitflip)

            # if custom_noise_channel is not None:
            #     synd_extr_circuit.append(custom_noise_channel[0],
            #                              data_qids,
            #                              custom_noise_channel[1])

            synd_extr_circuit.append("TICK")
            synd_extr_circuit.append("SHIFT_COORDS", (), (0, 0, 1))

            return synd_extr_circuit

        # Main circuit
        for qubit in tanner_graph.vs:
            coords = get_qubit_coords(qubit)
            circuit.append("QUBIT_COORDS", qubit.index, coords)

        # Initialize qubits
        qids_Z_reset = data_qids + anc_Z_qids
        circuit.append("RZ", qids_Z_reset)
        circuit.append("RX", anc_X_qids)

        if p_reset > 0:
            circuit.append("X_ERROR", qids_Z_reset, p_reset)
            circuit.append("Z_ERROR", anc_X_qids, p_reset)

        if p_bitflip > 0:
            circuit.append("X_ERROR", data_qids, p_bitflip)

        # if custom_noise_channel is not None:
        #     circuit.append(custom_noise_channel[0],
        #                    data_qids,
        #                    custom_noise_channel[1])

        circuit.append("TICK")

        circuit += get_synd_extr_circuit(first=True)
        circuit += get_synd_extr_circuit() * (rounds - 1)

        # Final data qubit measurements
        circuit.append("MZ", data_qids, p_meas)
        for j_anc, anc_qubit in enumerate(anc_Z_qubits):
            anc_qubit: ig.Vertex
            ngh_data_qubits = anc_qubit.neighbors()
            lookback_inds \
                = [-num_data_qubits + data_qids.index(q.index) for q in
                   ngh_data_qubits]
            lookback_inds.append(-num_data_qubits - num_anc_qubits + j_anc)
            target = [stim.target_rec(ind) for ind in lookback_inds]
            circuit.append("DETECTOR",
                           target,
                           get_qubit_coords(anc_qubit) + (0,))
        qubits_logical_Z \
            = tanner_graph.vs.select(boundary_in=['r', 'rg', 'rb'], pauli=None)
        lookback_inds = [-num_data_qubits + data_qids.index(q.index)
                         for q in qubits_logical_Z]
        target = [stim.target_rec(ind) for ind in lookback_inds]
        circuit.append("OBSERVABLE_INCLUDE", target, 0)
