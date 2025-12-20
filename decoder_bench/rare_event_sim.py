# This rare event simulator is used to probe very low logical error rate
# regimes for general QEC codes using the failure spectrum ansatz method
import numpy as np
import stim
from dataclasses import dataclass
from scipy.optimize import curve_fit
from typing import List, Dict
from scipy.special import gammaln, logsumexp
import argparse
import json
import os
import csv
try:
    from .sampler import DecoderRegistry, DecoderState
    from .generator import (gen_qldpc_circuit, 
                            gen_color_circuit, 
                            gen_lattice_surgery_circuit, 
                            gen_surface_circuit)
    import decoders
    from .common.build_circuit import dem_to_check_matrices
except ImportError:
    from sampler import DecoderRegistry, DecoderState
    from generator import (gen_qldpc_circuit, 
                            gen_color_circuit, 
                            gen_lattice_surgery_circuit, 
                            gen_surface_circuit)
    import decoders  # noqa: F401
    from common.build_circuit import dem_to_check_matrices

@dataclass
class ExpandedFaultModel:
    """
    Maps compressed faults -> expanded uniform-weight faults
    """
    H: np.ndarray          # M × Ñ
    A: np.ndarray          # K × Ñ
    multiplicities: np.ndarray  # m_j
    q: float               # uniform probability

    def expanded_size(self):
        return int(np.sum(self.multiplicities))

    def expand_fault(self, e_tilde):
        """
        Expand a compressed fault vector e_tilde ∈ {0,1}^{Ñ}
        into an expanded fault vector e ∈ {0,1}^N.
        """
        expanded = []
        for bit, m in zip(e_tilde, self.multiplicities):
            expanded.extend([bit] * m)
        return np.array(expanded, dtype=np.uint8)

def failure_spectrum_ansatz(
    w,
    w0,
    f0,
    gamma1,
    gamma2,
    wc,
    a
):
    """
    f_ansatz(w) from the paper (Eq. 10)
    """
    if w < w0:
        return 0.0

    ratio = (1 + (w / wc)**2) / (1 + (w0 / wc)**2)
    exponent = (w / w0)**gamma1 * (ratio ** ((gamma2 - gamma1) / 2))

    return a * (1 - np.exp(-(f0 / a) * exponent))


class RareEventSimulator:
    def __init__(self, d:int, circuit:stim.Circuit, decoder:str, **kwargs):
        self.circuit = circuit
        self.d = d

        self.dem = circuit.detector_error_model()
        self.H, self.A, priors = dem_to_check_matrices(self.dem,
                                                       return_col_dict=False)
        
        # Decoder init
        state = DecoderState(check_matrix=self.H,
                             obs_matrix=self.A,
                             priors=priors,
                             circuit=circuit,
                             dem=self.dem)
        decoder_class = DecoderRegistry.get(decoder)
        self.decoder = decoder_class(state, **kwargs)

        self.b = np.max(np.round(priors / np.min(priors)).astype(int))
        self.q = np.min(priors)  # uniform probability

        self.multiplicities = np.round(priors / self.q).astype(int)

        self.model = ExpandedFaultModel(
            H=self.H,
            A=self.A,
            multiplicities=self.multiplicities,
            q=self.q
        )

        self.N_expanded = self.model.expanded_size()
        self.K = self.A.shape[0]
        self.a = 1 - 2**(-self.K)
        
        self.num_trials_per_weight = kwargs.get('num_trials_per_weight', 10_000)
        self.weights = kwargs.get('weights', list(np.arange(d, 20 * d, 10)))
        self.w0 = kwargs.get('w0', int(np.ceil(d / 2)) - 1)
        pass
    
    def sample_fault_of_weight(self, w):
        """
        Sample a compressed fault vector whose expanded weight is w
        """
        e_tilde = np.zeros(len(self.multiplicities), dtype=np.uint8)

        remaining = w
        indices = np.random.permutation(len(self.multiplicities))

        for j in indices:
            if remaining <= 0:
                break

            m = self.multiplicities[j]

            # probability that this compressed fault fires
            # conditional on consuming some expanded weight
            if remaining >= m:
                e_tilde[j] = 1
                remaining -= m
            else:
                # partial inclusion: accept with probability remaining / m
                if np.random.rand() < remaining / m:
                    e_tilde[j] = 1
                    remaining = 0

        return e_tilde
    
    def is_failure(self, e):
        """
        Test whether compressed fault e causes logical failure
        """
        syndrome = self.model.H @ e % 2
        e_hat = self.decoder.decode(syndrome)

        logical_true = self.model.A @ e % 2
        logical_hat  = self.model.A @ e_hat % 2

        return not np.array_equal(logical_true, logical_hat)
    
    def estimate_failure_spectrum(self, weights, trials_per_weight):
        """
        Estimate f(w) for selected weights
        """
        results = {}

        for w in weights:
            failures = 0
            for _ in range(trials_per_weight[w]):
                e = self.sample_fault_of_weight(w)
                if self.is_failure(e):
                    failures += 1

            f_hat = failures / trials_per_weight[w]
            results[w] = {
                "f": f_hat,
                "failures": failures,
                "trials": trials_per_weight[w]
            }

        return results
    
    def fit_failure_spectrum(self, spectrum_data, w0_fixed=None):
        ws = np.array(sorted(spectrum_data.keys()))
        fs = np.array([spectrum_data[w]["f"] for w in ws])

        def fit_func(w, f0, gamma1, gamma2, wc):
            return np.array([
                failure_spectrum_ansatz(
                    wi, w0_fixed, f0, gamma1, gamma2, wc, self.a
                ) for wi in w
            ])

        p0 = [fs[0], 1.0, 2.0, ws[len(ws)//2]]

        popt, _ = curve_fit(
            fit_func,
            ws,
            fs,
            p0=p0,
            bounds=([0, 0, 0, w0_fixed], [1, 10, 10, self.N_expanded])
        )

        return {
            "w0": w0_fixed,
            "f0": popt[0],
            "gamma1": popt[1],
            "gamma2": popt[2],
            "wc": popt[3],
            "a": self.a
        }

    def logical_error_rate(self, p, ansatz_params, w_max=None):
        """
        Compute P(p) using fitted f(w), numerically stable.
        """
        q = p / self.b
        N = self.N_expanded

        if w_max is None:
            w_max = int(ansatz_params["wc"] * 5)

        log_terms = []

        for w in range(ansatz_params["w0"], w_max):
            f_w = failure_spectrum_ansatz(w, **ansatz_params)
            if f_w <= 0:
                continue

            log_binom = (
                gammaln(N + 1)
                - gammaln(w + 1)
                - gammaln(N - w + 1)
            )

            log_term = (
                np.log(f_w)
                + log_binom
                + w * np.log(q)
                + (N - w) * np.log1p(-q)
            )

            log_terms.append(log_term)

        if not log_terms:
            return 0.0

        return float(np.exp(logsumexp(log_terms)))


    def simulate(self, ps:List) -> Dict[float, float]:
        spectrum = self.estimate_failure_spectrum(self.weights, 
                                                  {w:self.num_trials_per_weight for w in self.weights})
        params = self.fit_failure_spectrum(spectrum, self.w0)
        results = {}
        for p in ps:
            results[p] = self.logical_error_rate(p, params)
        return results
 
if __name__ == "__main__":
    """Main entry point for the decoder-bench-rare-event-sim command."""
    parser = argparse.ArgumentParser(description='Rare event simulation with a given QEC code and decoder.')
    
    # Required arguments
    parser.add_argument('--code', type=str, required=True, choices=['surface', 'qldpc', 'color', 'ls'],
                        help='Type of quantum code; use "ls" for a Lattice Surgery dataset')
    parser.add_argument('--basis', type=str, required=True, choices=['x', 'z'],
                        help='Measurement basis (ignored for the color code); Used as the Lattice Surgery basis if code == "ls"')
    parser.add_argument('--distance', '-d', type=int, required=True,
                        help='Code distance')
    parser.add_argument('--depolarization', '-p', type=float, required=True,
                        help='Depolarization/error probability')
    parser.add_argument('--decoder', type=str, required=True,
                        help='Decoder to use (e.g.: "PyMatching", "Relay-BP", "MWPF", "BP-LSD", etc.)')
    
    # Optional arguments
    parser.add_argument('--num_trials_per_weight', type=int, default=10_000,
                        help='Number of trials to run per weight during spectrum estimation ( default: 10,000)')
    parser.add_argument('--weights', type=int, default=None, nargs='+',
                        help='List of weights to sample during spectrum estimation ( default: range(d, 20*d, 10) )')
    parser.add_argument('--ps', type=float, default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                        nargs='+',
                        help='List of physical error rates to estimate logical error rates for (default: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])')
    parser.add_argument('--w0', type=int, default=None,
                        help='Fix w0 parameter during fitting (default: ceil(d/2) - 1)')
    parser.add_argument('--decoder_params', type=str, default=None, 
                      help='JSON string of decoder parameters, e.g. \'{"max_iter": 50}\'')
    parser.add_argument('--decoder_dir', type=str, default=None,
                      help='Directory containing custom decoder implementations')
    
    args = parser.parse_args()
    
    code = args.code.lower()
    basis = args.basis.lower()
    distance = args.distance
    depolarization = args.depolarization
    decoder = args.decoder
    num_trials_per_weight = args.num_trials_per_weight
    weights = args.weights
    ps = args.ps
    w0 = args.w0
    
    if args.decoder_dir:
        num_loaded = DecoderRegistry.load_from_directory(args.decoder_dir)
        print(f"Loaded {num_loaded} custom decoders from {args.decoder_dir}")
    
    kwargs = {}
    if args.decoder_params:
        kwargs = json.loads(args.decoder_params)
        
    kwargs['num_trials_per_weight'] = num_trials_per_weight
    if weights is not None:
        kwargs['weights'] = weights
    if w0 is not None:
        kwargs['w0'] = w0
    
    circuit = None
    if code == 'surface':
        circuit = gen_surface_circuit((distance, depolarization, basis))
    elif code == 'qldpc':
        circuit = gen_qldpc_circuit((distance, depolarization, basis))
    elif code == 'color':
        circuit = gen_color_circuit((distance, depolarization))
    elif code == 'ls':
        circuit = gen_lattice_surgery_circuit((distance, depolarization, basis))
    else:
        raise ValueError(f"Unsupported code type: {code}")
    
    sim = RareEventSimulator(distance, circuit, decoder, **kwargs)
    results = sim.simulate(ps)
    
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "rare_event_simulation_results.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['code', 'decoder', 'p', 'P(p)'])
        for p, logical_error_rate in results.items():
            writer.writerow([f'{code}_{distance}_{basis}_circuit', decoder, p, logical_error_rate])