# This rare event simulator is used to probe very low logical error rate
# regimes for general QEC codes using the failure spectrum ansatz method
import numpy as np
import stim
from dataclasses import dataclass
from scipy.optimize import curve_fit
from math import comb
from typing import List, Dict
import matplotlib.pyplot as plt
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
        
        self.num_trials_per_w = kwargs.get('num_trials_per_w', 10_000)
        self.weights = kwargs.get('weights', list(np.arange(d, 20 * d, 10)))
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
        Test whether expanded fault e causes logical failure
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
        Compute P(p) using fitted f(w)
        """
        q = p / self.b
        if w_max is None:
            w_max = int(ansatz_params["wc"] * 5)

        P = 0.0
        for w in range(ansatz_params["w0"], w_max):
            f_w = failure_spectrum_ansatz(
                w,
                **ansatz_params
            )
            P += f_w * comb(self.N_expanded, w) * (q**w) * ((1 - q)**(self.N_expanded - w))

        return P

    def simulate(self, ps:List) -> Dict[float, float]:
        spectrum = self.estimate_failure_spectrum(self.weights, 
                                                  {w:self.num_trials_per_w for w in self.weights})
        print('Spectrum...')
        params = self.fit_failure_spectrum(spectrum, self.d // 2)
        results = {}
        for p in ps:
            results[p] = self.logical_error_rate(p, params)
        return results
 
if __name__ == "__main__":
    d = 6
    circuit = gen_qldpc_circuit((d, 0.001, 'Z'))
    decoder = 'bplsd'
    
    sim = RareEventSimulator(d, circuit, decoder)
    results = sim.simulate([0.001, 0.0005, 0.0001, 0.000001, 0.00000001])
    print('Simulated: ', results)
    fig, ax = plt.subplot(1, 1)
    ax.plot(results.keys(), results.values())
    fig.savefig('test.png')