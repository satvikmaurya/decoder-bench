try:
    from .sampler import Decoder, DecoderRegistry, DecoderState
except ImportError:
    from sampler import Decoder, DecoderRegistry, DecoderState
import numpy as np
from scipy.sparse import csr_matrix

@DecoderRegistry.register(name="belieffind")
class BeliefFindDecoderImpl(Decoder):
    """BeliefFind decoder implementation"""
    
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the BeliefFind decoder"""
        from ldpc.belief_find_decoder import BeliefFindDecoder
        
        self.max_iter = kwargs.get("max_iter", 100)
        self.bp_method = kwargs.get("bp_method", "minimum_sum")
        self.uf_method = kwargs.get("uf_method", "inversion")
        self.omp_thread_count = kwargs.get("omp_thread_count", 1)
        
        self.decoder = BeliefFindDecoder(
            state.check_matrix, 
            error_channel=list(state.priors), 
            max_iter=self.max_iter, 
            bp_method=self.bp_method,
            omp_thread_count=self.omp_thread_count,
            uf_method=self.uf_method
        )
        self.obs_matrix = state.obs_matrix
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome"""
        return self.decoder.decode(syndrome)
    
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        error_prediction = (self.obs_matrix.toarray() @ prediction) % 2
        return not np.any((error_prediction + observable) % 2)

@DecoderRegistry.register(name="bplsd")
class BpLsdDecoderImpl(Decoder):
    """BPLSD decoder implementation"""
    
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the BPLSD decoder"""
        from ldpc.bplsd_decoder import BpLsdDecoder
        
        self.max_iter = kwargs.get("max_iter", 100)
        self.bp_method = kwargs.get("bp_method", "minimum_sum")
        self.osd_method = kwargs.get("osd_method", "osd_cs")
        self.osd_order = kwargs.get("osd_order", 10)
        self.omp_thread_count = kwargs.get("omp_thread_count", 1)
        
        self.decoder = BpLsdDecoder(
            state.check_matrix, 
            error_channel=list(state.priors), 
            max_iter=self.max_iter, 
            bp_method=self.bp_method,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
            omp_thread_count=self.omp_thread_count
        )
        self.obs_matrix = state.obs_matrix
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome"""
        return self.decoder.decode(syndrome)
    
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        error_prediction = (self.obs_matrix.toarray() @ prediction) % 2
        return not np.any((error_prediction + observable) % 2)

@DecoderRegistry.register(name="pymatching")
class PyMatchingDecoderImpl(Decoder):
    """PyMatching decoder implementation"""
    
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the PyMatching decoder"""
        import pymatching
        self.decoder = pymatching.Matching.from_stim_circuit(circuit=state.circuit)
    
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome"""
        return self.decoder.decode(syndrome)
    
    def decode_batch(self, syndromes: np.ndarray) -> list:
        """Decode a batch of syndromes"""
        return self.decoder.decode_batch(syndromes)
    
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        return not np.any((prediction + observable) % 2)
    
@DecoderRegistry.register(name="relay-bp")
class RelayBPDecoderImpl(Decoder):
    """Relay-BP decoder implementation"""
    
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the Relay-BP decoder"""
        import relay_bp
        self.decoder = relay_bp.RelayDecoderF32(
            csr_matrix(state.check_matrix),
            error_priors=state.priors,
            gamma0=0.1,
            pre_iter=80,
            num_sets=60,
            set_max_iter=60,
            gamma_dist_interval=(-0.24, 0.66),
            stop_nconv=3
        ) # Parameters taken from the relay-bp example, better combinations probably exist
        self.obs_matrix = state.obs_matrix
        
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome"""
        return self.decoder.decode(syndrome)
    
    def decode_batch(self, syndromes: np.ndarray) -> list:
        """Decode a batch of syndromes"""
        return self.decoder.decode_batch(syndromes)
    
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        error_prediction = (self.obs_matrix.toarray() @ prediction) % 2
        return not np.any((error_prediction + observable) % 2)
        