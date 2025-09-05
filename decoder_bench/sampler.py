import os
from time import process_time
import h5py
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Type, List
import stim
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool
from abc import ABC, abstractmethod

try:
    from .common.build_circuit import dem_to_check_matrices
except ImportError:
    from common.build_circuit import dem_to_check_matrices

@dataclass
class ChunkTask:
    """A chunk of work to be processed by a worker"""
    task_id: int
    syndromes: np.ndarray
    observables: np.ndarray 

@dataclass
class DecoderState:
    """Shared state for decoders that can be initialized once and reused"""
    check_matrix: Any
    obs_matrix: Any
    priors: np.ndarray
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel

@dataclass
class ChunkResult:
    """Result from processing a chunk"""
    task_id: int
    error_count: int
    num_shots: int
    time_taken: float

class Decoder(ABC):
    """Abstract base class for all decoders"""
    
    @abstractmethod
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the decoder with the given state and parameters"""
        pass
    
    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome and return a prediction"""
        pass
    
    @abstractmethod
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        pass
    
    def decode_batch(self, syndromes: np.ndarray) -> List[np.ndarray]:
        """Decode a batch of syndromes (can be overridden for efficiency)"""
        return [self.decode(syndrome) for syndrome in syndromes]
    
    def verify_batch(self, predictions: List[np.ndarray], observables: np.ndarray) -> List[bool]:
        """Verify a batch of predictions (can be overridden for efficiency)"""
        return [self.verify(pred, obs) for pred, obs in zip(predictions, observables)]

class DecoderRegistry:
    """Registry for available decoders"""
    _decoders: Dict[str, Type[Decoder]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """Register a decoder class (can be used as a decorator)"""
        def decorator(decoder_class):
            decoder_name = name or decoder_class.__name__.lower()
            cls._decoders[decoder_name] = decoder_class
            return decoder_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Decoder]:
        """Get a decoder class by name"""
        if name not in cls._decoders:
            raise ValueError(f"Decoder '{name}' not registered")
        return cls._decoders[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available decoders"""
        return list(cls._decoders.keys())
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load decoder implementations from a Python file"""
        import importlib.util
        import sys
        
        module_name = os.path.basename(filepath).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return len([d for d in cls._decoders.keys() if hasattr(module, d)])
    
    @classmethod
    def load_from_directory(cls, directory: str):
        """Load all decoder implementations from a directory"""
        loaded = 0
        for filename in os.listdir(directory):
            if filename.endswith('.py'):
                filepath = os.path.join(directory, filename)
                loaded += cls.load_from_file(filepath)
        return loaded

class Sampler:
    """Parallel sampler for decoder evaluation"""
    
    def __init__(self, num_workers=None, batch_size=1000):
        """Initialize the sampler"""
        self.num_workers = num_workers or mp.cpu_count()
        self.batch_size = batch_size
        
    def _initialize_decoder_state(self, filename: str, dataset_dir: str) -> DecoderState:
        """Initialize decoder state from dataset file"""
        with h5py.File(os.path.join(dataset_dir, filename), 'r') as f:
            circuit_data = f['circuit'][()]
            
            if isinstance(circuit_data, bytes):
                circuit = stim.Circuit(circuit_data.decode('utf-8'))
            else:
                circuit = circuit_data
            
            dem = circuit.detector_error_model()
            check_matrix, obs_matrix, priors = dem_to_check_matrices(dem)
            
        return DecoderState(
            check_matrix=check_matrix,
            obs_matrix=obs_matrix,
            priors=priors,
            circuit=circuit,
            dem=dem
        )
    
    def _process_chunk(self, chunk: ChunkTask, decoder: Decoder) -> ChunkResult:
        """Process a chunk using the provided decoder"""
        start_time = process_time()
        
        predictions = decoder.decode_batch(chunk.syndromes)
        
        end_time = process_time()
        
        # Verify predictions
        error_count = 0
        for i, pred in enumerate(predictions):
            if not decoder.verify(pred, chunk.observables[i]):
                error_count += 1
        
        return ChunkResult(
            task_id = chunk.task_id,
            error_count = error_count,
            num_shots = len(chunk.syndromes),
            time_taken = end_time - start_time
        )
    
    def _worker_function(self, task_queue, result_queue, decoder_name, state_dict, decoder_kwargs=None):
        """Worker process function that processes chunks from the queue"""
        state = DecoderState(
            check_matrix=state_dict['check_matrix'],
            obs_matrix=state_dict['obs_matrix'],
            priors=state_dict['priors'],
            circuit=state_dict['circuit'],
            dem=state_dict['dem']
        )
        
        decoder_class = DecoderRegistry.get(decoder_name)
        decoder = decoder_class(state, **(decoder_kwargs or {}))
        
        while True:
            chunk = task_queue.get()
            if chunk is None:
                task_queue.put(None) 
                break
            
            result = self._process_chunk(chunk, decoder)
            result_queue.put(result)
    
    def _create_task_batches(self, filename, dataset_dir, total_shots, batch_size):
        """Generator that yields batches of tasks"""
        with h5py.File(os.path.join(dataset_dir, filename), 'r') as f:
            total_available = min(len(f['syndromes']), total_shots)
            task_id = 0
            
            for start_idx in range(0, total_available, batch_size):
                end_idx = min(start_idx + batch_size, total_available)
                syndromes = f['syndromes'][start_idx:end_idx]
                observables = f['observables'][start_idx:end_idx]
                
                yield ChunkTask(
                    task_id=task_id,
                    syndromes=syndromes,
                    observables=observables
                )
                
                task_id += 1
    
    def collect(self, filename, dataset_dir, decoder_name, num_shots=500_000, 
                decoder_kwargs=None, progress_callback=None):
        """Collect decoder results from a dataset"""
        if decoder_name not in DecoderRegistry.list_available():
            raise ValueError(f"Decoder '{decoder_name}' not registered. Available decoders: {DecoderRegistry.list_available()}")
            
        print(f"Using {self.num_workers} workers with batch size {self.batch_size}")
        print(f"Decoder: {decoder_name} with parameters: {decoder_kwargs or {}}")
        
        state = self._initialize_decoder_state(filename, dataset_dir)
        
        task_queue = mp.Queue(maxsize=self.num_workers * 2)  # Buffer 2x number of workers
        result_queue = mp.Queue()
        
        state_dict = {
            'check_matrix': state.check_matrix,
            'obs_matrix': state.obs_matrix,
            'priors': state.priors,
            'circuit': state.circuit,
            'dem': state.dem
        }
        
        workers = []
        for _ in range(self.num_workers):
            p = mp.Process(
                target=self._worker_function,
                args=(task_queue, result_queue, decoder_name, state_dict, decoder_kwargs)
            )
            p.daemon = True
            p.start()
            workers.append(p)
        
        with h5py.File(os.path.join(dataset_dir, filename), 'r') as f:
            total_available = min(len(f['syndromes']), num_shots)
        num_tasks = np.ceil(total_available / self.batch_size).astype(int)
        
        def queue_filler():
            for task in self._create_task_batches(filename, dataset_dir, num_shots, self.batch_size):
                task_queue.put(task)
            
            task_queue.put(None)
        
        filler_thread = ThreadPool(1)
        filler_thread.apply_async(queue_filler)
        
        results = []
        with tqdm(total=num_tasks) as pbar:
            for _ in range(num_tasks):
                result = result_queue.get()
                results.append(result)
                pbar.update(1)
                if progress_callback:
                    progress_callback(len(results), num_tasks)
        
        for p in workers:
            p.join()
        
        filler_thread.close()
        filler_thread.join()
        
        total_errors = sum(r.error_count for r in results)
        total_shots = sum(r.num_shots for r in results)
        total_time = sum(r.time_taken for r in results)
        
        return {
            'filename': filename,
            'decoder': decoder_name,
            'decoder_params': decoder_kwargs or {},
            'logical_error_rate': total_errors / total_shots if total_shots > 0 else 0,
            'shots': total_shots,
            'time_per_shot': total_time / max(1, total_shots),
            'total_time': total_time
        }