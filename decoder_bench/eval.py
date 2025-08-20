import os
import csv
import json
import argparse
try:
    from .sampler import Sampler, DecoderRegistry
except ImportError:
    from sampler import Sampler, DecoderRegistry
import decoders  # noqa: F401

def main():
    parser = argparse.ArgumentParser(description="Run parallel decoding on a dataset")
    parser.add_argument('--filename', type=str, required=True, help='Name of the dataset file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--decoder', type=str, required=True, help='Decoder to use')
    parser.add_argument('--num_shots', type=int, default=500_000, help='Number of shots to process')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size per worker')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2, help='Number of worker processes')
    parser.add_argument('--decoder_params', type=str, default=None, 
                      help='JSON string of decoder parameters, e.g. \'{"max_iter": 50}\'')
    parser.add_argument('--decoder_dir', type=str, default=None,
                      help='Directory containing custom decoder implementations')
    parser.add_argument('--output_file', type=str, default='./results/decoder_results.csv',
                      help='Path to output file for results')
    
    args = parser.parse_args()
    
    if args.decoder_dir:
        num_loaded = DecoderRegistry.load_from_directory(args.decoder_dir)
        print(f"Loaded {num_loaded} custom decoders from {args.decoder_dir}")
    
    decoder_kwargs = None
    if args.decoder_params:
        decoder_kwargs = json.loads(args.decoder_params)
    
    sampler = Sampler(num_workers=args.num_workers, batch_size=args.batch_size)
    results = sampler.collect(
        args.filename,
        args.dataset_dir,
        args.decoder,
        num_shots=args.num_shots,
        decoder_kwargs=decoder_kwargs
    )
    
    print("\nResults:")
    print(f"File: {results['filename']}")
    print(f"Decoder: {results['decoder']}")
    print(f"Decoder parameters: {results['decoder_params']}")
    print(f"Logical error rate: {results['logical_error_rate']:.6f}")
    print(f"Time per shot: {results['time_per_shot']*1000:.6f} ms")
    print(f"Total time: {results['total_time']:.2f} seconds")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    file_exists = os.path.isfile(args.output_file)
    with open(args.output_file, 'a') as file:
        fieldnames = [
            'filename', 'decoder', 'logical_error_rate',
            'shots', 'time_per_shot', 'total_time'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'filename': results['filename'],
            'decoder': results['decoder'],
            'logical_error_rate': results['logical_error_rate'],
            'shots': results['shots'],
            'time_per_shot': results['time_per_shot'],
            'total_time': results['total_time']
        })

if __name__ == "__main__":
    main()