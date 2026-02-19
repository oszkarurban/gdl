import json
import os
import sys
import numpy as np
import torch
import argparse
from main import Exp
from API.dataloader import make_cath_loader
from API.cath_dataset import CATH

def get_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='results', type=str)
    parser.add_argument('--ex_name', default='ProDesign', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)

    # CATH
    # dataset parameters
    parser.add_argument('--data_name', default='CATH', choices=['CATH', 'TS50'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    # method parameters
    parser.add_argument('--method', default='ProDesign', choices=['ProDesign'])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--hidden_dim',  default=128, type=int)
    parser.add_argument('--node_features',  default=128, type=int)
    parser.add_argument('--edge_features',  default=128, type=int)
    parser.add_argument('--k_neighbors',  default=30, type=int)
    parser.add_argument('--dropout',  default=0.1, type=int)
    parser.add_argument('--num_encoder_layers', default=10, type=int)

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=100, type=int)

    # ProDesign parameters
    parser.add_argument('--updating_edges', default=4, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)
    parser.add_argument('--virtual_num', default=3, type=int)
    
    args = parser.parse_args([])
    return args

def load_chain_set(path):
    print(f"Loading chain set from {path}")
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    alphabet_set = set([a for a in alphabet])
    max_length = 500
    
    data_list = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            entry = json.loads(line)
            seq = entry['seq']

            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)

            bad_chars = set([s for s in seq]).difference(alphabet_set)

            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    data_list.append({
                        'title':entry['name'],
                        'seq':entry['seq'],
                        'CA':entry['coords']['CA'],
                        'C':entry['coords']['C'],
                        'O':entry['coords']['O'],
                        'N':entry['coords']['N']
                    })
    print(f"Loaded {len(data_list)} proteins")
    return data_list

def evaluate_split(exp, full_data, split_file, split_name):
    print(f"\nEvaluating split: {split_name} ({split_file})")
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    if isinstance(split_data, dict) and 'test' in split_data:
        test_names = set(split_data['test'])
    else:
        print("Warning: unexpected json structure, assuming list or dict with keys")
        test_names = set() # Should inspect if failure

    test_list = []
    for data in full_data:
        if data['title'] in test_names:
            test_list.append(data)
            
    print(f"Found {len(test_list)} proteins for this split")
    
    if len(test_list) == 0:
        print("No data found for this split!")
        return

    exp.test_loader = make_cath_loader(CATH(data=test_list), 'SimDesign', 8)
    
    # exp.test() returns perplexity, recovery
    # It also prints log messages
    perplexity, recovery = exp.test()
    print(f"Result for {split_name}: Perplexity={perplexity:.4f}, Recovery={recovery:.4f}")

def main():
    args = get_parser()
    
    # Initialize Experiment
    exp = Exp(args)
    
    # Load Checkpoint
    # Check current directory results
    checkpoint_path = os.path.join(args.res_dir, args.ex_name, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=exp.device)
    exp.method.model.load_state_dict(state_dict)
    
    # Load Data
    chain_set_path = os.path.join(args.data_root, 'cath', 'chain_set.jsonl')
    full_data = load_chain_set(chain_set_path)
    
    # Splits to evaluate
    splits = [
        ('All', os.path.join(args.data_root, 'cath', 'chain_set_splits.json')),
        ('Short', os.path.join(args.data_root, 'cath', 'test_split_L100.json')),
        ('Single Chain', os.path.join(args.data_root, 'cath', 'test_split_sc.json'))
    ]
    
    for name, path in splits:
        if os.path.exists(path):
            evaluate_split(exp, full_data, path, name)
        else:
            print(f"Split file not found: {path}")

if __name__ == "__main__":
    main()
