import json
import sys, os
import re, pdb
import argparse
import numpy as np
import random

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def data_partition(args: dict):

    input_file = os.path.join(args.raw_data_dir, "train.jsonl")
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]    

    from collections import defaultdict

    label_groups = defaultdict(list)
    for entry in data:
        label_groups[entry['label']].append(entry)

    label_data = list(label_groups.values())

    alpha = args.alpha
    num_clients = args.num_clients
    proportions = np.random.dirichlet([alpha] * num_clients, len(label_data))

    client_data = [[] for _ in range(num_clients)]
    min_files = 1

    for label_idx, label_group in enumerate(label_data):
        splits = (proportions[label_idx] * len(label_group)).astype(int)
    
        while np.any(splits < min_files):
            deficit_clients = np.where(splits < min_files)[0]
            excess_clients = np.where(splits > min_files)[0]
            for client_idx in deficit_clients:
                if len(excess_clients) == 0:
                    break
                donor_idx = excess_clients[0]
                splits[client_idx] += 1
                splits[donor_idx] -= 1
                if splits[donor_idx] <= min_files:
                    excess_clients = excess_clients[1:]

        splits[-1] = len(label_group) - splits[:-1].sum()
        split_indices = np.cumsum(splits)[:-1]
        split_data = np.split(label_group, split_indices)

        for client_idx, split in enumerate(split_data):
            client_data[client_idx].extend(split)

    for client_idx, data in enumerate(client_data):
        random.shuffle(data)
        output_file = f"client_{client_idx}.json"
        with open(os.path.join(args.output_partition_path, output_file), 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default='../../../data/hateful_memes/raw_data',
        help="Raw data path of hateful memes data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default='../../../data/hateful_memes/raw_data/partition-alpha0.5-clt10',
        help="Output path of hateful memes data set",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=10, 
        help='Number of clients to cut from whole data.'
    )
    parser.add_argument(
        "--dataset",
        type=str, 
        default="hateful_memes",
        help='Dataset name.'
    )
    args = parser.parse_args()
    data_partition(args)
    
    
    
