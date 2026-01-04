import argparse
import numpy as np
import torch
import pickle
import os
from tqdm import trange
import random
from benchmark_sample import sample_dgl_graph
from pathlib import PosixPath
import json
from datetime import datetime



def generate_pairs_structure(args=None):

    sbm_1 = {'sizes': args.sizes_1, 'p': args.p, 'q': args.q, 
             "features":args.features}
    sbm_2 = {'sizes': args.sizes_2, 'p': args.p, 'q': args.q,
             "features":args.features}

    data = []
    label_list = []

    # negative label data points
    for _ in trange(args.n_samples, desc='Generating negative examples'):
        pair = [sample_dgl_graph(sbm_1), sample_dgl_graph(sbm_2)]
        random.shuffle(pair)
        graph1, graph2 = pair

        label = torch.tensor(0)
        data.append((graph1, graph2, label))
        label_list.append(label)

    # positive label data points
    for _ in trange(args.n_samples, desc='Generating positive examples'):
        if random.random() < 0.5:
            sbm = sbm_1
        else:
            sbm = sbm_2
        graph1, graph2 = sample_dgl_graph(sbm), sample_dgl_graph(sbm)
        label = torch.tensor(1)
        data.append((graph1, graph2, label))
        label_list.append(label)

    random.shuffle(data)
    label_list = [k for (i,j,k) in data]

    # save data
    n = sum(args.sizes_1)
    k1, k2 = len(args.sizes_1), len(args.sizes_2)
    time_exp = str(datetime.utcnow().strftime("%m_%d_%H:%M:%S"))
    save_dir = args.save_dir + f"{2*args.n_samples}_pairs_structure_sbm_{n}_{k1}_{k2}_{args.p}_{args.q}_{args.features}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(data, open(save_dir / 'data.p', 'wb'))
    pickle.dump(label_list, open(save_dir / 'labels.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)


def generate_sequence_structure(args=None):

    """Generate a time series of graphs where the underlying SBM parameters change.

    sbm_1/sbm_2: parameter dictionaries for the SBM before and after the change point
    length:
    sizes: list containing the number of nodes in each community
    length: number of graphs in each block
    """

    cp_time = np.random.randint(args.n_samples // 4, 3 * args.n_samples // 4)
    print(f" Change point at t = {cp_time}")

    sbm_1 = {'sizes': args.sizes_1, 'p': args.p, 'q': args.q, "features":args.features}
    sbm_2 = {'sizes': args.sizes_2,  'p': args.p, 'q': args.q, "features":args.features}

    g1, g2 = [], []
    for _ in range(cp_time):
        g1.append(sample_dgl_graph(sbm_1))
    for _ in range(args.n_samples- cp_time):
        g2.append(sample_dgl_graph(sbm_2))
    seq = g1 + g2

    # save data
    n = sum(args.sizes_1)
    k1, k2 = len(args.sizes_1), len(args.sizes_2)
    time_exp = str(datetime.utcnow().strftime("%m_%d_%H:%M:%S"))
    save_dir = args.save_dir + f"{time_exp}_merge_T_{args.n_samples}_n_{n}_k1_{k1}_k2_{k2}_p_{args.p}_q_{args.q}_{args.rep}"
    save_dir = PosixPath(save_dir).expanduser()
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(seq, open(save_dir / 'data.p', 'wb'))

    # save args
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    # save change point time
    with open(save_dir / 'time.json', 'w') as fp:
        json.dump(cp_time, fp)

    return seq, cp_time


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--sizes_1', type=int, nargs="+", default=[100, 100, 100, 100])
    parser.add_argument('--sizes_2', type=int, nargs="+", default=[200, 200])
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--q', type=float, default=0.2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--features', type=str, choices=['gaussian'], default=None)
    #parser.add_argument('--sequence', action='store_true', default=True)
    parser.add_argument('--sequence', dest='sequence', action='store_true', help='Generate a sequence of graphs')
    parser.add_argument('--no-sequence', dest='sequence', action='store_false', help='Generate pairs of graphs')
    parser.set_defaults(sequence=True)
    parser.add_argument('--save_dir', type=str,  default='/home/b/skoove/compute/entropy/flow_stability/')
    parser.add_argument('--rep', type=int, default=0)

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if not args.sequence:
        generate_pairs_structure(args=args)
    else:
        generate_sequence_structure(args=args)

if __name__ == '__main__':
    main()