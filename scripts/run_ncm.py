#############################################################################################################
# NCM evaluation : mix summets from simplex with mean of crops
#############################################################################################################

import os
from tqdm import tqdm 
import torch
import numpy as np
import random
import warnings
import pickle
from fstools.args import process_arguments
from fstools.utils import fix_seed, load_features, stats
from fstools.few_shot_utils import define_runs, generate_runs
warnings.filterwarnings("ignore")

def ncm(features, run_classes, run_indices, n_shots, n_runs, args):
    batch_few_shot_runs = 100
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        scores = []
        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, ""), scores

if __name__=='__main__':
    args = process_arguments()
    fix_seed(args.seed, deterministic=args.deterministic)

    novel_features, AS_feats, base_features = load_features(args.features_path, args.features_base_path, device=args.device)
    print('Features Loaded')
    if 'mean' in args.features_base_path:
        mean_base_features = base_features.to(args.device)
    else:
        mean_base_features = torch.mean(base_features.reshape(-1, base_features.shape[-1]), dim=0).to(args.device)

    if args.preprocessing == 'ME':
        AS_feats = AS_feats - mean_base_features.unsqueeze(0)
        AS_feats = AS_feats / torch.norm(AS_feats, dim = 2, keepdim = True)

    num_elements = [600]*20
    runs = list(zip(*[define_runs(args.n_ways, s, args.n_queries, 20, num_elements, args.n_runs) for s in args.n_shots]))
    run_classes, run_indices = runs[0], runs[1]
    (acc, conf), scores = ncm(AS_feats, run_classes[0], run_indices[0], args.n_shots[0], args.n_runs, args)
    print(f'{args.lamda_mix}: {np.round(100*acc,2)}% ±{np.round(conf*100, 2)}%')
