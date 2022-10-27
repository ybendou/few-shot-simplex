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
from fstools.few_shot_utils import define_runs, generate_runs, get_closest_features_to_simplex
warnings.filterwarnings("ignore")

def ncmInterpol(features, run_classes, run_indices, n_shots, n_runs, args, AS_feats=None, lam_mix=1):
    """
    Compute NCM based on simplex summets
    Perform a linear combination between mean of summets and each  summets
    - AS_feats : Tensor of Average of feature crops.
    - lam_mix : coefficient interpolatation between average crops of each image and the summets from the simplex
    """
    with torch.no_grad():
        batch_few_shot_runs = 1
        dim = features[0].shape[-1]
        K = [feat.shape[0] for feat in features]
        dim = features[0].shape[-1]
        minK, maxK = min(K), max(K)
        D = torch.ones(len(K), maxK, dim).to(args.device) # convert simplex data to Tensor format
        if minK!=maxK:
            for k in range(len(D)):
                D[k, :K[k]] = features[k]
        D = D.reshape(20,600, -1, dim)
        K_t = torch.Tensor(K).reshape(20,600, 1) # number of summets per image
        scores = []
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)

        for batch_idx in tqdm(range(n_runs // batch_few_shot_runs)):
            runs = torch.stack([generate_runs(D[:, :,k, :].to(args.device), run_classes, run_indices, batch_idx, batch_few_shot_runs=batch_few_shot_runs) for k in range(maxK)], dim=3).squeeze(0) # Generate runs
            runs_K = generate_runs(K_t.to(args.device), run_classes, run_indices, batch_idx, batch_few_shot_runs=batch_few_shot_runs).squeeze(0).squeeze(-1) # Number of summets per image
            AS_runs = generate_runs(AS_feats.to(args.device), run_classes, run_indices, batch_idx, batch_few_shot_runs=batch_few_shot_runs).squeeze(0)
            
            support = runs[:,:n_shots] 
            support_K = runs_K[:,:n_shots]
            query = runs[:,n_shots:n_shots+args.n_queries]
            query_K = runs_K[:,n_shots:n_shots+args.n_queries]

            # Retrieve true summets of support
            support_cat_K = [[support[c, n, :support_K[c, n].int().item()] for n in range(n_shots)] for c in range(args.n_ways)]    
            
            # Mix support data 
            mixed_supports = [lam_mix*torch.cat(support_cat_K[c])+(1-lam_mix)*AS_runs[c, 0] for c in range(args.n_ways)]
            distances_all_queries = []
            for cq in range(args.n_ways):
                for q in range(args.n_queries):
                    onequery = lam_mix*query[cq][q][:query_K[cq,q].int()]+(1-lam_mix)*AS_runs[cq, n_shots+q] # mix queries with mean
                    distances_c = torch.stack([torch.norm(onequery.unsqueeze(1)-mixed_supports[c].unsqueeze(0), dim=2).min() for c in range(args.n_ways)]) # compute distances of all summets between the shot and the query and get their min value 
                    distances_all_queries.append(distances_c)
                    
            distances_all_queries = torch.stack(distances_all_queries).reshape(args.n_ways, args.n_queries, args.n_ways) # out of all the min distances get the argmin distance between the classes and the query
            winners = torch.min(distances_all_queries, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, ""), torch.Tensor(scores)


if __name__=='__main__':

    args = process_arguments()
    fix_seed(args.seed, deterministic=args.deterministic)

    novel_features, AS_feats, base_features = load_features(args.features_path, args.features_base_path)
    print('Features Loaded')
    mean_base_features = torch.mean(base_features.reshape(-1, base_features.shape[-1]), dim=0).to('cpu')

    # For each image, get closest crop to each simplex summet

    with open(args.centroids_file, 'rb') as pickle_file:
        features = pickle.load(pickle_file)
    print('Summets loaded')
    # Find closest crops to summets and summets with values of the crops
    features = [get_closest_features_to_simplex(features[f], novel_features.reshape(-1, novel_features.shape[-2], novel_features.shape[-1])[f]) for f in range(len(features))]

    num_elements = [600]*20
    runs = list(zip(*[define_runs(args.n_ways, s, args.n_queries, 20, num_elements, args.n_runs) for s in args.n_shots]))
    run_classes, run_indices = runs[0], runs[1]

    if args.preprocessing == 'ME':
        features = [x - mean_base_features.unsqueeze(0) for x in features]
        features = [x / torch.norm(x, dim = 1, keepdim = True) for x in features]
        AS_feats = AS_feats - mean_base_features.unsqueeze(0)
        AS_feats = AS_feats / torch.norm(AS_feats, dim = 2, keepdim = True)

    (acc, conf), scores = ncmInterpol(features, run_classes[0], run_indices[0], args.n_shots[0], args.n_runs, args, AS_feats=AS_feats, lam_mix=args.lamda_mix)
    print(f'{args.lamda_mix}: {np.round(100*acc,2)}% ±{np.round(conf*100, 2)}%')

    if args.log_perf:
        with open(args.log_perf, 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
