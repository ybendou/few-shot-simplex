#############################################################################################################
# Run simplex extraction on novel dataset
#############################################################################################################

import os
from tqdm import tqdm 
import torch
import numpy as np
import random
import warnings
import pickle
from fstools.args import process_arguments
from fstools.utils import fix_seed, load_features
from fstools.solver import find_summetsBatch
warnings.filterwarnings("ignore")

args = process_arguments()
fix_seed(args.seed, deterministic=args.deterministic)

print(f'{args.n_ways}-ways | {args.n_shots[0]}-shots | total runs: {args.n_runs} | AS: {args.AS} | QR:{not args.notQR} | \u03BB:{args.lamda_reg} | seed: {args.seed} | extraction: {args.extraction} | alpha-iter: {args.alpha_iter} | threshold-elbow: {args.thresh_elbow} | Preprocessing : {args.preprocessing} | Postprocessing : {args.postprocessing}')

novel_features = torch.load(args.features_path)[0]['features']
novel_features = novel_features.permute(1,0,2)
list_D = []
for b in tqdm(range(0, novel_features.shape[0]//args.batch_size)):
    data = novel_features[b*args.batch_size:(b+1)*args.batch_size]
    D = find_summetsBatch(data, args, method=args.extraction, thresh_elbow=args.thresh_elbow, return_jumpsMSE=True, lamda_reg=args.lamda_reg, n_iter=150, alpha_iter=5, 
                                                    trainCfg={'lr':0.1, 'mmt':0.8, 'D_iter':1, 'loss_amp':10000,'loss_alpha':1}, verbose=False, maxK=4, concat=False)
    list_D = list_D + D

if args.centroids_file:
    with open(args.centroids_file, 'wb') as handle:
        pickle.dump(list_D, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done :)')