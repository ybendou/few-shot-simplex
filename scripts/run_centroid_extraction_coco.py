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
from torchvision import datasets
from fstools.args import process_arguments
from fstools.utils import fix_seed, load_features
from fstools.solver import find_summetsBatch
warnings.filterwarnings("ignore")

args = process_arguments()
fix_seed(args.seed, deterministic=args.deterministic)

print(f'{args.n_ways}-ways | {args.n_shots[0]}-shots | total runs: {args.n_runs} | AS: {args.AS} | QR:{not args.notQR} | \u03BB:{args.lamda_reg} | seed: {args.seed} | extraction: {args.extraction} | alpha-iter: {args.alpha_iter} | threshold-elbow: {args.thresh_elbow} | Preprocessing : {args.preprocessing} | Postprocessing : {args.postprocessing}')

novel_features = torch.load(args.features_path)[0]['features']
novel_features = novel_features.permute(1,0,2)
maxK = min(novel_features.shape[1], args.maxK)

# import coco dataset
MYSPACE = os.getenv('MYSPACE') # it just makes life easier
path2data=os.path.join(MYSPACE, 'datasets', 'coco', 'val2017')
path2json=os.path.join(MYSPACE, 'datasets', 'coco', 'annotations', 'instances_val2017.json')
coco_dataset = datasets.CocoDetection(root = path2data, annFile = path2json)

list_D = {}
for b in tqdm(range(0, novel_features.shape[0]//args.batch_size)):
    data = novel_features[b*args.batch_size:(b+1)*args.batch_size]
    try:
        D = find_summetsBatch(data, args, method=args.extraction, thresh_elbow=args.thresh_elbow, return_jumpsMSE=True, lamda_reg=args.lamda_reg, n_iter=100000, alpha_iter=5, 
            trainCfg={'lr':1, 'mmt':0.8, 'D_iter':1, 'loss_amp':1,'loss_alpha':1, 'early_stopping':True, 'early_stopping_eps':5e-6}, verbose=False, maxK=args.maxK, concat=False, normalize=False, K=len(coco_dataset[b][1]))
        list_D[b]=D[0]
    except Exception as e:
        print(e)
        continue

if args.centroids_file:
    with open(args.centroids_file, 'wb') as handle:
        pickle.dump(list_D, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done :)')
