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
import json
from torchvision import datasets
from fstools.args import process_arguments
from fstools.utils import fix_seed, load_features, stats
from fstools.few_shot_utils import define_runs, generate_runs
warnings.filterwarnings("ignore")
### NCM
def ncm(shots, queries):
    centroids = torch.stack([shotClass.mean(dim = 0) for shotClass in shots])
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - centroids.unsqueeze(0), dim = 2)
        score += (distances.argmin(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total
    ### kNN
def knn(shots, queries):
    k=1
    anchors = torch.cat(shots)
    labels = []
    for i in range(len(shots)):
        labels += [i] * shots[i].shape[0]
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - anchors.unsqueeze(0), dim = 2)
        sorting = distances.argsort(dim = 1)
        scores = torch.zeros(queriesClass.shape[0], len(shots))
        for j in range(queriesClass.shape[0]):
            for l in range(k):
                scores[j,labels[sorting[j,l]]] += 1
        score += (scores.argmax(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total
def knn_simplex_shots_only(shots, queries):
    k=1
    shots = [torch.cat(s) for s in shots]
    anchors = torch.cat(shots)
    labels = []
    for i in range(len(shots)):
        labels += [i] * shots[i].shape[0]
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - anchors.unsqueeze(0), dim = 2)
        sorting = distances.argsort(dim = 1)
        scores = torch.zeros(queriesClass.shape[0], len(shots))
        for j in range(queriesClass.shape[0]):
            for l in range(k):
                scores[j,labels[sorting[j,l]]] += 1
        score += (scores.argmax(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total
def knn_simplex(shots, queries):
    k=1
    shots = [torch.cat(s) for s in shots]
    anchors = torch.cat(shots)
    labels = []
    for i in range(len(shots)):
        labels += [i] * shots[i].shape[0]
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = []
        for query in queriesClass:
            distances.append(torch.norm(query.unsqueeze(1) - anchors.unsqueeze(0), dim = 2).min(dim=0)[0]) # min distance to each anchor
        distances = torch.stack(distances)
        sorting = distances.argsort(dim = 1)
        scores = torch.zeros(len(queriesClass), len(shots))
        for j in range(len(queriesClass)):
            for l in range(k):
                scores[j,labels[sorting[j,l]]] += 1
        score += (scores.argmax(dim = 1) - i == 0).float().sum()
        total += len(queriesClass)
    return score / total

if __name__=='__main__':

    args = process_arguments()
    if type(args.n_shots) == list:
        args.n_shots =  args.n_shots[0]
    fix_seed(args.seed, deterministic=args.deterministic)
    MYSPACE = os.environ['MYSPACE']
    #args.features_path = os.path.join(MYSPACE, 'experiments/simplex/coco/features/cocoval2017_dino_base8_224px_AS10coco_test_features.pt')
    novel_features = torch.load(args.features_path)[0]['features']
    novel_features = novel_features.permute(1,0,2)
    AS_feats = novel_features.mean(dim=1)
    print('Features Loaded')

    # For each image, get closest crop to each simplex summet

    #args.centroids_file = os.path.join(MYSPACE, 'experiments/simplex/coco/centroids/cocoval2017_dino_base8_224px_AS200coco_test_features_simplex005_dict_test.pickle')
    with open(args.centroids_file, 'rb') as pickle_file:
        features = pickle.load(pickle_file)
    print('Summets loaded')

    # create few shot problems from coco dataset:
    path2data=os.path.join(MYSPACE, 'datasets', 'coco', 'val2017')
    path2json=os.path.join(MYSPACE, 'datasets', 'coco', 'annotations', 'instances_val2017.json')
    coco_dataset = datasets.CocoDetection(root = path2data, annFile = path2json)
    # each element of coco_dataset is a tuple (image, targets)
    # targets is a dict with keys: 'boxes', 'labels', 'area', 'iscrowd'
    # get dict of targets and all images with the same target
    targets = {k:[] for k in coco_dataset.coco.cats.keys()}
    for i, (_, target) in enumerate(coco_dataset):
        if i in features.keys():
            for j in range(len(target)):
                targets[target[j]['category_id']].append(i)
    # get dict of targets and all images with the same target
    targets = {k:torch.tensor(list(set(v))) for k,v in targets.items() if len(v)>0}
    targets_list = torch.tensor(list(targets.keys()))
    images_list = list(targets.values())

    scores = {'lamda':args.lamda_mix, 'AS ncm':[], 'AS knn':[], 'simplex':[], 'simplex_shots_only':[]}
    for r in range(args.n_runs):
        # define n_ways classes:
        classes = targets_list[torch.randperm(len(targets))[:args.n_ways]]
        # define n_shots images per class:
        images_idx = [targets[c.item()][torch.randperm(len(targets[c.item()]))[:args.n_shots+args.n_queries]] for c in classes]
        shots = [[features[i.item()].squeeze(0) for i in images_idx[c][:args.n_shots]] for c in range(len(classes))]
        queries = [[features[i.item()].squeeze(0) for i in images_idx[c][args.n_shots:]] for c in range(len(classes))]

        shots_mix = [[(features[i.item()]*args.lamda_mix+(1-args.lamda_mix)*AS_feats[i]).squeeze(0) for i in images_idx[c][:args.n_shots]] for c in range(len(classes))]
        queries_mix = [[(features[i.item()]*args.lamda_mix+(1-args.lamda_mix)*AS_feats[i]).squeeze(0) for i in images_idx[c][args.n_shots:]] for c in range(len(classes))]

        shots_AS = [torch.stack([AS_feats[i.item()] for i in images_idx[c][:args.n_shots]]) for c in range(len(classes))]
        queries_AS = [torch.stack([AS_feats[i.item()] for i in images_idx[c][args.n_shots:]]) for c in range(len(classes))]
        scores['AS knn'].append(knn(shots_AS, queries_AS))
        scores['AS ncm'].append(ncm(shots_AS, queries_AS))
        scores['simplex_shots_only'].append(knn_simplex_shots_only(shots_mix, queries_AS))
        scores['simplex'].append(knn_simplex(shots_mix, queries_mix))

    for k,v in scores.items():
        if k != 'lamda':
            acc, conf = stats(v, "")
            print(f'lamda mix={args.lamda_mix} | {k}: {np.round(100*acc,2)}% ±{np.round(conf*100, 2)}%')

    if args.log_perf:
        with open(args.log_perf, 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

