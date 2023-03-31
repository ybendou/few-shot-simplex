import random
import argparse
import os
import torch
import numpy as np
import random

def process_arguments(params=None):
    """
    Processes the command line arguments.
    Args:
        params: (dict) Dictionary of parameters.
    Returns:
        params: (dict) Dictionary of parameters.
    """
    parser = argparse.ArgumentParser()
    ### hyperparameters
    parser.add_argument("--batch-size", type=int, default=50, help="batch size")
    parser.add_argument("--preprocessing", type=str, default="", help="preprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
    parser.add_argument("--postprocessing", type=str, default="", help="postprocessing sequence for few shot, can contain R:relu P:sqrt E:sphering and M:centering")
    parser.add_argument("--postprocess-after-simplex", action="store_true", help="postprocessing sequence for few shot, boolean if True run postprocessing after simplex extraction")

    parser.add_argument("--AS", type=int, default=100, help="number of crops to consider in augmented samples")
    parser.add_argument("--notQR", type=bool, default=False, help="Use QR Reduction to reduce dimentionality")
    parser.add_argument("--order-preprocessing", type=str, default='PM', help="Order in which to apply the preprocessing, MP: Mean crops then preprocess or PM: Preprocess features then Mean crops")
    parser.add_argument("--lamda-reg", type=float, default=0.05, help="Regularization for simplex estimation")
    parser.add_argument("--extraction", type=str, default='simplex', help="Extraction type")
    parser.add_argument("--alpha-iter", type=int, default=-1, help="Number of iteration for alphas in Gradient descent per epoch")
    parser.add_argument("--thresh-elbow", type=float, default=1.5, help="Threshold for MSE jumps in simplex extraction in order to automatically selecet the best number of  summits ")
    parser.add_argument("--lamda-mix", type=float, default=1, help="coefficient to mix simplex summets with mean")
    parser.add_argument("--maxK", type=int, default=63, help="Maximum number of summits to consider in simplex extraction")

    ### pytorch options
    parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
    parser.add_argument("--dataset-path", type=str, default='test/test/', help="dataset path")
    parser.add_argument("--features-path", type=str, default='/ssd2/data/AugmentedSamples/features/', help="features directory path")
    parser.add_argument("--features-base-path", type=str, default='/ssd2/data/AugmentedSamples/features/', help="base features directory path")
    parser.add_argument("--centroids-file", type=str, default='', help="path file to save or load the results of the centroids extraction")
    parser.add_argument("--log-perf", type=str, default='', help="file where to report performance")
    
    parser.add_argument("--dataset-device", type=str, default="", help="use a different device for storing the datasets (use 'cpu' if you are lacking VRAM)")
    parser.add_argument("--deterministic", action="store_true", help="use desterministic randomness for reproducibility")

    ### run options
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset to use")
    parser.add_argument("--seed", type=int, default=-1, help="set random seed manually, and also use deterministic approach")

    ### few-shot parameters
    parser.add_argument("--n-shots", type=str, default="[1,5]", help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
    parser.add_argument("--n-runs", type=int, default=10000, help="number of few-shot runs")
    parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
    parser.add_argument("--n-queries", type=int, default=15, help="number of few-shot queries")
    
    try:
        get_ipython()
        print('Notebook')
        args = parser.parse_args("")
    except:
        args = parser.parse_args()
        
    if params!=None:
        for key, value in params.items():
            args.__dict__[key]= value
    assert args.extraction in ['kmeans', 'simplex'], 'wrong choice of extraction, choose either "kmeans" or "simplex"'
    ### process arguments
    if args.dataset_device == "":
        args.dataset_device = args.device
    if args.dataset_path[-1] != '/':
        args.dataset_path += "/"

    if args.device[:5] == "cuda:" and len(args.device) > 5:
        args.devices = []
        for i in range(len(args.device) - 5):
            args.devices.append(int(args.device[i+5]))
        args.device = args.device[:6]
    else:
        args.devices = [args.device]

    if args.seed == -1:
        args.seed = random.randint(0, 1000000000)

    try:
        n_shots = int(args.n_shots)
        args.n_shots = [n_shots]
    except:
        args.n_shots = eval(args.n_shots)

    if '[' in args.features_path : 
        args.features_path = eval(args.features_path)
    print("args, ", end='')
    return args