import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import numpy as np
import random
from tqdm import tqdm 

def get_loss_mse(alpha, D, X, loss_amp=1, per_elem_batch=False):
    """
        Compute MSE loss on batches for weights
        Arguments: 
            - alpha: weights matrix.
            - D : summets matrix.
            - X : data matrix.
            - loss_amp : loss amplifier for vanishing gradients of the softmax.
        Returns:
            - loss.
    """
    reduction = {True: 'none', False:'mean'}[per_elem_batch] 
    L2 = nn.MSELoss(reduction=reduction)
    batch = X.shape[0]
    
    out = torch.einsum("bnk,bkd->bnd", nn.Softmax(dim=2)(alpha), D)
    if per_elem_batch:
        loss = L2(out, X).reshape(batch, -1).mean(axis=1)
    else:
        loss = L2(out, X) 
    return loss_amp*loss*batch

def get_close_form_D(alpha, X,lamda_reg=1, device='cuda:0'):
    """
    Returns the close form solution of the summits of the simplex with a regularization of the distance between the summits.
    The method is performed on batches of simplexes
    From https://ieeexplore.ieee.org/document/1344161
    Arguments:
        - alpha : weights matrix.
        -  X : data matrix.
        - lamda_reg : regularization between 0 and 1. 
    Returns : 
        - Summets of simplex.
    """
    K = alpha.shape[-1]
    dim = X.shape[-1]
    batch = X.shape[0]
    N = X.shape[1]

    K_div = K-1 if K>1 else 1
    lamda = N*lamda_reg/((K_div)*(1-lamda_reg))
    
    I = torch.eye(K).repeat(batch, 1).reshape(batch, K, K).to(device)
    Ones = torch.ones(batch, K, K).to(device)
    A = nn.Softmax(dim=2)(alpha)
    At = torch.transpose(A, 1, 2)
    AtA = torch.einsum('bkn,bnj->bkj', At, A)
    inv = torch.stack([torch.linalg.pinv((AtA +  lamda*(I-Ones/K))[b]) for b in range(batch)])
    invDotAt = torch.einsum('bjk,bkn->bjn', inv, At)
    return torch.einsum("bkn,bnd->bkd",invDotAt, X)
    

def get_loss_reg(D, loss_amp=1, device='cuda:0'):
    """
        Compute regularization loss : Perimeter of the simplex
    """
    K = D.shape[1]
    L2 = nn.MSELoss()
    perm = torch.arange(K)-1
    loss = L2(D - D[:, perm], torch.zeros(D.shape).to(device))/(2*K**2)
    return loss*loss_amp

def gradient_descent_ClosedForm(X, K, lamda=0.1, trainCfg={'lr':0.1, 'mmt':0.8, 'n_iter':10, 'loss_amp':1, 'loss_alpha':1, 'early_stopping':False}, D_init=None, verbose=True, device='cuda:0'):
    """
        Perform Gradient Descent to find simplex summets.
        The summets are computer using a close form.
        The weights matrix is computed using GD.
        The optimization is done on batches.
        Arguments:
            - X : data matrix.
            - K : number of summets.
            - lamda : simplex regularization.
            - trainCfg: dict containing training configuration, default is {'lr':0.1, 'mmt':0.8, 'n_iter':10, 'loss_amp':1, 'loss_alpha':1}.
            - D_init : if None start with random init, if str start with random points from the dataset if Tensor start with the given intialization.
            - verbose : if True print loss training.
        Returns : 
            - D_sol : Simplex solution.
            - loss : MSE loss.
    """
    # load images as tensors
    X = X.to(device) # create batch of one image
    X.requires_grad=False 
    
    batch = X.shape[0]
    dim = X.shape[-1]
    
    loss_amp, loss_alpha = trainCfg['loss_amp'], trainCfg['loss_alpha']

    init_alpha = torch.randn(batch, X.shape[1], K).to(device) # init alphas with randn
    alpha = nn.Parameter(init_alpha.clone())
    
    if type(D_init) == type(None):
        D = nn.Parameter(torch.randn(batch, K, dim).clone().to(device))
    else:
        if type(D_init) == str:
            permutations = [torch.randperm(X.shape[1])[:K] for _ in range(batch)]
            D = torch.stack([X[b][p] for b,p in enumerate(permutations)])
            D = nn.Parameter(D.to(device))
        else:
            D = nn.Parameter(D.to(device))
    D.requires_grad = False
    optimizerAlpha = torch.optim.SGD([alpha], lr=trainCfg['lr'], momentum=trainCfg['mmt'])
    ##### Stopped here
    best_epoch = {'loss':torch.tensor([10e5, 10e5]), 'n':0, 'loss_reg':10e5, 'D':D.data.cpu().clone(), 'alpha':alpha.data.cpu().clone()}
    count_val = 1
    previous_loss = 10e5
    for n in range(trainCfg['n_iter']):
        
        # For the first iteration, first look for alphas
        if n == 0: 
            # Fix D and re-update alpha:
            alpha.requires_grad = True
            for _ in range(trainCfg['alpha_iter']):
                optimizerAlpha.zero_grad()
                lossMSEAlpha = get_loss_mse(alpha, D, X, loss_amp=loss_amp)
                lossMSEAlpha.backward()
                optimizerAlpha.step()
        
        # Fix Alpha and udpate D
        alpha.requires_grad = False        
        D =  get_close_form_D(alpha, X, lamda_reg=lamda, device=device)
        
        if n>0:
            # Fix D and re-update alpha:
            alpha.requires_grad = True
            for _ in range(trainCfg['alpha_iter']):
                optimizerAlpha.zero_grad()
                lossMSEAlpha = get_loss_mse(alpha, D, X, loss_amp=loss_amp)
                lossMSEAlpha.backward()
                optimizerAlpha.step()

        with torch.no_grad():
            lossMSE_batch = get_loss_mse(alpha, D, X, loss_amp=loss_amp, per_elem_batch=True)
            loss1_eval, loss2_eval = lossMSE_batch.mean().item(), get_loss_reg(D, loss_amp=loss_amp, device=device)
            if K>1:
                loss2_eval = loss2_eval.item()
            if (1-lamda)*loss1_eval+lamda*loss2_eval<=best_epoch['loss'].mean()*(1-lamda)+lamda*best_epoch['loss_reg']:
                if count_val%100==0 and verbose:
                    print(f'Epoch {n} | Loss Total: {loss1_eval + lamda*loss2_eval}, loss MSE: {loss1_eval}, loss reg: {loss2_eval}')
                count_val += 1
                best_epoch['loss'] = lossMSE_batch
                best_epoch['loss_reg'] = loss2_eval
                best_epoch['n'] = n
                best_epoch['D'] = D.data.cpu().clone()
                best_epoch['alpha'] = alpha.data.cpu().clone()

            if trainCfg['early_stopping'] and abs((1-lamda)*loss1_eval+lamda*loss2_eval-previous_loss)<=trainCfg['early_stopping_eps']:
                if verbose:
                    print(f'Early stopping at epoch {n}')
                break
            previous_loss = (1-lamda)*loss1_eval+lamda*loss2_eval 
    if verbose:
        print(f'Best epoch {best_epoch["n"]} | Total Loss : {best_epoch["loss"].mean().item() + lamda*best_epoch["loss_reg"]} | Loss MSE: {best_epoch["loss"].mean().item()} | Loss reg: {best_epoch["loss_reg"]}')
    return best_epoch['D'], best_epoch["loss"]   # load images as tensors
    

def find_summetsBatch(data, args, method='simplex',thresh_elbow=1.5, return_jumpsMSE=False, lamda_reg=0.05, n_iter=100, alpha_iter=5, normalize=False, trainCfg={'lr':0.1, 'mmt':0.8, 'D_iter':1, 'loss_amp':100, 'loss_alpha':1}, verbose=False, maxK=3, concat=True, K=None):
    """
    Find summets of all images in a batch of data, returns the best number of summets per image based on the elbow method.
    Arguments:
        - data: shape is [batch, n_aug, dim].
        - args : args parser.
        - thresh_elbow : threshold for K selection based on MSE ratio.
        - return_jumpsMSE: return losses.
        - lamda_reg : regularization factor.
        - maxK : max number of summets to look for.
        - concat : if True, return the solutions of the batch in a Tensor format with padding.
        - n_iter : number of iterations to perform in the optimization loop.
        - alpha_iter : number of iterations of GD to perform inside each optimization loop before calculating the close form of the summets.
    """
    trainCfg['alpha_iter'] = alpha_iter
    trainCfg['n_iter'] = n_iter
    D_solutions = []
    # Run optim on multiple values of K summets for all data in the run:
    dim = data.shape[-1]
    if normalize: 
        mean = torch.mean(data.reshape(-1, dim), dim=0, keepdim=True).unsqueeze(0)
        std = torch.std(data.reshape(-1, dim), dim=0, keepdim=True).unsqueeze(0)
        data = (data-mean)/std
    K_list = {}
    MSE_list = []
    if maxK > 0:
        for K in range(1,maxK+1):
            if method=='simplex':
                D_sol, lossMSE = gradient_descent_ClosedForm(data, K, lamda=lamda_reg, trainCfg=trainCfg, D_init='oui', verbose=verbose, device=args.device)
            elif method=='kmeans':
                D_sol, lossMSE = kmeans(data, K)
            if normalize:
                D_sol = D_sol*std + mean
            K_list[K] = D_sol
            MSE_list.append(lossMSE)

        # Get the best value of K:
        jumps = torch.stack([MSE_list[k]/MSE_list[k+1] for k in range(len(MSE_list)-1)], axis=1)
        max_jump, bestK = jumps.max(axis=1)
        bestK[max_jump<thresh_elbow] = 1 # if unsignificant jump then 
        bestK[max_jump>=thresh_elbow] += 2 
        D_solutions = [K_list[K.item()][b] for b,K in enumerate(bestK)]
        if concat:
            # Pad D tensor:
            minK, maxK = min(bestK), max(bestK)
            if minK!=maxK:
                for k in range(len(D_solutions)):
                    if bestK[k]<maxK:
                        D_solutions[k] = torch.cat([D_solutions[k].reshape(-1, dim), torch.zeros(maxK-bestK[k], dim)])
                
            D = torch.stack(D_solutions).reshape(data.shape[0], -1, maxK, dim)
            bestK = bestK.reshape(data.shape[0], -1)
            if return_jumpsMSE:
                return D, bestK, jumps, MSE_list
            else:
                return D, bestK
        else:
            return D_solutions
    else:
        if method=='simplex':
            D_sol, lossMSE = gradient_descent_ClosedForm(data, K, lamda=lamda_reg, trainCfg=trainCfg, D_init='oui', verbose=verbose, device=args.device)
        elif method=='kmeans':
            D_sol, lossMSE = kmeans(data, K)
        if normalize:
            D_sol = D_sol*std + mean
        return [D_sol]


def get_closest_predictions(D, K_solutions_t, n_shots, args):
    """
    Return closest predictions based on the summets of the simplex
    - D: Tensor of the summets of the simplex
    - K_solutions_t: True number of summets for each sample, since in D in order to preserve a tensor structure, we pad it with zeros values
    """
    D_queries = D[:,n_shots:]
    D_support = D[:,:n_shots]
    predictions = []
    for c in range(D_queries.shape[0]):
        for s in range(D_queries.shape[1]):
            # One query sample here
            D_sample = D_queries[c,s][:K_solutions_t[c,s+n_shots]]
            # Compute distance from the sample to all other support samples
            distances_summets = []
            classes_summets = []
            for summet in D_sample:
                min_distance = 10e8
                closest_classe = -1
                # for each classe
                for support_class in range(args.n_ways):
                    # fix a shot
                    for shot in range(n_shots):
                        D_supp = D_support[support_class,shot][:K_solutions_t[support_class,shot]]
                        distance = torch.norm(summet-D_supp, dim=1)
                        min_dist_support = min(distance)
                        if min_dist_support<min_distance:
                            min_distance = min_dist_support
                            closest_classe = support_class
                distances_summets.append(min_distance)
                classes_summets.append(closest_classe)
            distances_summets = torch.tensor(distances_summets)
            classes_summets = torch.tensor(classes_summets)
            prediction = classes_summets[distances_summets.argmin()]
            predictions.append(prediction)
    predictions = torch.stack(predictions).reshape(args.n_ways, -1)
    return predictions

def kmeans(data, K=2):
    """
        Given a batch of data points, return centroids used kmeans algorithm
    """
    from sklearn.cluster import KMeans
    batchsize = data.shape[0]
    losses = []
    centroids = []
    for b in range(batchsize):
        kmeanModel = KMeans(n_clusters=K)
        kmeanModel.fit(data[b].cpu())
        losses.append(kmeanModel.inertia_)
        centroids.append(torch.from_numpy(kmeanModel.cluster_centers_))
    return torch.stack(centroids), torch.Tensor(losses)