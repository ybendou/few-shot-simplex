import torch
import numpy as np
import random
def generate_runs(data, run_classes, run_indices, batch_idx, batch_few_shot_runs=100):
    """
    Generates a few-shot run.
    Args:
        data: data to be used for the few-shot run
        run_classes: classes of the few-shot run
        run_indices: indices of the few-shot run
        batch_idx: index of the batch
        batch_few_shot_runs: number of runs per batch
    Returns:
        few_shot_run: a few-shot run
    """
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)
    return res

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class, n_runs, device='cuda:0'):
    """
    Define runs either randomly or by specifying one sample to insert either as a query or as a support
    Args:
        n_ways: number of classes in the few-shot run
        n_shots: number of samples per class in the few-shot run
        n_queries: number of queries per class in the few-shot run
        num_classes: number of classes in the dataset
        elements_per_class: number of elements in each class
        n_runs: number of runs to generate
    Returns:
        run_classes: classes of the few-shot run
        run_indices: indices of the few-shot run
    """
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices


def create_runs(args, elements, n_runs, num_classes, classe=None, sample=None, sample_is_support=True, elements_per_class=[]):
    """
    Define runs either randomly or by specifying one sample to insert either as a query or as a support
    """
    if classe == None and sample == None:
        runs = list(zip(*[define_runs(args.n_ways, s, args.n_queries, num_classes, elements, n_runs, device=args.device) for s in args.n_shots]))
        run_classes, run_indices = runs[0], runs[1]
    else:
        run_classes = []
        run_indices = []
        for n_shots in args.n_shots:
            # classes : 
            classe_choices = [i for i in range(num_classes)]
            classe_choices.remove(classe)
            classe_choices = torch.tensor(classe_choices)
            classe_choices_grid = torch.stack([classe_choices[torch.randperm(num_classes-1)] for _ in range(n_runs)])
            one_run_classes = torch.zeros(n_runs, args.n_ways)
            one_run_classes[:, 0] = classe
            one_run_classes[:, 1:] = classe_choices_grid[:,:args.n_ways-1]
            run_classes.append(one_run_classes.long().to(args.device))

            one_run_indices = []
            for _ in range(n_runs):
                indices_choices = [i for i in range(600)]
                indices_choices = torch.tensor(indices_choices)
                indices_choices_grid = torch.stack([torch.randperm(elements_per_class) for _ in range(args.n_ways)])[:, :n_shots+args.n_queries]

                # Make sure that the sample is not repeated twice
                indices_choices_sample = [i for i in range(600)]
                indices_choices_sample.remove(sample)
                indices_choices_sample = torch.tensor(indices_choices_sample)
                indices_choices_grid[0] = indices_choices_sample[torch.randperm(elements_per_class-1)][:n_shots+args.n_queries]

                sample_pos = {True:0, False:-1}[sample_is_support]
                indices_choices_grid[0,sample_pos] = sample # either insert in the beginning as support or as query in the end
                one_run_indices.append(indices_choices_grid)

            one_run_indices = torch.stack(one_run_indices)
            run_indices.append(one_run_indices.long().to(args.device))
    return run_classes, run_indices

def get_closest_features_to_simplex(simplex, crops_features):
    """
    Return closest image crops to each of the simplex summets
    Args:
        simplex: simplex features
        crops_features: image crops
    Returns:
        closest_crops: closest image crops to each of the simplex summets
    """
    dim = crops_features.shape[-1]
    distances = torch.norm(simplex.reshape(-1, 1, dim) - crops_features.reshape(1, -1, dim), dim=2, p=2)
    closest_crops = torch.argmin(distances, dim=1)
    return torch.stack([crops_features[index_aug] for index_aug in closest_crops])
