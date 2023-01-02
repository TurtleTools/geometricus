from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from geometricus.moment_invariants import SPLIT_INFOS, MomentType


class ShapemerLearn(torch.nn.Module):
    def __init__(self, hidden_layer_dimension=32, output_dimension=10, split_infos=SPLIT_INFOS):
        super(ShapemerLearn, self).__init__()
        self.split_infos = split_infos
        self.number_of_moments = len(split_infos) * len(MomentType)
        self.hidden_layer_dimension = hidden_layer_dimension
        self.output_dimension = output_dimension
        self.linear_segment = torch.nn.Sequential(
            torch.nn.Linear(self.number_of_moments, hidden_layer_dimension),
            torch.nn.BatchNorm1d(hidden_layer_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dimension, hidden_layer_dimension),
            torch.nn.BatchNorm1d(hidden_layer_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dimension, hidden_layer_dimension),
            torch.nn.BatchNorm1d(hidden_layer_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dimension, output_dimension),
            torch.nn.BatchNorm1d(output_dimension),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, y, z):
        return self.linear_segment(x), self.linear_segment(y), z

    def forward_single_segment(self, x):
        return self.linear_segment(x)

    def save(self, folder):
        self.eval()
        torch.save(self.state_dict(),
                   folder / self.filename)

    @property
    def filename(self):
        split_info_string = "_".join(
            [f"{split_info.split_type.name}-{split_info.split_size}" for split_info in self.split_infos])

        return f"ShapemerLearn_{split_info_string}_{self.number_of_moments}_{self.hidden_layer_dimension}_{self.output_dimension}.pt "

    @classmethod
    def load(cls, folder, hidden_layer_dimension=32, output_dimension=10, split_infos=SPLIT_INFOS):
        model = ShapemerLearn(hidden_layer_dimension, output_dimension, split_infos=split_infos)
        filename = Path(folder) / model.filename
        assert filename.exists(), f"Model file {filename} does not exist"
        if torch.cuda.is_available():
            m = torch.load(str(filename), map_location=torch.device("cuda"))
        else:
            m = torch.load(str(filename), map_location=torch.device("cpu"))
        model.load_state_dict(m)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model


# def loss_func(out, distant, y):
#     dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
#     dist = torch.sqrt(dist_sq + 1e-10)
#     mdist = 1 - dist
#     dist = torch.clamp(mdist, min=0.0)
#     loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#     loss = torch.sum(loss) / 2.0 / out.size()[0]
#     return loss

def loss_func(out, distant, y):
    # Calculate the squared Euclidean distance between out and distant
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    # Calculate the contrastive loss
    loss = y * dist_sq + (1 - y) * torch.pow(torch.clamp(1 - torch.sqrt(dist_sq + 1e-10), min=0.0), 2)
    # # Add a penalty for outputs that are far from 0 or 1
    # alpha = 0.1  # Hyperparameter to control the strength of the penalty
    # loss += torch.mean(alpha * torch.abs(out * (1 - out)), 1)
    # Return the mean loss over the batch
    return torch.mean(loss)


def moment_tensors_to_bits(list_of_moment_tensors):
    bits = []
    for i, segment in enumerate(list_of_moment_tensors):
        bits.append(tuple(list((segment > 0.5).astype(np.uint8))))
    return bits


def moments_to_tensors(segments, model):
    if torch.cuda.is_available():
        return model.forward_single_segment(torch.tensor(segments).cuda()).cpu().detach().numpy()
    return model.forward_single_segment(torch.tensor(segments)).cpu().detach().numpy()


def moments_to_shapemers(list_of_moments, model):
    if torch.cuda.is_available():
        moment_tensors = (
            model.forward_single_segment(torch.tensor(list_of_moments).cuda())
            .cpu()
            .detach()
            .numpy()
        )
    else:
        moment_tensors = (
            model.forward_single_segment(torch.tensor(list_of_moments))
            .cpu()
            .detach()
            .numpy()
        )
    return moment_tensors_to_bits(moment_tensors)


def get_all_keys(list_of_moment_hashes, model):
    all_keys = set()
    for prot in list_of_moment_hashes:
        all_keys |= set(moments_to_shapemers(prot, model))
    return list(all_keys)


def count_with_keys(prot_hashes, keys):
    d = OrderedDict.fromkeys(keys, 0)
    for prot_hash in prot_hashes:
        d[prot_hash] += 1
    return np.array([d[x] for x in keys])


def get_count_matrix(protein_moments, model, with_counts=False):
    ind_moments_compressed = [
        moments_to_shapemers(x, model) for x in protein_moments
    ]
    all_keys = get_all_keys(protein_moments, model)
    print(f"Total shapemers: {len(all_keys)}")
    protein_embeddings = [count_with_keys(x, all_keys) for x in ind_moments_compressed]
    if with_counts:
        return [x / x.sum() for x in protein_embeddings], protein_embeddings
    return protein_embeddings
