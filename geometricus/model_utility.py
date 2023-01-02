import numpy as np
import torch

from geometricus.moment_invariants import SPLIT_INFOS, MomentType
import importlib.resources as importlib_resources


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

        return f"ShapemerLearn_{split_info_string}_{self.number_of_moments}_{self.hidden_layer_dimension}_{self.output_dimension}.pt"

    @classmethod
    def load(cls, hidden_layer_dimension=32, output_dimension=10, split_infos=SPLIT_INFOS):
        model = ShapemerLearn(hidden_layer_dimension, output_dimension, split_infos=split_infos)
        if torch.cuda.is_available():
            m = torch.load(importlib_resources.files("geometricus") / "models" / model.filename,
                           map_location=torch.device("cuda"))
        else:
            m = torch.load(importlib_resources.files("geometricus") / "models" / model.filename,
                           map_location=torch.device("cpu"))
        model.load_state_dict(m)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model


def loss_func(out, distant, y):
    # Calculate the squared Euclidean distance between out and distant
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    # Calculate the contrastive loss
    loss = y * dist_sq + (1 - y) * torch.pow(torch.clamp(1 - torch.sqrt(dist_sq + 1e-10), min=0.0), 2)
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
