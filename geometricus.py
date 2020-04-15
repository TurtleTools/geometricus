import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import prody as pd
from collections import defaultdict
import utility, protein_utility, moment_utility


class GeometricusEmbedding:
    def __init__(self, invariants: MomentInvariants, resolution: typing.Union[float, np.ndarray], protein_keys: list, shape_keys: list = None):
        self.protein_keys = protein_keys
        self.protein_to_shapes, self.shape_to_proteins, self.embedding, self.shape_keys = geometricus.moments_to_embedding([invariants[name] for name in protein_keys], resolution = resolution, shape_keys = shape_keys)


@dataclass(eq=False)
class MomentInvariants(protein_utility.Structure):
    residue_splits: list = field(repr=False)
    original_indices: np.ndarray = field(repr=False)
    sequence: str = None
    split_type: str = "kmer"
    split_size: int = 16
    split_indices: typing.Union[typing.List, None] = None
    moments: np.ndarray = None

    @classmethod
    def from_coordinates(cls, name, coordinates: np.ndarray, sequence, split_type="kmer", split_size=16):
        residue_splits = [[i] for i in range(coordinates.shape[0])]
        shape = cls(name, coordinates.shape[0], coordinates,
                    residue_splits,
                    np.arange(coordinates.shape[0]),
                    sequence=sequence,
                    split_type=split_type,
                    split_size=split_size)
        shape.split(split_type)
        return shape

    @classmethod
    def from_prody_atomgroup(cls, name, protein: pd.AtomGroup, split_type="kmer", split_size=16):
        protein = protein.select("protein")
        indices = protein_utility.get_alpha_indices(protein)
        protein = protein[indices]
        coordinates = protein.getCoords()
        sequence = protein.getSequence()
        residue_splits = utility.group_indices(protein.getResindices())
        shape = cls(name, coordinates.shape[0], coordinates, residue_splits, np.array(indices), sequence=sequence, split_type=split_type, split_size=split_size)
        shape.split(split_type)
        return shape

    def split(self, split_type):
        if split_type == "kmer":
            split_indices, moments = self.kmerize()
        elif split_type == "allmer":
            split_indices, moments = self.allmerize()
        elif split_type == "radius":
            split_indices, moments = self.split_radius()
        elif split_type == "kmer_cut":
            split_indices, moments = self.kmerize_cut_ends()
        else:
            raise Exception("split_type must be one of kmer, kmer_cut, allmer, and radius")
        self.split_indices = split_indices
        self.moments = moments

    @classmethod
    def from_pdb_file(cls, pdb_file: typing.Union[str, Path], chain: str = None, split_type: str = "kmer", split_size=16):
        pdb_name = utility.get_file_parts(pdb_file)[1]
        protein = pd.parsePDB(str(pdb_file))
        if chain is not None:
            protein = protein.select(f"chain {chain}")
        return cls.from_prody_atomgroup(pdb_name, protein, split_type, split_size)

    @classmethod
    def from_pdb_id(cls, pdb_name: str, chain: str = None, split_type: str = "kmer", split_size=16):
        if chain:
            protein = pd.parsePDB(pdb_name, chain=chain)
        else:
            protein = pd.parsePDB(pdb_name)
        return cls.from_prody_atomgroup(pdb_name, protein, include_atoms, split_type, split_size)

    def kmerize(self):
        split_indices = []
        for i in range(self.length):
            kmer = []
            for j in range(max(0, i - self.split_size//2), min(len(self.residue_splits), i + self.split_size//2)):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def kmerize_cut_ends(self):
        split_indices = []
        overlap = 1
        for i in range(self.split_size//2, self.length - self.split_size//2, overlap):
            kmer = []
            for j in range(max(0, i - self.split_size // 2), min(len(self.residue_splits), i + self.split_size // 2)):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def allmerize(self):
        split_indices = []
        for i in range(self.length):
            for split_size in range(10, self.split_size + 1, 10):
                kmer = []
                for j in range(max(0, i - split_size // 2), min(len(self.residue_splits), i + split_size // 2)):
                    kmer += self.residue_splits[j]
                split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _get_moments(self, split_indices):
        moments = np.zeros((len(split_indices), moment_utility.NUM_MOMENTS))
        for i in range(len(split_indices)):
            moments[i] = moment_utility.get_second_order_moments(self.coordinates[split_indices[i]])
        return split_indices, moments

    def split_radius(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)
        for i in range(self.length):
            kd_tree.search(center=self.coordinates[self.residue_splits[i][0]], radius=self.split_size)
            split_indices.append(kd_tree.getIndices())
        return self._get_moments(split_indices)


def moments_to_embedding(moments: typing.List[MomentInvariants], resolution: float, shape_keys: list=None):
    shapes = get_shapes(moments, resolution)
    same_shapes = get_same_shapes(shapes)
    if shape_keys is None:
        shape_keys = sorted(list(same_shapes))
    embedding = make_embedding(shapes, shape_keys)
    return shapes, same_shapes, embedding, shape_keys


def get_shapes(moments: typing.List[MomentInvariants], resolution=2.):
    binned_moments = []
    for shape in moments:
        binned_moments.append((np.log1p(shape.moments) * resolution).astype(int))
    return binned_moments


def get_same_shapes(binned_moments):
    same_shapes = defaultdict(list)
    for i, moments in enumerate(binned_moments):
        for j in range(moments.shape[0]):
            same_shapes[tuple(moments[j])].append((i, j))
    return same_shapes


def get_shape_counts(same_shapes, shape_keys):
    shape_counts = np.zeros(len(shape_keys))
    for i, k in enumerate(shape_keys):
        shape_counts[i] = len(same_shapes[k])
    return shape_counts


def make_embedding(binned_moments, shape_keys):
    shape_to_index = dict(zip(shape_keys, range(len(shape_keys))))
    embedding = np.zeros((len(binned_moments), len(shape_keys)))
    for i in range(len(binned_moments)):
        for m in binned_moments[i]:
            if tuple(m) in shape_to_index:
                embedding[i, shape_to_index[tuple(m)]] += 1
    return embedding



