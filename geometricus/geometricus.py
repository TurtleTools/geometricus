import typing
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import prody as pd

from geometricus import moment_utility, protein_utility, utility


class GeometricusEmbedding:
    """
    Class for storing embedding information
    Embedding matrix of size (len(protein_keys), len(self.kmer_shape_keys) + len(self.radius_shape_keys))
    is stored in self.embedding

    Parameters
    ----------
    kmer_invariants
        dict mapping protein_key: MomentInvariants class based on kmer fragmentation
    radius_invariants
        dict mapping protein_name: MomentInvariants class based on radius fragmentation
    resolution
        multiplier that determines how coarse/fine-grained each shape is
        this can be a single number, multiplied to all four moment invariants
        or a numpy array of four numbers, one for each invariant
    protein_keys
        list of protein names = rows of the output embedding
    kmer_shape_keys
        if given uses only these shape-mers for the embedding
        if None, uses all shape-mers
    radius_shape_keys
        if given uses only these shape-mers for the embedding
        if None, uses all shape-mers
    """
    def __init__(self, kmer_invariants: dict, radius_invariants: dict, resolution: typing.Union[float, np.ndarray], protein_keys: list,
                 kmer_shape_keys: list = None, radius_shape_keys: list = None):

        self.protein_keys = protein_keys

        self.kmer_invariants = kmer_invariants
        self.radius_invariants = radius_invariants

        self.protein_to_kmer_shapes, self.kmer_shape_to_proteins, self.kmer_embedding, self.kmer_shape_keys = moments_to_embedding(
            [self.kmer_invariants[name] for name in protein_keys], resolution=resolution, shape_keys=kmer_shape_keys)
        self.protein_to_radius_shapes, self.radius_shape_to_proteins, self.radius_embedding, self.radius_shape_keys = moments_to_embedding(
            [self.radius_invariants[name] for name in protein_keys], resolution=resolution, shape_keys=radius_shape_keys)

        self.embedding = np.hstack((self.kmer_embedding, self.radius_embedding))

    def map_shapemer_index_to_shapemer(self, shapemer_index: int) -> tuple:
        """
        Gets shape-mer at a particular index in self.embedding.

        Parameters
        ----------
        shapemer_index
            index of the shape-mer in self.embedding

        Returns
        -------
        shape-mer
        """
        kmer_shapemer_nr = len(self.kmer_shape_keys)
        if shapemer_index >= kmer_shapemer_nr:  # this means the feature is a radius invariant
            shape_key = self.radius_shape_keys[shapemer_index - kmer_shapemer_nr]

        else:  # this means the feature is a kmer invariant
            shape_key = self.kmer_shape_keys[shapemer_index]
        return shape_key

    def map_shapemer_index_to_residues(self, shapemer_index: int) -> dict:
        """
        Gets residues within a particular shape-mer across all proteins.

        Parameters
        ----------
        shapemer_index
            index of the shape-mer in self.embedding

        Returns
        -------
        dict of protein_key: set(residues in shape-mer)
        """
        protein_to_shapemer_residues = defaultdict(set)

        kmer_shapemer_nr = len(self.kmer_shape_keys)
        shape_key = self.map_shapemer_index_to_shapemer(shapemer_index)
        if shapemer_index >= kmer_shapemer_nr:  # this means the feature is a radius invariant
            invariants = self.radius_invariants
            shape_to_proteins = self.radius_shape_to_proteins

        else:  # this means the feature is a kmer invariant
            invariants = self.kmer_invariants
            shape_to_proteins = self.kmer_shape_to_proteins

        locations = shape_to_proteins[shape_key]

        for protein_index, residue_index in locations:
            protein_name = self.protein_keys[protein_index]
            moment_invariants = invariants[protein_name]
            shapemer_residues = moment_invariants.split_indices[residue_index]
            for residue in shapemer_residues:
                protein_to_shapemer_residues[protein_name].add(residue)

        return protein_to_shapemer_residues


@dataclass(eq=False)
class MomentInvariants(protein_utility.Structure):
    """
    Class for storing moment invariants for a protein.
    Use from_* constructors to make an instance of this.
    Subclasses Structure so also has a name, length, and optional sequence.

    Parameters
    ----------
    residue_splits
        split a protein into residues using these atom indices, e.g
        [[1, 2], [3, 4]] could represent both alpha and beta carbons being used in a residue.
        Right now only the alpha carbons are used so this contains indices of alpha carbons for each residue if prody is used
        or just indices in a range if coordinates are given directly
    original_indices:
        Also alpha indices (if prody) / range (if coordinates)
    sequence
        Amino acid sequence
    split_type
        One of "kmer", "kmer_cut", "allmer", "radius" to generate structural fragments
        - kmer - each residue is taken as the center of a kmer of length split_size, ends are included but shorter
        - kmer_cut - same as kmer but ends are not included, only fragments of length split_size are kept
        - allmer - adds kmers of different lengths (split_size - 5 to split_size + 5)
                to take into account deletions/insertions that don't change the shape
        - radius - overlapping spheres of radius split_size
    split_size
        kmer size or radius (depending on split_type)
    split_indices
        Filled with a list of residue indices for each structural fragment
    moments
        Filled with moment invariant values for each structural fragment
    """
    residue_splits: list = field(repr=False)
    original_indices: np.ndarray = field(repr=False)
    sequence: str = typing.Union[str, None]
    split_type: str = "kmer"
    split_size: int = 16
    split_indices: typing.Union[typing.List, None] = None
    moments: np.ndarray = None

    @classmethod
    def from_coordinates(cls, name, coordinates: np.ndarray, sequence: typing.Union[str, None] = None, split_type="kmer", split_size=16):
        """
        Construct MomentInvariants instance from a set of coordinates.
        Assumes one coordinate per residue.
        """
        residue_splits = [[i] for i in range(coordinates.shape[0])]
        shape = cls(name, coordinates.shape[0], coordinates,
                    residue_splits,
                    np.arange(coordinates.shape[0]),
                    sequence=sequence,
                    split_type=split_type,
                    split_size=split_size)
        shape._split(split_type)
        return shape

    @classmethod
    def from_prody_atomgroup(cls, name, protein: pd.AtomGroup, split_type="kmer", split_size=16, selection: str = "calpha"):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)
        """
        protein = protein.select("protein").select(selection)
        coordinates = protein.getCoords()
        residue_splits = utility.group_indices(protein.getResindices())
        shape = cls(name, coordinates.shape[0], coordinates, residue_splits,
                    protein.getIndices(),
                    sequence=protein.getSequence(),
                    split_type=split_type,
                    split_size=split_size)
        shape._split(split_type)
        return shape

    def _split(self, split_type):
        if split_type == "kmer":
            split_indices, moments = self._kmerize()
        elif split_type == "allmer":
            split_indices, moments = self._allmerize()
        elif split_type == "radius":
            split_indices, moments = self._split_radius()
        elif split_type == "kmer_cut":
            split_indices, moments = self._kmerize_cut_ends()
        else:
            raise Exception("split_type must be one of kmer, kmer_cut, allmer, and radius")
        self.split_indices = split_indices
        self.moments = moments

    @classmethod
    def from_pdb_file(cls, pdb_file: typing.Union[str, Path], chain: str = None, split_type: str = "kmer", split_size=16):
        """
        Construct MomentInvariants instance from a PDB file and optional chain.
        Selects alpha carbons only.
        """
        pdb_name = utility.get_file_parts(pdb_file)[1]
        protein = pd.parsePDB(str(pdb_file))
        if chain is not None:
            protein = protein.select(f"chain {chain}")
        return cls.from_prody_atomgroup(pdb_name, protein, split_type, split_size)

    @classmethod
    def from_pdb_id(cls, pdb_name: str, chain: str = None, split_type: str = "kmer", split_size=16):
        """
        Construct MomentInvariants instance from a PDB ID and optional chain (downloads the PDB file from RCSB).
        Selects alpha carbons only.
        """
        if chain:
            protein = pd.parsePDB(pdb_name, chain=chain)
        else:
            protein = pd.parsePDB(pdb_name)
        return cls.from_prody_atomgroup(pdb_name, protein, split_type, split_size)

    def _kmerize(self):
        split_indices = []
        for i in range(self.length):
            kmer = []
            for j in range(max(0, i - self.split_size // 2), min(len(self.residue_splits), i + self.split_size // 2)):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _kmerize_cut_ends(self):
        split_indices = []
        overlap = 1
        for i in range(self.split_size // 2, self.length - self.split_size // 2, overlap):
            kmer = []
            for j in range(max(0, i - self.split_size // 2), min(len(self.residue_splits), i + self.split_size // 2)):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _allmerize(self):
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

    def _split_radius(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)
        for i in range(self.length):
            kd_tree.search(center=self.coordinates[self.residue_splits[i][0]], radius=self.split_size)
            split_indices.append(kd_tree.getIndices())
        return self._get_moments(split_indices)


def moments_to_embedding(moments: typing.List[MomentInvariants], resolution: typing.Union[float, np.ndarray], shape_keys: list = None) -> \
        typing.Tuple[
            list, dict, np.ndarray, list]:
    """list of MomentInvariants to
        - shapemers - list of shape-mers per protein
        - shapemer_to_indices - dict of shape-mer: list of (protein_index, residue_index)s in which it occurs
        - embedding - Geometricus embedding matrix
        - shape_keys - list of shape-mers in order of embedding matrix
    """
    shapemers = get_shapes(moments, resolution)
    shapemer_to_indices = map_shapemers_to_indices(shapemers)
    if shape_keys is None:
        shape_keys = sorted(list(shapemer_to_indices))
    embedding = make_embedding(shapemers, shape_keys)
    return shapemers, shapemer_to_indices, embedding, shape_keys


def get_shapes(moments: typing.List[MomentInvariants], resolution: typing.Union[float, np.ndarray] = 2.) -> list:
    """
    moment invariants -> log transformation -> multiply by resolution -> floor = shape-mers
    """
    shapemers = []
    if type(resolution) == np.ndarray:
        assert resolution.shape[0] == moment_utility.NUM_MOMENTS
    for shape in moments:
        shapemers.append((np.log1p(shape.moments) * resolution).astype(int))
    return shapemers


def map_shapemers_to_indices(binned_moments: list) -> dict:
    """
    Maps shape-mer to (protein_index, residue_index)
    """
    shapemer_to_indices = defaultdict(list)
    for i, moments in enumerate(binned_moments):
        for j in range(moments.shape[0]):
            shapemer_to_indices[tuple(moments[j])].append((i, j))
    return shapemer_to_indices


def make_embedding(binned_moments: list, shape_keys: list) -> np.ndarray:
    """
    Counts occurrences of each shape-mer across proteins
    """
    shape_to_index = dict(zip(shape_keys, range(len(shape_keys))))
    embedding = np.zeros((len(binned_moments), len(shape_keys)))
    for i in range(len(binned_moments)):
        for m in binned_moments[i]:
            if tuple(m) in shape_to_index:
                embedding[i, shape_to_index[tuple(m)]] += 1
    return embedding
