import typing as ty
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import prody as pd
from scipy.signal import resample

from geometricus import moment_utility, protein_utility, utility

Shapemer = ty.Tuple[int, int, int, int]
Shapemers = ty.List[Shapemer]
SplitType: IntEnum = IntEnum('SplitType', ('kmer', 'radius', 'radius_upsample', 'allmer', 'kmer_cut'))


@dataclass
class GeometricusEmbedding:
    """
    Class for storing embedding information
    Embedding matrix of size (len(protein_keys), len(self.kmer_shape_keys) + len(self.radius_shape_keys))
    is stored in self.embedding

    Parameters
    ----------
    invariants
        dict mapping protein_key: MomentInvariants class based on kmer fragmentation
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

    invariants: ty.Dict[str, 'MomentInvariants']
    resolution: ty.Union[float, np.ndarray]
    protein_keys: ty.List[str]
    shapemer_keys: Shapemers
    embedding: np.ndarray
    shapemer_to_protein_indices: ty.Dict[Shapemer, ty.List[ty.Tuple[int, int]]]
    proteins_to_shapemers: ty.Dict[str, Shapemers]

    @classmethod
    def from_invariants(cls,
                        invariants: ty.List['MomentInvariants'],
                        resolution: ty.Union[float, np.ndarray] = 1.,
                        protein_keys: ty.Union[None, ty.List[str]] = None,
                        shapemer_keys: ty.Union[None, ty.List[Shapemer]] = None):
        """

        Parameters
        ----------
        invariants
            List of MomentInvariant objects.
        resolution

        protein_keys

        shapemer_keys

        Returns
        -------

        """
        if type(resolution) == np.ndarray:
            assert resolution.shape[0] == moment_utility.NUM_MOMENTS
        invariants: ty.Dict[str, 'MomentInvariants'] = {x.name: x for x in invariants}
        if protein_keys is None:
            protein_keys: ty.List[str] = [k for k in invariants.keys()]
        assert all(k in invariants for k in protein_keys)
        shapemers, shapemer_to_indices, embedding, shapemer_keys = moments_to_embedding(
                [invariants[name] for name in protein_keys], resolution=resolution, restrict_to=shapemer_keys)
        return cls(invariants, resolution, protein_keys, shapemer_keys, embedding, shapemer_to_indices)

    def embed(self,
              invariants: ty.List['MomentInvariants'],
              protein_keys: ty.Union[None, ty.List[str]] = None) -> 'GeometricusEmbedding':
        return self.from_invariants(invariants, self.resolution, protein_keys, self.shapemer_keys)


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
        One of "kmer", "kmer_cut", "allmer", "radius", "radius_upsample" to generate structural fragments
        - kmer - each residue is taken as the center of a kmer of length split_size, ends are included but shorter
        - kmer_cut - same as kmer but ends are not included, only fragments of length split_size are kept
        - allmer - adds kmers of different lengths (split_size - 5 to split_size + 5)
                to take into account deletions/insertions that don't change the shape
        - radius - overlapping spheres of radius split_size
        - radius_upsample - overlapping spheres of radius split_size on coordinates upsampled by upsample rate.
            Only works if there's one atom selected per residue
    split_size
        kmer size or radius (depending on split_type)
    upsample_rate
        ignored unless split_type = "radius_upsample"
    split_indices
        Filled with a list of residue indices for each structural fragment
    moments
        Filled with moment invariant values for each structural fragment
    """
    residue_splits: list = field(repr=False)
    original_indices: np.ndarray = field(repr=False)
    sequence: str = ty.Union[str, None]
    split_type: SplitType = SplitType.kmer
    split_size: int = 16
    upsample_rate: int = 50
    split_indices: ty.Union[ty.List, None] = None
    moments: np.ndarray = None

    @classmethod
    def from_coordinates(cls, name: str, coordinates: np.ndarray, sequence: ty.Union[str, None] = None,
                         split_type: SplitType = SplitType.kmer, split_size: int = 16, upsample_rate: int = 50):
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
                    split_size=split_size,
                    upsample_rate=upsample_rate)
        shape._split(split_type)
        return shape

    @classmethod
    def from_prody_atomgroup(cls, name: str, protein: pd.AtomGroup,
                             split_type: SplitType = SplitType.kmer, split_size: int = 16,
                             selection: str = "calpha", upsample_rate: int = 50):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)
        """
        protein: pd.AtomGroup = protein.select("protein").select(selection)
        coordinates: np.ndarray = protein.getCoords()
        residue_splits = utility.group_indices(protein.getResindices())
        shape = cls(name, len(residue_splits), coordinates,
                    residue_splits,
                    protein.getIndices(),
                    sequence=protein.getSequence(),
                    split_type=split_type,
                    split_size=split_size,
                    upsample_rate=upsample_rate)
        shape._split(split_type)
        return shape

    def _split(self, split_type: SplitType):
        if split_type == SplitType.kmer:
            split_indices, moments = self._kmerize()
        elif split_type == SplitType.allmer:
            split_indices, moments = self._allmerize()
        elif split_type == SplitType.radius:
            split_indices, moments = self._split_radius()
        elif split_type == SplitType.radius_upsample:
            assert all(len(r) == 1 for r in self.residue_splits)
            split_indices, moments = self._split_radius_upsample()
        elif split_type == SplitType.kmer_cut:
            split_indices, moments = self._kmerize_cut_ends()
        else:
            raise Exception("split_type must be one of kmer, kmer_cut, allmer, and radius")
        self.split_indices = split_indices
        self.moments = moments

    @classmethod
    def from_pdb_file(cls, pdb_file: ty.Union[str, Path], chain: str = None, split_type: str = "kmer", split_size=16, upsample_rate=50):
        """
        Construct MomentInvariants instance from a PDB file and optional chain.
        Selects alpha carbons only.
        """
        pdb_name = utility.get_file_parts(pdb_file)[1]
        protein = pd.parsePDB(str(pdb_file))
        if chain is not None:
            protein = protein.select(f"chain {chain}")
        return cls.from_prody_atomgroup(pdb_name, protein, split_type, split_size, upsample_rate=upsample_rate)

    @classmethod
    def from_pdb_id(cls, pdb_name: str, chain: str = None, split_type: str = "kmer", split_size=16, upsample_rate=50):
        """
        Construct MomentInvariants instance from a PDB ID and optional chain (downloads the PDB file from RCSB).
        Selects alpha carbons only.
        """
        if chain:
            protein = pd.parsePDB(pdb_name, chain=chain)
        else:
            protein = pd.parsePDB(pdb_name)
        return cls.from_prody_atomgroup(pdb_name, protein, split_type, split_size, upsample_rate=upsample_rate)

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

    def _split_radius_upsample(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)

        split_indices_upsample = []
        coordinates_upsample = resample(self.coordinates, self.upsample_rate * self.coordinates.shape[0])
        kd_tree_upsample = pd.KDTree(coordinates_upsample)

        for i in range(self.length):
            kd_tree_upsample.search(center=self.coordinates[self.residue_splits[i][0]], radius=self.split_size)
            split_indices_upsample.append(kd_tree_upsample.getIndices())
            kd_tree.search(center=self.coordinates[self.residue_splits[i][0]], radius=self.split_size)
            split_indices.append(kd_tree.getIndices())
        moments = np.zeros((len(split_indices), moment_utility.NUM_MOMENTS))
        for i in range(len(split_indices_upsample)):
            moments[i] = moment_utility.get_second_order_moments(coordinates_upsample[split_indices_upsample[i]])
        return split_indices, moments


def moments_to_embedding(moments: ty.List[MomentInvariants],
                         resolution: ty.Union[float, np.ndarray], restrict_to: list = None) -> \
        ty.Tuple[
            ty.Dict[str, Shapemers], ty.Dict[Shapemer, ty.List[ty.Tuple[int, int]]], np.ndarray, ty.List[str]]:
    """list of MomentInvariants to
        - shapemers - list of shape-mers per protein
        - shapemer_to_indices - dict of shape-mer: list of (protein_index, residue_index)s in which it occurs
        - embedding - Geometricus embedding matrix
        - shape_keys - list of shape-mers in order of embedding matrix
    """
    shapemers: ty.List[Shapemers] = get_shapes(moments, resolution)
    shapemer_to_indices = map_shapemers_to_indices(shapemers)
    if restrict_to is None:
        restrict_to = sorted(list(shapemer_to_indices))
    embedding = make_embedding(shapemers, restrict_to)
    return {k.name: shapemers[i] for i, k in enumerate(moments)}, shapemer_to_indices, embedding, restrict_to


def get_shapes(moment_invariants_list: ty.List[MomentInvariants],
               resolution: ty.Union[float, np.ndarray] = 2.) -> ty.List[Shapemers]:
    """
    moment invariants -> log transformation -> multiply by resolution -> floor = moment_invariants-mers
    """
    shapemers: ty.List[Shapemers] = []
    if type(resolution) == np.ndarray:
        assert resolution.shape[0] == moment_invariants_list[0].moments.shape[1]
    for moment_invariants in moment_invariants_list:
        shapemers.append([tuple(x) for x in (np.log1p(moment_invariants.moments) * resolution).astype(int)])
    return shapemers


def map_shapemers_to_indices(shapemers_list: ty.List[Shapemers]) -> ty.Dict[Shapemer, ty.List[ty.Tuple[int, int]]]:
    """
    Maps shape-mer to (protein_index, residue_index)
    """
    shapemer_to_indices: ty.Dict[Shapemer, ty.List[ty.Tuple[int, int]]] = defaultdict(list)
    for i, shapemers in enumerate(shapemers_list):
        for j in range(len(shapemers)):
            shapemer_to_indices[shapemers[j]].append((i, j))
    return shapemer_to_indices


def make_embedding(shapemers: ty.List[Shapemers],
                   restrict_to: ty.List[Shapemer]) -> np.ndarray:
    """
    Counts occurrences of each shape-mer across proteins
    """
    shape_to_index = dict(zip(restrict_to, range(len(restrict_to))))
    embedding = np.zeros((len(shapemers), len(restrict_to)))
    for i in range(len(shapemers)):
        for m in shapemers[i]:
            if m in shape_to_index:
                embedding[i, shape_to_index[m]] += 1
    return embedding
