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
SplitType: IntEnum = IntEnum(
    "SplitType", ("kmer", "radius", "radius_upsample", "allmer", "kmer_cut")
)


@dataclass
class GeometricusEmbedding:
    """
    Class for storing embedding information
    Embedding matrix of size (len(protein_keys), len(self.shapemer_keys)) is stored in self.embedding

    Parameters
    ----------
    invariants
        dict of protein_key: MomentInvariant
    resolution
        multiplier that determines how coarse/fine-grained each shape is
        this can be a single number, multiplied to all four moment invariants
        or a numpy array of four numbers, one for each invariant
    protein_keys
        list of protein names = rows of the output embedding
    shapemer_keys
        if given uses only these shapemers for the embedding
        if None, uses all shapemers
    embedding
        Embedding matrix of size (len(protein_keys), len(self.shapemer_keys))
    shapemer_to_protein_indices
        Maps each shapemer to the proteins which have it and to the corresponding residue indices within these proteins
    protein_to_shapemers
        Maps each protein to a list of shapemers in order of its residues
    """

    invariants: ty.Dict[str, "MomentInvariants"]
    resolution: ty.Union[float, np.ndarray]
    protein_keys: ty.List[str]
    shapemer_keys: Shapemers
    embedding: np.ndarray
    shapemer_to_protein_indices: ty.Dict[Shapemer, ty.List[ty.Tuple[str, int]]]
    proteins_to_shapemers: ty.Dict[str, Shapemers]

    @classmethod
    def from_invariants(
        cls,
        invariants: ty.List["MomentInvariants"],
        resolution: ty.Union[float, np.ndarray] = 1.0,
        protein_keys: ty.Union[None, ty.List[str]] = None,
        shapemer_keys: ty.Union[None, ty.List[Shapemer]] = None,
    ):
        """
        Make a GeometricusEmbedding object from a list of MomentInvariant objects

        Parameters
        ----------
        invariants
            List of MomentInvariant objects.
        resolution
            multiplier that determines how coarse/fine-grained each shape is
            this can be a single number, multiplied to all four moment invariants
            or a numpy array of four numbers, one for each invariant
        protein_keys
            list of protein names = rows of the output embedding
            if None, takes all keys in `invariants`
        shapemer_keys
            if given uses only these shapemers for the embedding
            if None, uses all shapemers
        """
        if isinstance(resolution, np.ndarray):
            assert resolution.shape[0] == moment_utility.NUM_MOMENTS
        invariants: ty.Dict[str, "MomentInvariants"] = {x.name: x for x in invariants}
        if protein_keys is None:
            protein_keys: ty.List[str] = list(invariants.keys())
        assert all(k in invariants for k in protein_keys)
        (
            proteins_to_shapemers,
            shapemers_to_protein_indices,
            embedding,
            shapemer_keys,
        ) = moments_to_embedding(
            protein_keys, invariants, resolution=resolution, shapemer_keys=shapemer_keys
        )
        return cls(
            invariants,
            resolution,
            protein_keys,
            shapemer_keys,
            embedding,
            shapemers_to_protein_indices,
            proteins_to_shapemers,
        )

    def embed(
        self,
        invariants: ty.List["MomentInvariants"],
        protein_keys: ty.Union[None, ty.List[str]] = None,
    ) -> "GeometricusEmbedding":
        """
        Embed a new set of proteins using an existing embedding's shapemers

        Parameters
        ----------
        invariants
            List of MomentInvariant objects.
        protein_keys
            list of protein names = rows of the output embedding

        Returns
        -------
        a GeometricusEmbedding object on the new (test) keys
        """
        return self.from_invariants(
            invariants, self.resolution, protein_keys, self.shapemer_keys
        )

    def map_shapemer_index_to_residues(self, shapemer_index: int) -> dict:
        """
        Gets residues within a particular shapemer across all proteins.

        Parameters
        ----------
        shapemer_index
            index of the shapemer in self.embedding

        Returns
        -------
        dict of protein_key: set(residues in shapemer)
        """
        protein_to_shapemer_residues = defaultdict(set)

        for protein_key, residue_index in self.shapemer_to_protein_indices[
            self.shapemer_keys[shapemer_index]
        ]:
            moment_invariants = self.invariants[protein_key]
            shapemer_residues = moment_invariants.split_indices[residue_index]
            for residue in shapemer_residues:
                protein_to_shapemer_residues[protein_key].add(residue)

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
    def from_coordinates(
        cls,
        name: str,
        coordinates: np.ndarray,
        sequence: ty.Union[str, None] = None,
        split_type: SplitType = SplitType.kmer,
        split_size: int = 16,
        upsample_rate: int = 50,
    ):
        """
        Construct MomentInvariants instance from a set of coordinates.
        Assumes one coordinate per residue.
        """
        residue_splits = [[i] for i in range(coordinates.shape[0])]
        shape = cls(
            name,
            coordinates.shape[0],
            coordinates,
            residue_splits,
            np.arange(coordinates.shape[0]),
            sequence=sequence,
            split_type=split_type,
            split_size=split_size,
            upsample_rate=upsample_rate,
        )
        shape._split(split_type)
        return shape

    @classmethod
    def from_prody_atomgroup(
        cls,
        name: str,
        protein: pd.AtomGroup,
        split_type: SplitType = SplitType.kmer,
        split_size: int = 16,
        selection: str = "calpha",
        upsample_rate: int = 50,
    ):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)
        """
        protein: pd.AtomGroup = protein.select("protein").select(selection)
        coordinates: np.ndarray = protein.getCoords()
        residue_splits = utility.group_indices(protein.getResindices())
        shape = cls(
            name,
            len(residue_splits),
            coordinates,
            residue_splits,
            protein.getIndices(),
            sequence=protein.getSequence(),
            split_type=split_type,
            split_size=split_size,
            upsample_rate=upsample_rate,
        )
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
            raise Exception(
                "split_type must be one of kmer, kmer_cut, allmer, and radius"
            )
        self.split_indices = split_indices
        self.moments = moments

    @classmethod
    def from_pdb_file(
        cls,
        pdb_file: ty.Union[str, Path],
        chain: str = None,
        split_type: str = "kmer",
        split_size=16,
        upsample_rate=50,
    ):
        """
        Construct MomentInvariants instance from a PDB file and optional chain.
        Selects alpha carbons only.
        """
        pdb_name = utility.get_file_parts(pdb_file)[1]
        protein = pd.parsePDB(str(pdb_file))
        if chain is not None:
            protein = protein.select(f"chain {chain}")
        return cls.from_prody_atomgroup(
            pdb_name, protein, split_type, split_size, upsample_rate=upsample_rate
        )

    @classmethod
    def from_pdb_id(
        cls,
        pdb_name: str,
        chain: str = None,
        split_type: str = "kmer",
        split_size=16,
        upsample_rate=50,
    ):
        """
        Construct MomentInvariants instance from a PDB ID and optional chain (downloads the PDB file from RCSB).
        Selects alpha carbons only.
        """
        if chain:
            protein = pd.parsePDB(pdb_name, chain=chain)
        else:
            protein = pd.parsePDB(pdb_name)
        return cls.from_prody_atomgroup(
            pdb_name, protein, split_type, split_size, upsample_rate=upsample_rate
        )

    def _kmerize(self):
        split_indices = []
        for i in range(self.length):
            kmer = []
            for j in range(
                max(0, i - self.split_size // 2),
                min(len(self.residue_splits), i + self.split_size // 2),
            ):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _kmerize_cut_ends(self):
        split_indices = []
        overlap = 1
        for i in range(
            self.split_size // 2, self.length - self.split_size // 2, overlap
        ):
            kmer = []
            for j in range(
                max(0, i - self.split_size // 2),
                min(len(self.residue_splits), i + self.split_size // 2),
            ):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _allmerize(self):
        split_indices = []
        for i in range(self.length):
            for split_size in range(10, self.split_size + 1, 10):
                kmer = []
                for j in range(
                    max(0, i - split_size // 2),
                    min(len(self.residue_splits), i + split_size // 2),
                ):
                    kmer += self.residue_splits[j]
                split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _get_moments(self, split_indices):
        moments = np.zeros((len(split_indices), moment_utility.NUM_MOMENTS))
        for i, indices in enumerate(split_indices):
            moments[i] = moment_utility.get_second_order_moments(
                self.coordinates[indices]
            )
        return split_indices, moments

    def _split_radius(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)
        for i in range(self.length):
            kd_tree.search(
                center=self.coordinates[self.residue_splits[i][0]],
                radius=self.split_size,
            )
            split_indices.append(kd_tree.getIndices())
        return self._get_moments(split_indices)

    def _split_radius_upsample(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)

        split_indices_upsample = []
        coordinates_upsample = resample(
            self.coordinates, self.upsample_rate * self.coordinates.shape[0]
        )
        kd_tree_upsample = pd.KDTree(coordinates_upsample)

        for i in range(self.length):
            kd_tree_upsample.search(
                center=self.coordinates[self.residue_splits[i][0]],
                radius=self.split_size,
            )
            split_indices_upsample.append(kd_tree_upsample.getIndices())
            kd_tree.search(
                center=self.coordinates[self.residue_splits[i][0]],
                radius=self.split_size,
            )
            split_indices.append(kd_tree.getIndices())
        moments = np.zeros((len(split_indices), moment_utility.NUM_MOMENTS))
        for i, indices in enumerate(split_indices_upsample):
            moments[i] = moment_utility.get_second_order_moments(
                coordinates_upsample[indices]
            )
        return split_indices, moments


def moments_to_embedding(
    protein_keys: ty.List[str],
    invariants: ty.Dict[str, MomentInvariants],
    resolution: ty.Union[float, np.ndarray],
    shapemer_keys: ty.List[Shapemer] = None,
) -> ty.Tuple[
    ty.Dict[str, Shapemers],
    ty.Dict[Shapemer, ty.List[ty.Tuple[str, int]]],
    np.ndarray,
    ty.List[Shapemer],
]:
    """list of MomentInvariants to
        - shapemers - list of shape-mers per protein
        - shapemer_to_indices - dict of shape-mer: list of (protein_index, residue_index)s in which it occurs
        - embedding - Geometricus embedding matrix
        - shape_keys - list of shape-mers in order of embedding matrix
    """
    protein_to_shapemers: ty.Dict[str, Shapemers] = get_shapes(invariants, resolution)
    shapemer_to_protein_indices = map_shapemers_to_indices(protein_to_shapemers)
    if shapemer_keys is None:
        shapemer_keys = sorted(list(shapemer_to_protein_indices))
    embedding = make_embedding(protein_keys, protein_to_shapemers, shapemer_keys)
    return protein_to_shapemers, shapemer_to_protein_indices, embedding, shapemer_keys


def get_shapes(
    invariants: ty.Dict[str, MomentInvariants],
    resolution: ty.Union[float, np.ndarray] = 2.0,
) -> ty.Dict[str, Shapemers]:
    """
    moment invariants -> log transformation -> multiply by resolution -> floor = shapemers
    """
    proteins_to_shapemers: ty.Dict[str, Shapemers] = dict()
    if isinstance(resolution, np.ndarray):
        assert resolution.shape[0] == moment_utility.NUM_MOMENTS
    for key in invariants:
        proteins_to_shapemers[key] = [
            tuple(x)
            for x in (np.log1p(invariants[key].moments) * resolution).astype(int)
        ]
    return proteins_to_shapemers


def map_shapemers_to_indices(
    protein_to_shapemers: ty.Dict[str, Shapemers]
) -> ty.Dict[Shapemer, ty.List[ty.Tuple[str, int]]]:
    """
    Maps shapemer to (protein_key, residue_index)
    """
    shapemer_to_protein_indices: ty.Dict[
        Shapemer, ty.List[ty.Tuple[str, int]]
    ] = defaultdict(list)
    for key in protein_to_shapemers:
        for j, shapemer in enumerate(protein_to_shapemers[key]):
            shapemer_to_protein_indices[shapemer].append((key, j))
    return shapemer_to_protein_indices


def make_embedding(
    protein_keys: ty.List[str],
    protein_to_shapemers: ty.Dict[str, Shapemers],
    restrict_to: ty.List[Shapemer],
) -> np.ndarray:
    """
    Counts occurrences of each shapemer across proteins
    """
    shape_to_index = dict(zip(restrict_to, range(len(restrict_to))))
    embedding = np.zeros((len(protein_keys), len(restrict_to)))
    for i, key in enumerate(protein_keys):
        for shapemer in protein_to_shapemers[key]:
            if shapemer in shape_to_index:
                embedding[i, shape_to_index[shapemer]] += 1
    return embedding
