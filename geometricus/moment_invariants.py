import typing as ty
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import prody as pd
from scipy.signal import resample

from typing import List, Union

from geometricus.moment_utility import get_moments_from_coordinates, MomentType
from geometricus.protein_utility import ProteinKey, Structure, group_indices

MOMENT_TYPES = tuple(m.name for m in MomentType)


class SplitType(IntEnum):
    """
    Different approaches to structural fragmentation
    """

    KMER = 1
    """each residue is taken as the center of a kmer of length split_size, ends are included but shorter"""
    RADIUS = 2
    """overlapping spheres of radius split_size"""
    RADIUS_UPSAMPLE = 3
    """overlapping spheres of radius split_size on coordinates upsampled by upsample rate. Only works if there's one atom selected per residue"""
    ALLMER = 4
    """adds kmers of different lengths (split_size - 5 to split_size + 5) to take into account deletions/insertions that don't change the shape"""
    KMER_CUT = 5
    """same as kmer but ends are not included, only fragments of length split_size are kept"""


@dataclass
class SplitInfo:
    """
    Class to store information about structural fragmentation type.
    """
    split_type: SplitType
    split_size: int
    selection: str = "calpha"
    upsample_rate: float = 1.0


SPLIT_INFOS = (SplitInfo(SplitType.RADIUS, 5),
               SplitInfo(SplitType.RADIUS, 10),
               SplitInfo(SplitType.KMER, 8),
               SplitInfo(SplitType.KMER, 16))
RANGE_SPLIT_TYPE = {"5_16": [0, 1000000], "2_10": [8, -8], "5_8": [4, -4], "2_5": [8, -8]}


@dataclass
class MomentInvariants(Structure):
    residue_splits: List[List[int]] = field(repr=False)
    """split a protein into residues using these atom indices, e.g
        [[1, 2], [3, 4]] could represent both alpha and beta carbons being used in a residue.
        Right now only the alpha carbons are used so this contains indices of alpha carbons for each residue if prody is used
        or just indices in a range if coordinates are given directly"""
    original_indices: np.ndarray = field(repr=False)
    """Also alpha indices (if prody) / range (if coordinates)"""
    sequence: str = Union[str, None]
    """Amino acid sequence"""
    split_info: SplitInfo = SplitInfo(SplitType.KMER, 16, "calpha")
    """How to fragment structure, see SplitType's documentation for options"""
    split_indices: Union[List, None] = None
    """Filled with a list of residue indices for each structural fragment"""
    moments: np.ndarray = None
    """Filled with moment invariant values for each structural fragment"""
    moment_types: List[str] = None
    """Names of moments used"""
    calpha_coordinates: ty.Union[np.ndarray, None] = None
    """
    calpha coordinates
    """

    @classmethod
    def from_prody_atomgroup(
            cls, name: ProteinKey, atom_group: pd.AtomGroup,
            split_info: SplitInfo = SplitInfo(SplitType.KMER, 16, "calpha"),
            moment_types: ty.List[str] = MOMENT_TYPES,
    ):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)
        `moment_types` determines which moments are calculated.

        Example
        --------
        >>> invariants = MomentInvariants.from_prody_atomgroup(name, atom_group, split_type=SplitType.RADIUS, moment_types=[MomentType.O_3, MomentType.F, MomentType.phi_7, MomentType.phi_12])
        """
        sequence: str = str(atom_group.select("protein and calpha").getSequence())
        calpha_coordinates: np.ndarray = atom_group.select("protein and calpha").getCoords()
        residue_splits = group_indices(atom_group.select(split_info.selection).getResindices())

        shape = cls(
            name,
            len(residue_splits),
            atom_group.select(split_info.selection).getCoords(),
            residue_splits,
            atom_group.select(split_info.selection).getIndices(),
            sequence=sequence,
            split_info=split_info,
            calpha_coordinates=calpha_coordinates,
            moment_types=moment_types
        )
        shape.length = len(calpha_coordinates)
        shape._split(shape.split_info.split_type)
        return shape

    @classmethod
    def from_pdb_file(
            cls,
            pdb_file: Union[str, Path],
            chain: str = None,
            split_info: SplitInfo = SplitInfo(SplitType.KMER, 16, "calpha"),
            moment_types: List[str] = MOMENT_TYPES,
    ):
        """
        Construct MomentInvariants instance from a PDB file and optional chain.
        Selects according to `selection` string, (default = alpha carbons)

        Example
        --------
        >>> invariants = MomentInvariants.from_pdb_file("5EAU.pdb", chain="A", split_type=SplitType.RADIUS)
        """
        pdb_name = Path(pdb_file).stem
        protein = pd.parsePDB(str(pdb_file))
        if chain is not None:
            protein = protein.select(f"chain {chain}")
            pdb_name = (pdb_name, chain)
        return cls.from_prody_atomgroup(
            pdb_name,
            protein,
            split_info=split_info,
            moment_types=moment_types,
        )

    @classmethod
    def from_coordinates(
            cls,
            name: ProteinKey,
            coordinates: np.ndarray,
            sequence: Union[str, None] = None,
            split_info: SplitInfo = SplitInfo(SplitType.KMER, 16, "calpha"),
            moment_types: List[str] = MOMENT_TYPES,
    ):
        """
        Construct MomentInvariants instance from a set of calpha coordinates.
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
            split_info=split_info,
            moment_types=moment_types,
            calpha_coordinates=coordinates
        )
        shape._split(split_info.split_type)
        return shape

    def _split(self, split_type: SplitType):
        if split_type == SplitType.KMER:
            split_indices, moments = self._kmerize()
        elif split_type == SplitType.RADIUS:
            split_indices, moments = self._split_radius()
        elif split_type == SplitType.KMER_CUT:
            split_indices, moments = self._kmerize_cut_ends()
        elif split_type == SplitType.ALLMER:
            split_indices, moments = self._allmerize()
        elif split_type == SplitType.RADIUS_UPSAMPLE:
            assert all(len(r) == 1 for r in self.residue_splits)
            split_indices, moments = self._split_radius_upsample()
        else:
            raise Exception("split_type must a RADIUS or KMER SplitType object")
        self.split_indices = split_indices
        self.moments = moments

    def _kmerize(self):
        split_indices = []
        for i in range(self.length):
            kmer = []
            for j in range(
                    max(0, i - self.split_info.split_size // 2),
                    min(len(self.residue_splits), i + self.split_info.split_size // 2),
            ):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _kmerize_cut_ends(self):
        split_indices = []
        overlap = 1
        for i in range(
                self.split_info.split_size // 2, self.length - self.split_info.split_size // 2, overlap
        ):
            kmer = []
            for j in range(
                    max(0, i - self.split_info.split_size // 2),
                    min(len(self.residue_splits), i + self.split_info.split_size // 2),
            ):
                kmer += self.residue_splits[j]
            split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _allmerize(self):
        split_indices = []
        for i in range(self.length):
            for split_size in range(10, self.split_info.split_size + 1, 10):
                kmer = []
                for j in range(
                        max(0, i - split_size // 2),
                        min(len(self.residue_splits), i + split_size // 2),
                ):
                    kmer += self.residue_splits[j]
                split_indices.append(kmer)
        return self._get_moments(split_indices)

    def _split_radius(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)
        for i in range(self.length):
            kd_tree.search(
                center=self.calpha_coordinates[i],
                radius=self.split_info.split_size,
            )
            split_indices.append(kd_tree.getIndices())
        return self._get_moments(split_indices)

    def _split_radius_upsample(self):
        split_indices = []
        kd_tree = pd.KDTree(self.coordinates)

        split_indices_upsample = []
        coordinates_upsample = resample(
            self.coordinates, self.split_info.upsample_rate * self.coordinates.shape[0]
        )
        kd_tree_upsample = pd.KDTree(coordinates_upsample)

        for i in range(self.length):
            kd_tree_upsample.search(
                center=self.calpha_coordinates[i],
                radius=self.split_info.split_size,
            )
            split_indices_upsample.append(kd_tree_upsample.getIndices())
            kd_tree.search(
                center=self.calpha_coordinates[i],
                radius=self.split_info.split_size,
            )
            split_indices.append(kd_tree.getIndices())
        moments = np.zeros((len(split_indices), len(self.moment_types)))
        for i, indices in enumerate(split_indices_upsample):
            if indices is None:
                moments[i] = np.NaN
            else:
                moments[i] = get_moments_from_coordinates(
                    coordinates_upsample[indices], [MomentType[m] for m in self.moment_types]
                )
        return split_indices, moments

    def _get_moments(self, split_indices):
        moments = np.zeros((len(split_indices), len(self.moment_types)))
        for i, indices in enumerate(split_indices):
            if indices is None:
                moments[i] = np.NaN
            else:
                moments[i] = get_moments_from_coordinates(
                    self.coordinates[indices], [MomentType[m] for m in self.moment_types]
                )
        return split_indices, moments

    @property
    def normalized_moments(self):
        return (np.sign(self.moments) * np.log1p(np.abs(self.moments))).astype("float32")
