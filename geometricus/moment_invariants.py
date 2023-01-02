from geometricus import istarmap
from multiprocessing import Pool

from tqdm import tqdm
from time import time

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import prody as pd
from scipy.signal import resample

from geometricus import model_utility
from geometricus.moment_utility import get_moments_from_coordinates, MomentType
from geometricus.protein_utility import ProteinKey, Structure, group_indices, parse_structure_file, get_structure_files

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
    """overlapping spheres of radius split_size on coordinates upsampled by upsample rate. Only works if there's one 
    atom selected per residue """
    ALLMER = 4
    """adds kmers of different lengths (split_size - 5 to split_size + 5) to take into account deletions/insertions 
    that don't change the shape """
    KMER_CUT = 5
    """same as kmer but ends are not included, only fragments of length split_size are kept, rest are nan"""


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
NUM_MOMENTS = len(MomentType) * len(SPLIT_INFOS)


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
    calpha_coordinates: Union[np.ndarray, None] = None
    """
    calpha coordinates
    """

    @classmethod
    def from_prody_atomgroup(
            cls, name: ProteinKey, atom_group: pd.AtomGroup,
            split_info: SplitInfo = SplitInfo(SplitType.KMER, 16, "calpha"),
            moment_types: List[str] = MOMENT_TYPES,
    ):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)
        `moment_types` determines which moments are calculated.

        Example
        --------
        >>> invariants = MomentInvariants.from_prody_atomgroup(name, atom_group,
        >>>                    split_info=SplitInfo(SplitType.RADIUS, 10),
        >>>                    moment_types=["O_3", "O_4", "O_5", "F"])
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
        shape.split(shape.split_info.split_type)
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
        >>> invariants = MomentInvariants.from_pdb_file("5EAU.pdb", chain="A", split_info=SplitInfo(SplitType.RADIUS, 10))
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
        shape.split(split_info.split_type)
        return shape

    def split(self, split_type: SplitType):
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
        split_indices = [[] for _ in range(self.length)]
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
            split_indices[i] = kmer
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


@dataclass
class MultipleMomentInvariants:
    name: ProteinKey
    invariants: List[MomentInvariants]
    sequence: str
    calpha_coordinates: np.ndarray
    length: int

    @classmethod
    def from_prody_atomgroup(
            cls,
            name: ProteinKey,
            atom_group: pd.AtomGroup,
            split_infos=SPLIT_INFOS,
            moment_types=MOMENT_TYPES):
        """
        Construct MultipleMomentInvariants instance from a ProDy AtomGroup object.
        `moment_types` determines which moments are calculated.
        `split_infos` determines which fragmentation methods are used.

        Example
        --------
        >>> invariants_old = MultipleMomentInvariants.from_prody_atomgroup(name, atom_group,
        >>>                     split_infos=[SplitInfo(SplitType.RADIUS, 10)],
        >>>                     moment_types=["O_3", "O_4", "O_5", "F"])
        >>> invariants_new = MultipleMomentInvariants.from_prody_atomgroup(name, atom_group)

        Parameters
        ----------
        name
        atom_group
            ProDy AtomGroup object
        split_infos
            List of SplitInfo objects, only set if not using trained ShapemerLearn model
        moment_types
            List of moment types, only set if not using trained ShapemerLearn model
        Returns
        -------
        MultipleMomentInvariants
        """
        multi: List[MomentInvariants] = []
        sequence: str = str(atom_group.select("protein and calpha").getSequence())
        calpha_coordinates: np.ndarray = atom_group.select("protein and calpha").getCoords()
        for split_info in split_infos:
            residue_splits = group_indices(atom_group.select(split_info.selection).getResindices())

            shape = MomentInvariants(
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
            shape.split(shape.split_info.split_type)
            shape.coordinates = None
            shape.calpha_coordinates = None
            shape.sequence = None
            multi.append(shape)
        return cls(multi[0].name, multi, sequence, calpha_coordinates, len(calpha_coordinates))

    @classmethod
    def from_structure_file(
            cls,
            structure_file: Union[str, Tuple[str, str]],
            split_infos=SPLIT_INFOS,
            moment_types=MOMENT_TYPES
    ):
        """
        Construct MultipleMomentInvariants instance from a structure file

        Parameters
        ----------

        structure_file:  filename or (filename, chain) or PDBID or PDBID_Chain or (PDBID, chain)
        """
        protein = parse_structure_file(structure_file)
        return cls.from_prody_atomgroup(
            protein.getTitle(),
            protein,
            split_infos=split_infos,
            moment_types=moment_types
        )

    @classmethod
    def from_coordinates(
            cls,
            name: ProteinKey,
            coordinates: np.ndarray,
            sequence: Union[str, None] = None,
            split_infos=SPLIT_INFOS,
            moment_types=MOMENT_TYPES
    ):
        """
        Construct MultipleMomentInvariants instance from a set of calpha coordinates.
        Assumes one coordinate per residue.
        """
        multi: List[MomentInvariants] = []
        for split_info in split_infos:
            residue_splits = group_indices(list(np.arange(len(coordinates))))

            shape = MomentInvariants(
                name,
                len(residue_splits),
                coordinates,
                residue_splits,
                np.arange(len(coordinates)),
                sequence=sequence,
                split_info=split_info,
                calpha_coordinates=coordinates,
                moment_types=moment_types
            )
            shape.length = len(coordinates)
            shape.split(shape.split_info.split_type)
            shape.coordinates = None
            shape.calpha_coordinates = None
            shape.sequence = None
            multi.append(shape)
        return cls(multi[0].name, multi, sequence, coordinates, len(coordinates))

    @property
    def moments(self):
        return np.hstack([x.moments for x in self.invariants])

    @property
    def normalized_moments(self):
        moments = np.zeros((self.length, len(self.invariants) * len(MOMENT_TYPES)), dtype=np.float32)
        for i, invariant in enumerate(self.invariants):
            moments[:, i * len(MOMENT_TYPES): (i + 1) * len(MOMENT_TYPES)] = invariant.normalized_moments
        return moments

    @property
    def normalized_moments_size(self):
        moments = np.zeros((self.length, len(self.invariants) * len(MOMENT_TYPES)), dtype=np.float32)
        for i, invariant in enumerate(self.invariants):
            moments[:, i * len(MOMENT_TYPES): (i + 1) * len(
                MOMENT_TYPES)] = invariant.normalized_moments / invariant.split_info.split_size
        return moments

    def get_tensor_model(self, model):
        """
        Get the learned shapemer float representation for each residue in the protein.

        Parameters
        ----------
        model
            Trained ShapemerLearn model

        Returns
        -------
        numpy array of float shapemers, one for each residue
        """
        return model_utility.moments_to_tensors(self.normalized_moments, model)

    def get_shapemers_model(self, model):
        """
        Get learned shapemers for each residue in the protein.

        Parameters
        ----------
        model
            Trained ShapemerLearn model

        Returns
        -------
        list of binary shapemers, one for each residue
        """
        return model_utility.moments_to_shapemers(self.normalized_moments, model)

    def get_shapemers_binned(self, resolution):
        """
        Get binned shapemers for each residue in the protein using the old way of binning raw moment values.

        Parameters
        ----------
        resolution
            Multiplier that determines how coarse/fine-grained each shape is.
            This can be a single number, multiplied to all calculated moment invariants
            or a numpy array of numbers, one for each invariant

        Returns
        -------
        list of integer shapemers, one for each residue
        """
        return [tuple(list(x)) for x in
                (np.nan_to_num(self.invariants[0].normalized_moments) * resolution).astype(np.int64)]

    @property
    def largest_kmer_split(self):
        return max([i for i in self.invariants if i.split_info.split_type in [SplitType.KMER, SplitType.KMER_CUT]],
                   key=lambda x: x.split_info.split_size)

    def get_backbone(self):
        assert len(self.largest_kmer_split.split_indices) == self.length
        return self.largest_kmer_split.split_indices

    def get_neighbors(self):
        """
        Get the neighbors of each residue in the protein used in calculating the shapemer.
        Returns
        -------
        list of lists of indices, one for each residue
        """
        neighbors = [set() for _ in range(self.length)]
        for moment in self.invariants:
            for i, indices in enumerate(moment.split_indices):
                neighbors[i] |= set(indices)
        return neighbors


def get_invariants_for_file(protein_file, split_infos=SPLIT_INFOS, moment_types=MOMENT_TYPES):
    try:
        return MultipleMomentInvariants.from_structure_file(protein_file, split_infos=split_infos,
                                                            moment_types=moment_types)
    except ValueError:
        return None


def get_invariants_for_structures(input_files, n_threads=1, verbose=True,
                                  split_infos: List[SplitInfo] = SPLIT_INFOS,
                                  moment_types: List[str] = MOMENT_TYPES) -> Tuple[List[MultipleMomentInvariants],
                                                                                   List[str]]:
    """
    Get invariants for a list of structures using multiple threads.


    Parameters
    ----------
    input_files
        Can be \n
        A list of structure files (.pdb, .pdb.gz, .cif, .cif.gz),
        A list of (structure_file, chain)
        A list of PDBIDs or PDBID_chain or (PDB ID, chain)
        A folder with input structure files,
        A file which lists structure filenames or "structure_filename, chain" on each line,
        A file which lists PDBIDs or PDBID_chain or PDBID, chain on each line
    n_threads
    verbose
    split_infos
        List of SplitInfo objects, only set if not using trained ShapemerLearn model
    moment_types
        List of moment types, only set if not using trained ShapemerLearn model

    Returns
    -------
    List of MultipleMomentInvariants objects, one for each structure, List of files which threw an error
    """
    if split_infos is None:
        split_infos = SPLIT_INFOS
    if moment_types is None:
        moment_types = MOMENT_TYPES
    start_time = time()
    protein_files = get_structure_files(input_files)
    if verbose:
        print(f"Found {len(protein_files)} protein structures", flush=True)
    invariants = []
    index = 0
    errors = []
    with Pool(n_threads) as pool:
        for i in tqdm(pool.istarmap(get_invariants_for_file,
                                    zip(protein_files,
                                        [split_infos] * len(
                                            protein_files),
                                        [moment_types] * len(
                                            protein_files),
                                        )), total=len(protein_files)):
            if i is not None:
                invariants.append(i)
            else:
                errors.append(protein_files[index])
            index += 1

    if verbose:
        print(f"Computed invariants in {time() - start_time:.2f} seconds")
        if len(errors) > 0:
            print(f"Errors for {len(errors)} proteins: {errors}")
    return invariants, errors
