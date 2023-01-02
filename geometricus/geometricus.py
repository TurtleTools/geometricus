from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Union, Generator, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numba as nb
from geometricus.model_utility import ShapemerLearn
from geometricus.moment_invariants import MultipleMomentInvariants, SplitInfo, get_invariants_for_structures
from geometricus.protein_utility import ProteinKey

Shapemer = Union[bytes, tuple]
"""
An integer (in the case of model) or a list of integers for each moment (the old way)
"""
Shapemers = List[Shapemer]
"""
A list of Shapemer types
"""


@dataclass
class Geometricus:
    """
    Class for storing embedding information
    """
    protein_keys: List[ProteinKey]
    """
    List of protein names = rows of the output embedding
    """
    shapemer_to_protein_indices: Dict[Shapemer, List[Tuple[ProteinKey, int]]]
    """
    Maps each shapemer to the proteins which have it and to the corresponding residue indices within these proteins
    """
    proteins_to_shapemers: Dict[ProteinKey, Shapemers]
    """
    Maps each protein to a list of shapemers in order of its residues\n\n
    """
    shapemer_keys: Shapemers
    """
    List of shapemers found
    """
    proteins_to_shapemer_residue_indices: Dict[ProteinKey, Shapemers]
    """
    Maps each protein to a set of residue indices covered by the current residue's shapemer in order of its residues\n\n
    """
    resolution: Union[float, np.ndarray] = None
    """
    Multiplier that determines how coarse/fine-grained each shape is. 
    This can be a single number, multiplied to all four moment invariants 
    or a numpy array of four numbers, one for each invariant
    (This is for the old way of binning shapemers)
    """

    @classmethod
    def from_protein_files(cls,
                           input_files: Union[Path, str, List[str]],
                           model: ShapemerLearn = None,
                           split_infos: List[SplitInfo] = None,
                           moment_types: List[str] = None,
                           resolution: Union[float, np.ndarray] = None,
                           n_threads: int = 1,
                           verbose: bool = True):
        """
        Creates a Geometricus object from protein structure files

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
        model
            trained ShapemerLearn model
            if this is not None, shapemers are generated using the trained model
            and split_infos, moment_types, and resolution is ignored
        split_infos
            List of SplitInfo objects
        moment_types
            List of moment types to use
        resolution
            Multiplier that determines how coarse/fine-grained each shape is.
            This can be a single number, multiplied to all four moment invariants
            or a numpy array of four numbers, one for each invariant
            (This is for the old way of binning shapemers)
        n_threads
            Number of threads to use
        verbose
            Whether to print progress

        Returns
        -------
        Geometricus object
        """
        invariants, errors = get_invariants_for_structures(input_files,
                                                           split_infos=split_infos,
                                                           moment_types=moment_types,
                                                           n_threads=n_threads,
                                                           verbose=verbose)
        return cls.from_invariants(
            invariants,
            model=model, resolution=resolution)

    @classmethod
    def from_invariants(
            cls,
            invariants: Union[Generator[MultipleMomentInvariants], List[MultipleMomentInvariants]],
            protein_keys: Optional[List[ProteinKey]] = None,
            model: Optional[ShapemerLearn] = None,
            resolution: Optional[Union[float, np.ndarray]] = None,
    ):
        """
        Make a GeometricusEmbedding object from a list of MultipleMomentInvariant objects

        Parameters
        ----------
        invariants
            List of MultipleMomentInvariant objects
        protein_keys
            list of protein names = rows of the output embedding.
            if None, takes all keys in `invariants`
        model
            if given, uses this model to make the shapemers
        resolution
            multiplier that determines how coarse/fine-grained each shape is
            this can be a single number, multiplied to all four moment invariants
            or a numpy array of four numbers, one for each invariant
            (This is for the old way of binning shapemers)
        """
        assert model is not None or resolution is not None, "Must provide either a model or resolution"
        if isinstance(resolution, np.ndarray):
            assert resolution.shape[0] == invariants[0].invariants[0].moments.shape[1]
        invariants: Dict[ProteinKey, MultipleMomentInvariants] = {
            x.name: x for x in invariants
        }
        if protein_keys is None:
            protein_keys: List[ProteinKey] = list(invariants.keys())
        assert all(k in invariants for k in protein_keys)
        if model is None:
            proteins_to_shapemers = {k: invariants[k].get_shapemers_binned(resolution) for k in
                                     tqdm(protein_keys, total=len(protein_keys))}
        else:
            proteins_to_shapemers = {k: invariants[k].get_shapemers_model(model) for k in
                                     tqdm(protein_keys, total=len(protein_keys))}

        proteins_to_shapemer_residue_indices = {k: invariants[k].get_neighbors() for k in protein_keys}
        geometricus_class = cls(
            proteins_to_shapemers=proteins_to_shapemers,
            protein_keys=protein_keys,
            resolution=resolution,
            proteins_to_shapemer_residue_indices=proteins_to_shapemer_residue_indices,
            shapemer_keys=[],
            shapemer_to_protein_indices={},
        )
        geometricus_class.shapemer_to_protein_indices = geometricus_class.map_shapemers_to_indices()
        geometricus_class.shapemer_keys = sorted(list(geometricus_class.shapemer_to_protein_indices.keys()))
        return geometricus_class

    def map_shapemers_to_indices(self, protein_keys=None):
        """
        Maps each shapemer to the proteins which have it and to the corresponding residue indices within these proteins
        Maps shapemer to (protein_key, residue_index)
        """
        if protein_keys is None:
            protein_keys = self.protein_keys
        shapemer_to_protein_indices: Dict[
            Shapemer, List[Tuple[ProteinKey, int]]
        ] = defaultdict(list)
        for key in protein_keys:
            for j, shapemer in enumerate(self.proteins_to_shapemers[key]):
                shapemer_to_protein_indices[shapemer].append((key, j))
        return shapemer_to_protein_indices

    def map_protein_to_shapemer_indices(self, protein_keys=None, shapemer_keys=None):
        """
        Maps each protein to a list of shapemer indices where the index corresponds to the shapemer in shapemer_keys
        in order of its residues\n\n
        """
        if protein_keys is not None and shapemer_keys is None:
            shapemer_keys = sorted(list(self.map_shapemers_to_indices(protein_keys).keys()))
        elif protein_keys is None:
            protein_keys = self.protein_keys
            if shapemer_keys is None:
                shapemer_keys = self.shapemer_keys
        shapemer_index = {k: i for i, k in enumerate(shapemer_keys)}
        return {
                   k: np.array([shapemer_index[x] for x in self.proteins_to_shapemers[k] if x in shapemer_index],
                               dtype=int)
                   for
                   k in
                   protein_keys}, shapemer_keys

    def map_shapemer_to_residues(
            self, shapemer: Shapemer
    ) -> Dict[ProteinKey, Set[int]]:
        """
        Gets residue indices within a particular shapemer across all proteins.
        """
        protein_to_shapemer_residues: Dict[ProteinKey, Set[int]] = defaultdict(set)
        for protein_key, residue_index in self.shapemer_to_protein_indices[shapemer]:
            shapemer_residues = self.proteins_to_shapemer_residue_indices[protein_key][residue_index]
            for residue in shapemer_residues:
                protein_to_shapemer_residues[protein_key].add(residue)

        return protein_to_shapemer_residues

    def get_count_matrix(self, protein_keys=None, shapemer_keys=None):
        if protein_keys is None:
            protein_keys = self.protein_keys
        proteins_to_shapemer_indices, shapemer_keys = self.map_protein_to_shapemer_indices(protein_keys, shapemer_keys)
        return make_count_matrix([proteins_to_shapemer_indices[k] for k in protein_keys],
                                 len(shapemer_keys))


@nb.njit(parallel=True)
def make_count_matrix(residues_list, alphabet_size: int):
    out = np.zeros((len(residues_list), alphabet_size))
    for i in nb.prange(len(residues_list)):
        for j in range(len(residues_list[i])):
            out[i, residues_list[i][j]] += 1
    return out
