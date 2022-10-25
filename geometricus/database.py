#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from geometricus import MomentInvariants, GeometricusEmbedding, SplitInfo
from geometricus.protein_utility import parse_protein_file, ProteinKey
from multiprocessing import Pool
import typing as ty
from time import time
import numba as nb


def get_moments(protein_file, split_info, moment_types):
    protein = parse_protein_file(protein_file)
    name = protein_file.name
    protein.setTitle(name)
    return MomentInvariants.from_prody_atomgroup(name, protein,
                                                 split_info=split_info,
                                                 moment_types=moment_types)


@nb.njit(parallel=True)
def braycurtis(X, Y):
    out = np.zeros((X.shape[0], Y.shape[0]))
    for i in nb.prange(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = np.abs(X[i] - Y[j]).sum() / np.abs(X[i] + Y[j]).sum()
    return out


@dataclass
class Database:
    split_infos: ty.List[SplitInfo]
    moment_types: ty.List[str]
    resolution: ty.List[ty.Union[float, np.ndarray]]
    names: ty.List[ProteinKey]
    embedders: ty.List[GeometricusEmbedding]
    embeddings: np.ndarray
    n_threads: int = 1
    verbose: bool = True

    @classmethod
    def from_folder(cls, input_folder: ty.Union[Path, str],
                    split_infos: ty.List[SplitInfo],
                    moment_types: ty.List[str],
                    resolution: ty.List[ty.Union[float, np.ndarray]],
                    n_threads: int = 1,
                    verbose: bool = True):
        """Create a database of protein structures.

        Args:
            input_folder (str): Path to folder containing protein structures.
            split_infos (List[SplitInfo]): List of split types and split sizes.
            moment_types (List[str]): List of moment types.
            resolution (List[Union[float, np.ndarray]]): List of resolutions.
            n_threads (int): Number of threads to use for multiprocessing.
            verbose (bool): If True, print progress.

        Returns:
            Database: Database of protein structures.
        """
        start_time = time()
        input_folder = Path(input_folder)
        assert input_folder.exists(), f"Folder {input_folder} does not exist"
        protein_files = list(input_folder.glob("*"))
        if verbose:
            print(f"Found {len(protein_files)} protein structures", flush=True)
        current_time = time()
        database = cls(split_infos, moment_types, resolution, [p.name for p in protein_files], [], None, n_threads,
                       verbose)
        with Pool(n_threads) as pool:
            embedders = [GeometricusEmbedding.from_invariants(pool.starmap(get_moments,
                                                                           zip(protein_files,
                                                                               [split_info] * len(protein_files),
                                                                               [moment_types] * len(protein_files))),
                                                              resolution=resolution[i]) for i, split_info in
                         enumerate(split_infos)]
        embeddings = np.hstack([e.embedding for e in embedders])
        database.embedders = embedders
        database.embeddings = embeddings
        if verbose:
            print(f"Created shape-mer count matrix in {time() - current_time:.2f} seconds", flush=True)
            print(f"Number of shape-mers: {embeddings.shape[1]}", flush=True)
            print(f"Total database creation time: {time() - start_time:.2f} seconds", flush=True)
        return database

    def query(self, input_query_folder, output_file):
        """
        Args:
            input_query_folder (str): Path to folder containing protein structures.
            output_file (str): Path to output file.

        Creates output_file with the following columns:
            query_name, target_name, distance
        """
        start_time = time()
        nb.set_num_threads(int(self.n_threads))
        braycurtis(np.random.random((5, 5)), np.random.random((5, 5)))
        input_query_folder = Path(input_query_folder)
        assert input_query_folder.exists(), f"Folder {input_query_folder} does not exist"
        protein_files = list(input_query_folder.glob("*"))
        if self.verbose:
            print(f"Found {len(protein_files)} protein structures", flush=True)
        names = [p.name for p in protein_files]
        current_time = time()
        with Pool(self.n_threads) as pool:
            embedders = [self.embedders[i].embed(pool.starmap(get_moments, zip(protein_files,
                                                                               [split_info] * len(protein_files),
                                                                               [self.moment_types] * len(
                                                                                   protein_files))),
                                                 names) for i, split_info in enumerate(self.split_infos)]
        embeddings = np.hstack([e.embedding for e in embedders])
        if self.verbose:
            print(f"Created shape-mer count matrix in {time() - current_time:.2f} seconds", flush=True)
        current_time = time()
        distances = braycurtis(embeddings, self.embeddings)
        if self.verbose:
            print(f"Computed distances in {time() - current_time:.2f} seconds")
            print(f"Total query time: {time() - start_time:.2f} seconds")
        current_time = time()
        with open(output_file, "w") as f:
            for i, name in enumerate(names):
                for j, distance in enumerate(distances[i]):
                    f.write(f"{name}\t{self.names[j]}\t{distance}\n")
        if self.verbose:
            print(f"Wrote to file: {time() - current_time:.2f} seconds")
        return output_file
