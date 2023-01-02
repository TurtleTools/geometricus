from __future__ import annotations

from pathlib import PosixPath, Path

import gzip
import io
from dataclasses import dataclass, field
from typing import Union, Tuple, List

import numpy as np
import numba as nb
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import prody as pd

ProteinKey = Union[str, Tuple[str, str]]
"""
A protein key is either its PDB ID (str) or a tuple of (PDB ID, chain)
"""


@dataclass(eq=False)
class Structure:
    """
    Class to store basic protein structure information
    """

    name: ProteinKey
    """PDB ID or (PDB ID, chain)"""
    length: int
    """Number of residues"""
    coordinates: np.ndarray = field(repr=False)
    """Coordinates"""


def parse_structure_file(input_value: Union[Path, (Path, str), str]):
    """
    Parse a protein structure file (.pdb, .pdb.gz, .cif, .cif.gz) or PDBID or PDBID_Chain
    and returns a prody AtomGroup object

    Parameters
    ----------
    input_value: filename or (filename, chain) or PDBID or PDBID_Chain or (PDBID, chain)

    Returns
    -------
    prody AtomGroup object
    """
    chain = None
    if type(input_value) == tuple:
        input_value, chain = input_value
    if not Path(input_value).is_file():
        if "_" in input_value:
            pdb_id, chain = input_value.split("_")
        else:
            pdb_id = input_value
        protein = pd.parsePDB(pdb_id, compressed=False, chain=chain)
        if chain is not None:
            protein.setTitle(f"{pdb_id}_{chain}")
        else:
            protein.setTitle(pdb_id)
    else:
        filename = str(input_value)
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            protein = pd.parsePDB(filename)
            if protein is None:
                with open(filename) as f:
                    protein = pd.parsePDBStream(f)
        elif filename.endswith('.cif'):
            protein = pd.parseMMCIF(filename)
            if protein is None:
                with open(filename) as f:
                    protein = pd.parseMMCIFStream(f)
        elif filename.endswith(".cif.gz"):
            with gzip.open(filename, 'r') as mmcif:
                with io.TextIOWrapper(mmcif, encoding='utf-8') as decoder:
                    protein = pd.parseMMCIFStream(decoder)
        else:
            with open(filename) as f:
                protein = pd.parsePDBStream(f)
        input_value = Path(input_value).name

    if protein is None:
        raise ValueError(f"Could not parse {input_value}")
    if chain is not None:
        protein = protein[chain].toAtomGroup()
        if protein is None:
            raise ValueError(f"Could not parse {input_value} chain {chain}")
        protein.setTitle(f"{input_value}_{chain}")
    else:
        protein.setTitle(input_value)
    return protein


def get_structure_files(input_value: Union[Path, str, List[str]]) -> List[Union[str, (str, str)]]:
    """
    Get a list of structure files or PDB IDs from a string representing:
        A list of structure files (.pdb, .pdb.gz, .cif, .cif.gz),
        A list of (structure_file, chain)
        A list of PDBIDs or PDBID_chain or (PDB ID, chain)
        A folder with input structure files,
        A file which lists structure filenames or "structure_filename, chain" on each line,
        A file which lists PDBIDs or PDBID_chain or PDBID, chain on each line
    Parameters
    ----------
    input_value

    Returns
    -------
    List of structure files or (structure_file, chain) or PDBIDs or (PDB ID, chain)
    """
    if type(input_value) == str or type(input_value) == PosixPath:
        input_value = Path(input_value)
        if input_value.is_dir():
            protein_files = list(input_value.glob("*"))
        elif input_value.is_file():
            with open(input_value) as f:
                protein_files = f.read().strip().split("\n")
        else:
            raise ValueError(f"Could not parse {input_value}")
    else:
        assert type(input_value) == list or type(input_value) == tuple, "Input must be a path or a list"
        protein_files = input_value
    final_protein_files = []
    for protein_file in protein_files:
        if (type(protein_file) == str or type(protein_file) == PosixPath) and Path(protein_file).is_file():
            final_protein_files.append(protein_file)
        elif type(protein_file) == tuple:
            protein_file, chain = protein_file
            final_protein_files.append((protein_file, chain))
        else:
            assert type(protein_file) == str, f"Could not understand input {protein_file}"
            if ", " in protein_file:
                protein_file, chain = protein_file.split(", ")
                final_protein_files.append((protein_file, chain))
            elif "_" in protein_file:
                pdb_id, chain = protein_file.split("_")
                final_protein_files.append((pdb_id, chain))
            else:
                final_protein_files.append(protein_file)
    return list(set(final_protein_files))


def group_indices(input_list: List[int]) -> List[List[int]]:
    """
    e.g [1, 1, 1, 2, 2, 3, 3, 3, 4] -> [[0, 1, 2], [3, 4], [5, 6, 7], [8]]
    """
    output_list = []
    current_list = []
    current_index = None
    for i in range(len(input_list)):
        if current_index is None:
            current_index = input_list[i]
        if input_list[i] == current_index:
            current_list.append(i)
        else:
            output_list.append(current_list)
            current_list = [i]
        current_index = input_list[i]
    output_list.append(current_list)
    return output_list


def get_alpha_indices(protein: pd.AtomGroup) -> List[int]:
    """
    Get indices of alpha carbons of pd AtomGroup object
    """
    return [i for i, a in enumerate(protein.iterAtoms()) if a.getName() == "CA"]


def get_beta_indices(protein: pd.AtomGroup) -> List[int]:
    """
    Get indices of beta carbons of pd AtomGroup object
    (If beta carbon doesn't exist, alpha carbon index is returned)
    """
    residue_splits = group_indices(protein.getResindices())
    i = 0
    indices = []
    for split in residue_splits:
        ca = None
        cb = None
        for _ in split:
            if protein[i].getName() == "CB":
                cb = protein[i].getIndex()
            if protein[i].getName() == "CA":
                ca = protein[i].getIndex()
            i += 1
        if cb is not None:
            indices.append(cb)
        else:
            assert ca is not None
            indices.append(ca)
    return indices


def get_sequences_from_fasta_yield(fasta_file: Union[str, Path], comments=("#")) -> tuple:
    """
    Returns (accession, sequence) iterator
    Parameters
    ----------
    fasta_file
    comments
        ignore lines containing any of these strings
    Returns
    -------
    (accession, sequence)
    """
    with open(fasta_file) as f:
        current_sequence = ""
        current_key = None
        for line in f:
            if not len(line.strip()) or any(comment in line for comment in comments):
                continue
            if ">" in line:
                if current_key is None:
                    current_key = line.split(">")[1].strip()
                else:
                    if current_sequence[-1] == "*":
                        current_sequence = current_sequence[:-1]
                    yield current_key, current_sequence
                    current_sequence = ""
                    current_key = line.split(">")[1].strip()
            else:
                current_sequence += line.strip()
        if current_sequence[-1] == "*":
            current_sequence = current_sequence[:-1]
        yield current_key, current_sequence


def get_sequences_from_fasta(fasta_file: Union[str, Path], comments=("#")) -> dict:
    """
    Returns dict of accession to sequence from fasta file
    Parameters
    ----------
    fasta_file
    comments
        ignore lines containing any of these strings
    Returns
    -------
    {accession:sequence}
    """
    return {
        key: sequence for (key, sequence) in get_sequences_from_fasta_yield(fasta_file, comments=comments)
    }


@nb.njit
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


@nb.njit
def get_rotation_matrix(coords_1: np.ndarray, coords_2: np.ndarray):
    """
    Superpose paired coordinates on each other using Kabsch superposition (SVD)
    Assumes centered coordinates

    Parameters
    ----------
    coords_1
        numpy array of coordinate data for the first protein; shape = (n, 3)
    coords_2
        numpy array of corresponding coordinate data for the second protein; shape = (n, 3)

    Returns
    -------
    rotation matrix for optimal superposition
    """
    correlation_matrix = np.dot(coords_2.T, coords_1)
    u, s, v = np.linalg.svd(correlation_matrix)
    reflect = np.linalg.det(u) * np.linalg.det(v) < 0
    if reflect:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    rotation_matrix = np.dot(u, v)
    return rotation_matrix.astype(np.float64)


def alignment_to_numpy(alignment):
    aln_np = {}
    for n in alignment:
        aln_seq = []
        index = 0
        for a in alignment[n]:
            if a == "-":
                aln_seq.append(-1)
            else:
                aln_seq.append(index)
                index += 1
        aln_np[n] = np.array(aln_seq)
    return aln_np
