import typing as ty
from dataclasses import dataclass, field

import numpy as np
import prody as pd

# Protein key is either PDB ID or (PDB ID, chain)
ProteinKey = ty.Union[str, (str, str)]


@dataclass(eq=False)
class Structure:
    name: ProteinKey
    length: int
    coordinates: np.ndarray = field(repr=False)


def group_indices(input_list: list) -> list:
    """
    [1, 1, 1, 2, 2, 3, 3, 3, 4] -> [[0, 1, 2], [3, 4], [5, 6, 7], [8]]
    Parameters
    ----------
    input_list

    Returns
    -------
    list of lists
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


def get_alpha_indices(protein: pd.AtomGroup) -> ty.List[int]:
    """
    Get indices of alpha carbons of pd AtomGroup object
    """
    return [i for i, a in enumerate(protein.iterAtoms()) if a.getName() == "CA"]


def get_beta_indices(protein: pd.AtomGroup) -> ty.List[int]:
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
