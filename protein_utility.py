import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import prody as pd

from . import utility


@dataclass(eq=False)
class Structure:
    name: str
    length: int
    coordinates: np.ndarray = field(repr=False)


def parse_pdb_files(input_pdb):
    if type(input_pdb) == str or type(input_pdb) == Path:
        input_pdb = Path(input_pdb)
        if input_pdb.is_dir():
            pdb_files = list(input_pdb.glob("*.pdb"))
        elif input_pdb.is_file():
            with open(input_pdb) as f:
                pdb_files = f.read().strip().split('\n')
        else:
            pdb_files = str(input_pdb).split('\n')
    else:
        pdb_files = list(input_pdb)
        if not Path(pdb_files[0]).is_file():
            pdb_files = [pd.fetchPDB(pdb_name) for pdb_name in pdb_files]
    print(f"Found {len(pdb_files)} PDB files")
    return pdb_files


def parse_pdb_files_and_clean(input_pdb: str, output_pdb: typing.Union[str, Path] = "./cleaned_pdb") -> typing.List[typing.Union[str, Path]]:
    if not Path(output_pdb).exists():
        Path(output_pdb).mkdir()
    pdb_files = parse_pdb_files(input_pdb)
    output_pdb_files = []
    for pdb_file in pdb_files:
        pdb = pd.parsePDB(pdb_file).select("protein")
        chains = pdb.getChids()
        if len(chains) and len(chains[0].strip()):
            pdb = pdb.select(f"chain {chains[0]}")
        output_pdb_file = str(Path(output_pdb) / f"{utility.get_file_parts(pdb_file)[1]}.pdb")
        pd.writePDB(output_pdb_file, pdb)
        output_pdb_files.append(output_pdb_file)
    return output_pdb_files


def get_alpha_indices(protein):
    """
    Get indices of alpha carbons of pd AtomGroup object
    """
    return [i for i, a in enumerate(protein.iterAtoms()) if a.getName() == 'CA']


def get_beta_indices(protein: pd.AtomGroup) -> list:
    """
    Get indices of beta carbons of pd AtomGroup object
    (If beta carbon doesn't exist, alpha carbon index is returned)
    """
    residue_splits = utility.group_indices(protein.getResindices())
    i = 0
    indices = []
    for split in residue_splits:
        ca = None
        cb = None
        for _ in split:
            if protein[i].getName() == 'CB':
                cb = protein[i].getIndex()
            if protein[i].getName() == 'CA':
                ca = protein[i].getIndex()
            i += 1
        if cb is not None:
            indices.append(cb)
        else:
            assert ca is not None
            indices.append(ca)
    return indices
