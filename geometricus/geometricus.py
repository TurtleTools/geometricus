from __future__ import annotations
from typing import List, Tuple, Dict, Set, Union, Generator
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import prody as pd
from scipy.signal import resample

from geometricus.protein_utility import ProteinKey, Structure, group_indices
from geometricus.moment_utility import NUM_MOMENTS, get_second_order_moments

Shapemer = Tuple[int, int, int, int]
"""
A tuple of four integers, each derived from one of the four moments
"""
Shapemers = List[Shapemer]
"""
A list of Shapemer types
"""


class SplitType(IntEnum):
    """
    Different approaches to structural fragmentation
    """

    KMER = (1,)
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
class GeometricusEmbedding:
    """
    Class for storing embedding information
    """

    invariants: Dict[ProteinKey, "MomentInvariants"]
    """
    Dictionary mapping protein_keys to MomentInvariant objects
    """
    resolution: Union[float, np.ndarray]
    """
    Multiplier that determines how coarse/fine-grained each shape is. 
    This can be a single number, multiplied to all four moment invariants 
    or a numpy array of four numbers, one for each invariant
    """
    protein_keys: List[ProteinKey]
    """
    List of protein names = rows of the output embedding
    """
    shapemer_keys: Shapemers
    """
    If given uses only these shapemers for the embedding.
    If None, uses all shapemers
    """
    embedding: np.ndarray
    """
    Embedding matrix of size (len(protein_keys), len(self.shapemer_keys))
    """
    shapemer_to_protein_indices: Dict[Shapemer, List[Tuple[ProteinKey, int]]]
    """
    Maps each shapemer to the proteins which have it and to the corresponding residue indices within these proteins
    """
    proteins_to_shapemers: Dict[ProteinKey, Shapemers]
    """
    Maps each protein to a list of shapemers in order of its residues\n\n
    """

    @classmethod
    def from_invariants(
        cls,
        invariants: Union[Generator["MomentInvariants"], List["MomentInvariants"]],
        resolution: Union[float, np.ndarray] = 1.0,
        protein_keys: Union[None, List[ProteinKey]] = None,
        shapemer_keys: Union[None, List[Shapemer]] = None,
    ):
        """
        Make a GeometricusEmbedding object from a list of MomentInvariant objects

        Parameters
        ----------
        invariants
            List of MomentInvariant objects
        resolution
            multiplier that determines how coarse/fine-grained each shape is
            this can be a single number, multiplied to all four moment invariants
            or a numpy array of four numbers, one for each invariant
        protein_keys
            list of protein names = rows of the output embedding.
            if None, takes all keys in `invariants`
        shapemer_keys
            if given uses only these shapemers for the embedding.
            if None, uses all shapemers
        """
        if isinstance(resolution, np.ndarray):
            assert resolution.shape[0] == NUM_MOMENTS
        invariants: Dict[ProteinKey, "MomentInvariants"] = {
            x.name: x for x in invariants
        }
        if protein_keys is None:
            protein_keys: List[ProteinKey] = list(invariants.keys())
        assert all(k in invariants for k in protein_keys)

        proteins_to_shapemers: Dict[ProteinKey, Shapemers] = get_shapes(
            invariants, resolution
        )
        shapemers_to_protein_indices = map_shapemers_to_indices(proteins_to_shapemers)
        if shapemer_keys is None:
            shapemer_keys = sorted(list(shapemers_to_protein_indices))
        embedding = make_embedding(protein_keys, proteins_to_shapemers, shapemer_keys)
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
        invariants: Union[Generator["MomentInvariants"], List["MomentInvariants"]],
        protein_keys: Union[None, List[ProteinKey]] = None,
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

    def map_shapemer_to_residues(
        self, shapemer: Shapemer
    ) -> Dict[ProteinKey, Set[int]]:
        """
        Gets residues within a particular shapemer across all proteins.
        """
        protein_to_shapemer_residues: Dict[ProteinKey, Set[int]] = defaultdict(set)

        for protein_key, residue_index in self.shapemer_to_protein_indices[shapemer]:
            moment_invariants = self.invariants[protein_key]
            shapemer_residues = moment_invariants.split_indices[residue_index]
            for residue in shapemer_residues:
                protein_to_shapemer_residues[protein_key].add(residue)

        return protein_to_shapemer_residues
    
    def plot_shapemers(self, shapemer: Shapemer, subsample_upto: Union[int, None] = None, distance_between: int = 0, opacity = 0.2):
        from plotly import graph_objects as go
        from caretta import superposition_functions, score_functions
        
        sizes, counts = np.unique([len(self.invariants[p].coordinates[self.invariants[p].split_indices[r]]) for p, r in self.shapemer_to_protein_indices[shapemer]], return_counts=True)
        size = sizes[np.argmax(counts)]
        protein_indices = [(p, r) for p, r in self.shapemer_to_protein_indices[shapemer] if len(self.invariants[p].coordinates[self.invariants[p].split_indices[r]]) == size]
        
        if subsample_upto is not None:
            chosen = np.random.choice(range(len(protein_indices)), min(subsample_upto, len(protein_indices)), replace=False)
        else:
            chosen = np.arange(len(protein_indices))
        
        protein_key, residue_index = protein_indices[chosen[0]]
        
        first = self.invariants[protein_key].coordinates[self.invariants[protein_key].split_indices[residue_index]] 
        first -= np.mean(first, axis=0)
        
        data = [go.Scatter3d(x = first[:, 0], y=first[:, 1], z=first[:, 2], 
                             name = f"{protein_key} residue index {residue_index}",
                             mode="lines", line=dict(color="gray", width=3), opacity=opacity)]
        
        average_coords = [np.array(first)]
        for i, x in enumerate(chosen[1:]):
            protein_key, residue_index = protein_indices[x]
            second = self.invariants[protein_key].coordinates[self.invariants[protein_key].split_indices[residue_index]]
            rot, tran = superposition_functions.paired_svd_superpose(first, second)
            second = superposition_functions.apply_rotran(second, rot, tran)
            rmsd = score_functions.get_rmsd(first, second)

            second_reversed = second[::-1]
            rot, tran = superposition_functions.paired_svd_superpose(first, second_reversed)
            second_reversed = superposition_functions.apply_rotran(second_reversed, rot, tran)
            rmsd_reversed = score_functions.get_rmsd(first, second_reversed)

            add = (i+1) * distance_between
            if rmsd < rmsd_reversed:
                data.append(go.Scatter3d(x = second[:, 0], 
                                         y = second[:, 1], 
                                         z = second[:, 2], 
                                         name = f"{protein_key} residue index {residue_index}",
                                         mode="lines", line=dict(color="gray", width=3), opacity=opacity))
                average_coords += [second]
            else:
                data.append(go.Scatter3d(x = second_reversed[:, 0], 
                                         y = second_reversed[:, 1], 
                                         z = second_reversed[:, 2], 
                                         name = f"{protein_key} residue index {residue_index}",
                                         mode="lines", line=dict(color="gray", width=3), opacity=opacity))
                average_coords += [second_reversed]
        average_coords =  np.median(average_coords, axis=0)
        data.append(go.Scatter3d(x = average_coords[:, 0], 
                                 y = average_coords[:, 1], 
                                 z = average_coords[:, 2], 
                                 name = "median shape-mer",
                                 mode="lines", line=dict(color="black", width=5), opacity=1.))
        
        figure = go.Figure(data, layout=dict(scene=dict(xaxis=dict(showbackground=False), 
                                                        yaxis=dict(showbackground=False), 
                                                        zaxis=dict(showbackground=False))))
        figure.update_layout(title=f"{len(protein_indices)} times across {len(set(i[0] for i in protein_indices))} proteins")
        return figure
        
    def plot_shapemer_on_protein(self, shapemer: Shapemer, protein_key: ProteinKey, opacity=0.5):
        from plotly import graph_objects as go

        indices = [r for p, r in self.shapemer_to_protein_indices[shapemer] if p == protein_key]
        if len(indices) == 0:
            print(f"Shapemer {shapemer} not found in protein {protein_key}")
            return None
        coordinates = self.invariants[protein_key].coordinates
        data = [go.Scatter3d(x = coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2], mode="lines", line=dict(color="gray", width=3))]
        for index in indices:
            shapemer_coords = coordinates[self.invariants[protein_key].split_indices[index]]
            data.append(go.Scatter3d(x = shapemer_coords[:, 0], 
                                     y = shapemer_coords[:, 1], 
                                     z = shapemer_coords[:, 2], 
                                     name = f"{protein_key} residue index {index}",
                                     mode="markers", marker=dict(size=7, opacity=opacity)))
        figure = go.Figure(data, layout=dict(scene=dict(xaxis=dict(showbackground=False), 
                                                        yaxis=dict(showbackground=False), 
                                                        zaxis=dict(showbackground=False))))
        figure.update_layout(title=f"{len(indices)} shapemer occurrences in {protein_key}")
        return figure
    
    def plot_shapemer_heatmap_on_protein(self, shapemers: Shapemers, protein_key, cmap="Greys"):
        import matplotlib.colors as mc
        from matplotlib import cm
        from plotly import graph_objects as go
        
        heatmap = np.zeros(self.invariants[protein_key].coordinates.shape[0])
        for shapemer in shapemers:
            indices = [self.invariants[p].split_indices[r] for p, r in self.shapemer_to_protein_indices[shapemer] if p == protein_key]
            for idx in indices:
                if len(idx):
                    heatmap[idx] += 1
        
        norm = mc.Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)       
        colors = [mapper.to_rgba(h) for h in heatmap]
        
        coordinates = self.invariants[protein_key].coordinates
        data = [go.Scatter3d(x = coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                             mode="markers+lines", line=dict(color="gray", width=3), marker=dict(color=colors, size=7, 
                                                                                                 line=dict(width=2,
                                                                                                           color='black')))]
        figure = go.Figure(data, layout=dict(scene=dict(xaxis=dict(showbackground=False), 
                                                        yaxis=dict(showbackground=False), 
                                                        zaxis=dict(showbackground=False))))
        figure.update_layout(title=f"{protein_key}")
        return figure, heatmap

@dataclass(eq=False)
class MomentInvariants(Structure):
    """
    Class for storing moment invariants for a protein.
    Use `from_*` constructors to make an instance of this.
    Subclasses Structure so also has `name`, `length`, and `coordinates` attributes.
    """

    residue_splits: List[List[int]] = field(repr=False)
    """split a protein into residues using these atom indices, e.g
        [[1, 2], [3, 4]] could represent both alpha and beta carbons being used in a residue.
        Right now only the alpha carbons are used so this contains indices of alpha carbons for each residue if prody is used
        or just indices in a range if coordinates are given directly"""
    original_indices: np.ndarray = field(repr=False)
    """Also alpha indices (if prody) / range (if coordinates)"""
    sequence: str = Union[str, None]
    """Amino acid sequence"""
    split_type: SplitType = SplitType.KMER
    """How to fragment structure, see SplitType's documentation for options"""
    split_size: int = 16
    """kmer size or radius (depending on split_type)"""
    upsample_rate: int = 50
    """ignored unless split_type = "radius_upsample"""
    split_indices: Union[List, None] = None
    """Filled with a list of residue indices for each structural fragment"""
    moments: np.ndarray = None
    """Filled with moment invariant values for each structural fragment"""

    @classmethod
    def from_coordinates(
        cls,
        name: ProteinKey,
        coordinates: np.ndarray,
        sequence: Union[str, None] = None,
        split_type: SplitType = SplitType.KMER,
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
        name: ProteinKey,
        protein: pd.AtomGroup,
        split_type: SplitType = SplitType.KMER,
        split_size: int = 16,
        selection: str = "calpha",
        upsample_rate: int = 50,
    ):
        """
        Construct MomentInvariants instance from a ProDy AtomGroup object.
        Selects according to `selection` string, (default = alpha carbons)

        Example
        --------
        >>> invariants = MomentInvariants.from_prody_atomgroup(atom_group, split_type=SplitType.RADIUS)
        """
        protein: pd.AtomGroup = protein.select("protein").select(selection)
        coordinates: np.ndarray = protein.getCoords()
        residue_splits = group_indices(protein.getResindices())
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
        if split_type == SplitType.KMER:
            split_indices, moments = self._kmerize()
        elif split_type == SplitType.ALLMER:
            split_indices, moments = self._allmerize()
        elif split_type == SplitType.RADIUS:
            split_indices, moments = self._split_radius()
        elif split_type == SplitType.RADIUS_UPSAMPLE:
            assert all(len(r) == 1 for r in self.residue_splits)
            split_indices, moments = self._split_radius_upsample()
        elif split_type == SplitType.KMER_CUT:
            split_indices, moments = self._kmerize_cut_ends()
        else:
            raise Exception("split_type must a SplitType object")
        self.split_indices = split_indices
        self.moments = moments

    @classmethod
    def from_pdb_file(
        cls,
        pdb_file: Union[str, Path],
        chain: str = None,
        split_type: SplitType = SplitType.KMER,
        split_size: int = 16,
        selection: str = "calpha",
        upsample_rate: int = 50,
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
            split_type,
            split_size,
            selection=selection,
            upsample_rate=upsample_rate,
        )

    @classmethod
    def from_pdb_id(
        cls,
        pdb_name: str,
        chain: str = None,
        split_type: SplitType = SplitType.KMER,
        split_size: int = 16,
        selection: str = "calpha",
        upsample_rate: int = 50,
    ):
        """
        Construct MomentInvariants instance from a PDB ID and optional chain (downloads the PDB file from RCSB).
        Selects according to `selection` string, (default = alpha carbons)

        Example
        --------
        >>> invariants = MomentInvariants.from_pdb_id("5EAU", chain="A", split_type=SplitType.RADIUS)
        """
        if chain:
            protein = pd.parsePDB(pdb_name, chain=chain)
            pdb_name = (pdb_name, chain)
        else:
            protein = pd.parsePDB(pdb_name)
        return cls.from_prody_atomgroup(
            pdb_name,
            protein,
            split_type,
            split_size,
            selection=selection,
            upsample_rate=upsample_rate,
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
        moments = np.zeros((len(split_indices), NUM_MOMENTS))
        for i, indices in enumerate(split_indices):
            moments[i] = get_second_order_moments(self.coordinates[indices])
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
        moments = np.zeros((len(split_indices), NUM_MOMENTS))
        for i, indices in enumerate(split_indices_upsample):
            moments[i] = get_second_order_moments(coordinates_upsample[indices])
        return split_indices, moments


def get_shapes(
    invariants: Dict[ProteinKey, MomentInvariants],
    resolution: Union[float, np.ndarray] = 2.0,
) -> Dict[ProteinKey, Shapemers]:
    """
    moment invariants -> log transformation -> multiply by resolution -> floor = shapemers
    """
    proteins_to_shapemers: Dict[ProteinKey, Shapemers] = dict()
    if isinstance(resolution, np.ndarray):
        assert resolution.shape[0] == NUM_MOMENTS
    for key in invariants:
        proteins_to_shapemers[key] = [
            tuple(x)
            for x in (np.log1p(invariants[key].moments) * resolution).astype(int)
        ]
    return proteins_to_shapemers


def map_shapemers_to_indices(
    protein_to_shapemers: Dict[ProteinKey, Shapemers]
) -> Dict[Shapemer, List[Tuple[ProteinKey, int]]]:
    """
    Maps shapemer to (protein_key, residue_index)
    """
    shapemer_to_protein_indices: Dict[
        Shapemer, List[Tuple[ProteinKey, int]]
    ] = defaultdict(list)
    for key in protein_to_shapemers:
        for j, shapemer in enumerate(protein_to_shapemers[key]):
            shapemer_to_protein_indices[shapemer].append((key, j))
    return shapemer_to_protein_indices


def make_embedding(
    protein_keys: List[ProteinKey],
    protein_to_shapemers: Dict[ProteinKey, Shapemers],
    restrict_to: List[Shapemer],
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
