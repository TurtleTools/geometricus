from ftplib import FTP

import torch

from dataclasses import dataclass
import h5py
import numba as nb
import numpy as np
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import prody as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from geometricus import protein_utility
from geometricus.moment_invariants import MultipleMomentInvariants, NUM_MOMENTS
from geometricus.moment_utility import nb_mean_axis_0

POSITIVE_TM_THRESHOLD = 0.8  # only protein pairs with >this TM score considered for positive residue pairs
NEGATIVE_TM_THRESHOLD = 0.6  # only protein pairs with <this TM score considered for negative residue pairs


def get_cath_data(data_folder):
    ftp = FTP("orengoftp.biochem.ucl.ac.uk")
    ftp.login()
    ftp.cwd("cath/releases/latest-release/")
    ftp.nlst()
    ftp.cwd('non-redundant-data-sets')
    filename = 'cath-dataset-nonredundant-S40.pdb.tgz'
    with open(data_folder / filename, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)


@dataclass
class ResiduePair:
    protein_1: str
    protein_2: str
    index_1: int
    index_2: int
    moments_1: np.ndarray
    moments_2: np.ndarray
    aligned: bool
    positive: bool
    rmsd: float

    def __str__(self):
        return "\t".join(
            [self.protein_1, self.protein_2,
             str(self.index_1), str(self.index_2),
             str(int(self.positive)), str(int(self.aligned)),
             str(self.rmsd)])


@nb.njit
def get_backbone_superposed_rmsd_matrix(backbone_coords_1, coords_1, backbone_coords_2, coords_2):
    """
    Superpose based on rotation matrix from backbone coordinates and calculate all vs. all RMSD

    Parameters
    ----------
    backbone_coords_1
    coords_1
    backbone_coords_2
    coords_2

    Returns
    -------
    matrix of RMSD values between coords_1 and coords_2 after superposing based on backbone_coords_1 and backbone_coords_2
    """
    center_1 = nb_mean_axis_0(backbone_coords_1)
    center_2 = nb_mean_axis_0(backbone_coords_2)
    backbone_coords_1 -= center_1
    coords_1 -= center_1
    backbone_coords_2 -= center_2
    coords_2 -= center_2
    coords_2 = np.dot(coords_2, protein_utility.get_rotation_matrix(backbone_coords_1, backbone_coords_2))
    matrix = np.zeros((len(coords_1), len(coords_2)))
    for i1 in range(len(coords_1)):
        for i2 in range(len(coords_2)):
            matrix[i1, i2] = protein_utility.get_rmsd(coords_1[i1], coords_2[i2])
    return matrix


@nb.njit
def get_rmsd_weights(distances, min_weight=0.5):
    """
    Calculate weights for RMSD values based on distances to center residue
    Parameters
    ----------
    distances
        distance of each residue to center residue
    min_weight
        minimum weight
    Returns
    -------
    list of weights, one for each residue in distances
    """
    max_distance, min_distance = np.max(distances), np.min(distances)
    weights = 1 - ((distances - min_distance) / (max_distance - min_distance))
    weights = (min_weight * weights) + min_weight
    return weights


def get_rmsd_neighbors_same_pair(backbone_coords_1, coords_1, backbone_coords_2, coords_2):
    """
    Gets total weighted RMSD between coords_1 and coords_2 based on superposition of backbone_coords_1 and backbone_coords_2
    and linear sum assignment to find best matching residues

    Parameters
    ----------
    backbone_coords_1
    coords_1
    backbone_coords_2
    coords_2

    Returns
    -------
    weighted RMSD
    """
    distances = np.array(
        [protein_utility.get_rmsd(backbone_coords_1[backbone_coords_1.shape[0] // 2], c) for c in coords_1])
    weights = get_rmsd_weights(distances)
    assert backbone_coords_1.shape[0] == backbone_coords_2.shape[0], (
        backbone_coords_1.shape[0], backbone_coords_2.shape[0])
    matrix = get_backbone_superposed_rmsd_matrix(backbone_coords_1, coords_1, backbone_coords_2, coords_2)
    mapping = linear_sum_assignment(matrix)
    rmsd = sum([matrix[i1, i2] * weights[i] for i, (i1, i2) in enumerate(zip(mapping[0], mapping[1]))]) / sum(weights)
    return rmsd


def sample_from_same_protein(protein_moments: MultipleMomentInvariants,
                             num=100,
                             close=2,
                             far_min=5,
                             far_max=20):
    """
    Creates positive and negative ResiduePair objects from the same protein
    Positive pairs are close to each other in sequence (upto `close` away)
    Negative pairs are far from each other in sequence (between `far_min` and `far_max` away)

    Parameters
    ----------
    protein_moments
        MultipleMomentInvariants object for a protein
    num
        number of total pairs to create, randomly assigned to positive or negative
    close
        maximum distance between residues for positive pairs
    far_min
        minimum distance between residues for negative pairs
    far_max
        maximum distance between residues for negative pairs

    Returns
    -------
    generator of ResiduePair objects
    """
    pos_neg_label = np.array([np.random.choice([0, 1]) for _ in range(num)]).astype("float32")
    length = protein_moments.length
    backbone = protein_moments.get_backbone()
    neighbors = [list(n) for n in protein_moments.get_neighbors()]
    start_val = 8
    for i in range(num):
        lists = list(range(start_val + close, length - close - start_val))
        # lists = list(range(close, length - close))
        if len(lists) == 0:
            continue
        chosen_idx = np.random.choice(lists)
        assert len(backbone[chosen_idx]) > 0
        if pos_neg_label[i] == 1:
            close_idx = np.random.choice(list(range(-close, 0)) + list(range(1, close + 1))) + chosen_idx
            assert len(backbone[close_idx]) > 0
            rmsd = get_rmsd_neighbors_same_pair(protein_moments.calpha_coordinates[backbone[chosen_idx]],
                                                protein_moments.calpha_coordinates[neighbors[chosen_idx]],
                                                protein_moments.calpha_coordinates[backbone[close_idx]],
                                                protein_moments.calpha_coordinates[neighbors[close_idx]])
            yield ResiduePair(protein_moments.name, protein_moments.name,
                              chosen_idx, close_idx,
                              protein_moments.normalized_moments[chosen_idx],
                              protein_moments.normalized_moments[close_idx],
                              False, True, rmsd)
        else:
            lists = list(range(max(start_val, chosen_idx - far_max), max(start_val, chosen_idx - far_min))) + list(
                range(min(length - start_val, chosen_idx + far_min), min(length - start_val, chosen_idx + far_max)))
            if len(lists) == 0:
                continue
            far_idx = np.random.choice(lists)
            assert len(backbone[far_idx]) > 0
            rmsd = get_rmsd_neighbors_same_pair(protein_moments.calpha_coordinates[backbone[chosen_idx]],
                                                protein_moments.calpha_coordinates[neighbors[chosen_idx]],
                                                protein_moments.calpha_coordinates[backbone[far_idx]],
                                                protein_moments.calpha_coordinates[neighbors[far_idx]])
            yield ResiduePair(protein_moments.name, protein_moments.name,
                              chosen_idx,
                              far_idx,
                              protein_moments.normalized_moments[chosen_idx],
                              protein_moments.normalized_moments[far_idx],
                              False, False, rmsd)


@nb.njit
def get_rmsd_neighbors_aligned_pair(coords_1, coords_2, center_1, neighbors_1, mapping):
    """
    Gets total weighted RMSD between superposed coords_1 and coords_2 based on alignment mapping

    Parameters
    ----------
    coords_1
    coords_2
    center_1
        index of center residue in coords_1
    neighbors_1
        list of indices of neighbors of center residue in coords_1
    mapping
        maps alignment indices to indices of coords_2

    Returns
    -------
    weighted RMSD
    """
    distances = np.array([protein_utility.get_rmsd(coords_1[center_1], coords_1[n]) for n in neighbors_1])
    weights = get_rmsd_weights(distances)
    rmsd = 0
    for i in range(len(neighbors_1)):
        c = neighbors_1[i]
        if c != -1:
            rmsd += protein_utility.get_rmsd(coords_1[c], coords_2[mapping[c]]) * weights[i]
    return rmsd / sum(weights)


def check_tm_score(filename, is_positive):
    """
    Checks if TM score is within TM_THRESHOLDS for a given fasta file returned by US-align

    Parameters
    ----------
    filename
    is_positive

    Returns
    -------
    True if TM score is within TM_THRESHOLDS, False otherwise
    """
    min_tmscore = 2
    max_tmscore = 0
    for key, _ in protein_utility.get_sequences_from_fasta_yield(filename):
        if key is None:
            return []
        tmscore = float(key.split("\t")[-1].split("=")[-1])
        if tmscore < min_tmscore:
            min_tmscore = tmscore
        if tmscore > max_tmscore:
            max_tmscore = tmscore

    if is_positive and min_tmscore < POSITIVE_TM_THRESHOLD:
        return False
    if not is_positive and max_tmscore > NEGATIVE_TM_THRESHOLD:
        return False
    return True


def superpose_proteins(protein_1_file, protein_2_file, matrix_file):
    """
    Superposes two proteins using US-align-generated rotation, translation matrix

    Parameters
    ----------
    protein_1_file
        PDB file of protein_1
    protein_2_file
        PDB file of protein_2
    matrix_file
        US-align-generated rotation, translation matrix file
    Returns
    -------

    """
    matrix = np.zeros((3, 4))
    with open(matrix_file) as f:
        for i, line in enumerate(f):
            if 1 < i < 5:
                matrix[i - 2] = list(map(float, line.strip().split()[1:]))
    with open(protein_1_file) as f:
        pdb_1 = pd.parsePDBStream(f)
    with open(protein_2_file) as f:
        pdb_2 = pd.parsePDBStream(f)
    transformation = pd.Transformation(matrix[:, 1:], matrix[:, 0])
    pdb_1 = pd.applyTransformation(transformation, pdb_1)
    return pdb_1, pdb_2


def map_residue_indices(aln, protein_1, protein_2, is_positive):
    """
    Select aligning indices and map alignment indices of protein_2

    Parameters
    ----------
    aln
        alignment in dictionary format
    protein_1
        protein_1 name
    protein_2
        protein_2 name
    is_positive
        True if positive pair, False otherwise

    Returns
    -------
    list of matching indices to use between protein_1 and protein_2,
    dictionary mapping alignment indices of protein_2 to indices of protein_2
    """
    aln_np = protein_utility.alignment_to_numpy(aln)
    length_1 = np.where(aln_np[protein_1] != -1)[0].shape[0]
    mapping = np.zeros(length_1, dtype=int)
    mapping[:] = -1
    for i, x in enumerate(aln_np[protein_1]):
        if x == -1:
            continue
        mapping[x] = aln_np[protein_2][i]
    indices = []
    for x in range(len(aln_np[protein_1])):
        a1, a2 = aln_np[protein_1][x], aln_np[protein_2][x]
        if a1 == -1 or (is_positive and a2 == -1):
            continue
        aligned = True
        if a2 == -1 and not is_positive:
            aligned = False
            a2 = aln_np[protein_2][
                np.random.choice([x2 for x2 in range(len(aln_np[protein_2])) if aln_np[protein_2][x2] != -1])]

        assert a1 != -1 and a2 != -1, f"{protein_1} {protein_2} {x} {a1} {a2} {bool(is_positive)} {aligned}"
        indices.append((a1, a2, aligned))
    return indices, mapping


def sample_from_protein_pair(protein_1_moments: MultipleMomentInvariants,
                             protein_2_moments: MultipleMomentInvariants,
                             matrices_folder,
                             pdb_folder,
                             is_positive):
    """
    Creates ResiduePair objects from two aligned proteins

    Parameters
    ----------
    protein_1_moments
        protein_1 MultipleMomentInvariants object
    protein_2_moments
        protein_2 MultipleMomentInvariants object
    matrices_folder
        folder containing US-align-generated alignment fasta files and rotation matrix files
    pdb_folder
        folder containing PDB files
    is_positive
        True if positive pair, False otherwise

    Returns
    -------
    list of ResiduePair objects
    """
    protein_1 = protein_1_moments.name
    protein_2 = protein_2_moments.name
    aln = protein_utility.get_sequences_from_fasta(matrices_folder / f"{protein_1}_{protein_2}.fasta")
    aln = {k.split("\t")[0].split(":")[0].split("/")[-1]: aln[k] for k in aln}
    max_length = min(len(aln[protein_1]), len(aln[protein_2]))
    aln = {k: v[:max_length] for k, v in aln.items()}

    pdb_1, pdb_2 = superpose_proteins(pdb_folder / protein_1, pdb_folder / protein_2,
                                      matrices_folder / f"{protein_1}_{protein_2}")
    coords_1 = pdb_1.select("calpha").getCoords()
    neighbors_1, vector_1 = protein_1_moments.get_neighbors(), protein_1_moments.normalized_moments
    coords_2 = pdb_2.select("calpha").getCoords()
    vector_2 = protein_2_moments.normalized_moments
    indices, mapping = map_residue_indices(aln, protein_1, protein_2, is_positive)
    for a1, a2, aligned in indices:
        assert a1 < len(vector_1) and a2 < len(
            vector_2), f"{protein_1} {protein_2} {a1} {a2} {len(vector_1)} {len(vector_2)}"
        rmsd = get_rmsd_neighbors_aligned_pair(coords_1, coords_2, a1, np.array(list(neighbors_1[a1])), mapping)
        yield ResiduePair(protein_1, protein_2, a1, a2,
                          vector_1[a1], vector_2[a2],
                          aligned, is_positive, rmsd)


def make_training_data_pair(output_folder,
                            protein_moments: dict,
                            id_to_funfam_cluster,
                            matrices_folder, pdb_folder,
                            max_training_points=5e6,
                            output_suffix="_pair", num_moments=NUM_MOMENTS):
    n_pos = 0
    n_total = 0
    with open(output_folder / f"data{output_suffix}.txt", "w") as info_file:
        with h5py.File(output_folder / f"moments{output_suffix}.hdf5", "w") as moment_file:
            moments = moment_file.create_dataset("moments", (max_training_points, num_moments * 2), dtype=np.float32)
            header = ["protein_1", "protein_2",
                      "index_1", "index_2",
                      "label", "aligned", "rmsd"]
            info_file.write("\t".join(header) + "\n")
            for filename in tqdm(matrices_folder.glob("*.fasta"), total=len(protein_moments)):
                protein_1, protein_2 = filename.stem.split("_")
                if protein_1 not in protein_moments or protein_2 not in protein_moments or not (
                        matrices_folder / f"{protein_1}_{protein_2}").exists():
                    continue
                is_positive = id_to_funfam_cluster[protein_1] == id_to_funfam_cluster[protein_2]
                if check_tm_score(filename, is_positive):
                    for residue_pair in sample_from_protein_pair(protein_moments[protein_1],
                                                                 protein_moments[protein_2],
                                                                 matrices_folder,
                                                                 pdb_folder, is_positive):
                        if n_total >= max_training_points:
                            break
                        moments[n_total, :num_moments] = residue_pair.moments_1
                        moments[n_total, num_moments:] = residue_pair.moments_2
                        info_file.write(f"{str(residue_pair)}\n")
                        n_total += 1
                        n_pos += int(residue_pair.positive)
    print(f"Total number of training points: {n_total}\n{n_pos} positive pairs ({n_pos / n_total * 100:.2f}%)")


def make_training_data_self(output_folder, protein_moments, max_from_single=50, max_training_points=3e6,
                            output_suffix="_self", num_moments=NUM_MOMENTS):
    n_pos = 0
    n_total = 0
    with open(output_folder / f"data{output_suffix}.txt", "w") as info_file:
        with h5py.File(output_folder / f"moments{output_suffix}.hdf5", "w") as moment_file:
            moments = moment_file.create_dataset("moments", (max_training_points, num_moments * 2), dtype=np.float32)
            header = ["protein_1", "protein_2",
                      "index_1", "index_2",
                      "label", "aligned", "rmsd"]
            info_file.write("\t".join(header) + "\n")

            for m in tqdm(protein_moments.values(), total=len(protein_moments)):
                for residue_pair in sample_from_same_protein(m, max_from_single):
                    if n_total >= max_training_points:
                        break
                    moments[n_total, :num_moments] = residue_pair.moments_1
                    moments[n_total, num_moments:] = residue_pair.moments_2
                    info_file.write(f"{str(residue_pair)}\n")
                    n_total += 1
                    n_pos += int(residue_pair.positive)
    print(f"Total number of training points: {n_total}\n{n_pos} positive pairs ({n_pos / n_total * 100:.2f}%)")


@dataclass
class Data:
    _indices: np.ndarray
    indices_a: np.ndarray
    indices_b: np.ndarray
    pairs_a: np.ndarray
    pairs_b: np.ndarray
    labels: np.ndarray
    aligned: np.ndarray
    rmsds: np.ndarray
    same_protein: np.ndarray

    @classmethod
    def from_files(cls, folder, suffixes, representation_prefix="moments", representation_length: int = NUM_MOMENTS):
        labels = []
        aligned = []
        rmsds = []
        same_protein = []
        pairs_a = []
        pairs_b = []
        indices_a = []
        indices_b = []
        for suffix in suffixes:
            num_suffix = 0
            with open(folder / f"data{suffix}.txt") as f:
                for i, line in tqdm(enumerate(f)):
                    if i == 0:
                        header = line.strip().split("\t")
                        header_dict = {h: i for i, h in enumerate(header)}
                        continue
                    parts = line.strip().split("\t")
                    indices_a.append((parts[header_dict["protein_1"]], int(parts[header_dict["index_1"]])))
                    indices_b.append((parts[header_dict["protein_2"]], int(parts[header_dict["index_2"]])))
                    labels.append(int(parts[header_dict["label"]]))
                    aligned.append(int(parts[header_dict["aligned"]]))
                    rmsds.append(float(parts[header_dict["rmsd"]]))
                    same_protein.append(int(parts[header_dict["protein_1"]] == parts[header_dict["protein_2"]]))
                    num_suffix += 1
            with h5py.File(folder / f"{representation_prefix}{suffix}.hdf5", "r") as representation_file:
                representation = representation_file[representation_prefix]
                pairs_a.append(representation[:num_suffix, :representation_length])
                pairs_b.append(representation[:num_suffix, representation_length:])
        pairs_a = np.nan_to_num(np.concatenate(pairs_a).astype(np.float32))
        pairs_b = np.nan_to_num(np.concatenate(pairs_b).astype(np.float32))
        labels = np.array(labels)
        aligned = np.array(aligned)
        rmsds = np.array(rmsds)
        same_protein = np.array(same_protein)
        print(f"Loaded {len(labels)} pairs")
        print(f"{np.sum(labels)} positive pairs ({np.sum(labels) / len(labels) * 100:.2f}%)")
        print(f"{np.sum(same_protein)} same protein pairs ({np.sum(same_protein) / len(same_protein) * 100:.2f}%)")
        print(
            f"{np.sum(aligned)} aligned pairs ({np.sum(aligned) / (len(same_protein) - np.sum(same_protein)) * 100:.2f}%)")
        idx = np.arange(len(rmsds))
        np.random.shuffle(idx)
        indices_a = np.array(indices_a)[idx]
        indices_b = np.array(indices_b)[idx]
        return cls(idx, indices_a, indices_b, pairs_a[idx], pairs_b[idx], labels[idx], aligned[idx], rmsds[idx],
                   same_protein[idx])

    def __len__(self):
        return len(self.labels)

    def train_test_split(self, test_size=0.1, rmsd_threshold=8, ignore_first_last=True, protein_lengths=None):
        num_test = int(len(self) * (test_size / 2))
        test_ids_neg = np.random.choice(
            np.where(((self.aligned == 1) & (self.labels == 0) & (self.rmsds <= rmsd_threshold)))[0],
            num_test)
        test_ids_pos = np.random.choice(np.where((self.labels == 1) & (self.rmsds <= rmsd_threshold))[0], num_test)
        test_ids = np.concatenate((test_ids_neg, test_ids_pos))
        test_ban = set(list(test_ids))
        train_ids = np.array([x for x in range(len(self)) if x not in test_ban])
        if ignore_first_last:
            assert protein_lengths is not None
            train_ids = np.array(
                [x for x in train_ids if int(self.indices_a[x][1]) >= 4 and int(self.indices_b[x][1]) >= 4])
            train_ids = np.array([x for x in train_ids if
                                  int(self.indices_a[x][1]) < protein_lengths[self.indices_a[x][0]] - 4 and
                                  int(self.indices_b[x][1]) < protein_lengths[self.indices_b[x][0]] - 4])
        return train_ids, test_ids

    def make_train_batches(self, train_ids, batch_size=1024):
        num_batches = len(train_ids) // batch_size
        train_pairs_a = self.pairs_a[train_ids]
        train_pairs_b = self.pairs_b[train_ids]
        train_labels = self.labels[train_ids].astype(np.float32)
        for i in tqdm(range(num_batches)):
            if (i + 1) * batch_size > len(train_ids):
                break
            batch = (torch.tensor(train_pairs_a[i * batch_size:(i + 1) * batch_size]),
                     torch.tensor(train_pairs_b[i * batch_size:(i + 1) * batch_size]),
                     torch.tensor(train_labels[i * batch_size:(i + 1) * batch_size]))
            if torch.cuda.is_available():
                yield batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            else:
                yield batch

    def make_test(self, test_ids):
        test_pairs_a = self.pairs_a[test_ids]
        test_pairs_b = self.pairs_b[test_ids]
        test_labels = self.labels[test_ids].astype(np.float32)
        if torch.cuda.is_available():
            return (torch.tensor(test_pairs_a).cuda(),
                    torch.tensor(test_pairs_b).cuda(),
                    torch.tensor(test_labels).cuda(),
                    self.rmsds[test_ids])
        else:
            return (
                torch.tensor(test_pairs_a),
                torch.tensor(test_pairs_b),
                torch.tensor(test_labels),
                self.rmsds[test_ids]
            )
