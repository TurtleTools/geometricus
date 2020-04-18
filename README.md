# Geometricus Represents Protein Structures as Shape-mers derived from Moment Invariants

A structure-based, alignment-free embedding approach for proteins. Can be used as input to machine learning algorithms.

TODO: Add link to paper

## Requirements
* `python 3.7+`
* `scipy`
* `numba`
* `prody`

## Example Usage

Embeddings can be made from 
- a list of PDB IDs (`from_pdb_id`)
- a list of PDB files (`from_pdb_file`)
- a list of ProDy AtomGroup objects (`from_prody_atomgroup`)
- a list of numpy arrays containing coordinates (one x, y, z per residue) (`from_coordinates`)

Resolution is a parameter that can be optimized for the task at hand. Higher values result in more specific shape-mers

```python
from geometricus import geometricus

invariants_kmer = {}
invariants_radius = {}
for name in pdb_ids:
    invariants_kmer[name] = geometricus.MomentInvariants.from_pdb_id(name, chain=None, split_type="kmer", split_size=16)
    invariants_radius[name] = geometricus.MomentInvariants.from_pdb_id(name, chain=None, split_type="radius", split_size=10)

embedder = geometricus.GeometricusEmbedding(invariants_kmer, invariants_radius, resolution=2., protein_keys=pdb_ids)
embedding = embedder.embedding
```

For supervised learning scenarios with a train and a test set:

```python
train_embedder = geometricus.GeometricusEmbedding(invariants_kmer, invariants_radius, resolution=2., 
                                                  protein_keys=train_pdb_ids)
train_embedding = train_embedder.embedding

test_embedder = geometricus.GeometricusEmbedding(invariants_kmer, invariants_radius, resolution=2., 
                                                 protein_keys=test_pdb_ids, 
                                                 kmer_shape_keys=train_embedder.kmer_shape_keys,
                                                 radius_shape_keys=train_embedder.radius_shape_keys,
)
test_embedding = test_embedder.embedding
```

To map back from a particular shape-mer index to the residues contained within it across all proteins:

```python
protein_and_residue_indices = train_embedder.map_shapemer_to_residues(shapemer_index)
```

This can be used, for instance, to map predictive shape-mers (using feature importance etc.) back to the residues they represent.