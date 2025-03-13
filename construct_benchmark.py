import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("Starting benchmark construction...")

    print("Sampling 1k random DreaMS embeddings from MassSpecGym dataset")
    # Sample 1k random DreaMS embeddings from the MassSpecGym dataset
    # from dreams.utils.io import sample_hdf
    # sample_hdf(
    #     hdf_pth=Path("data/MassSpecGym_DreaMS.hdf5"),
    #     out_pth=Path("data/MassSpecGym_DreaMS_rand1k.hdf5"),
    #     n_samples=1_000,
    # )

    # Prepare paths to reference and query DreaMS embeddings
    ref_pths = [
        Path("data/GeMS_A1_DreaMS_rand50k.hdf5"),
        Path("data/GeMS_A1_DreaMS_rand500k.hdf5"),
        Path("data/GeMS_A1_DreaMS_rand5M.hdf5"),
    ]
    query_pth = Path("data/MassSpecGym_DreaMS_rand1k.hdf5")

    print("\nConverting HDF5 files to numpy arrays...")
    # Store all embeddings in numpy arrays compatible with matchms EmbeddingBaseSimilarity codebase
    for pth in ref_pths + [query_pth]:
        if pth.with_suffix(".npy").exists():
            print(f"Skipping {pth.with_suffix(".npy")} because it already exists")
            continue
        print(f"Processing {pth}")
        with h5py.File(pth, "r") as f:
            # Store embeddings in numpy array
            embs = f["DreaMS_embedding"][:]
            np.save(pth.with_suffix(".npy"), embs)
            print(f"Saved numpy array with shape {embs.shape}")

    print("\nPre-computing similarities between query and reference embeddings...")
    # Pre-compute similarities between query and reference embeddings
    top_k = 50
    for ref_pth in ref_pths:
        print(f"\nProcessing reference dataset: {ref_pth}")
        # Load embeddings
        query_embs = np.load(query_pth.with_suffix(".npy"))
        ref_embs = np.load(ref_pth.with_suffix(".npy"))
        print(f"Loaded query embeddings: {query_embs.shape}")
        print(f"Loaded reference embeddings: {ref_embs.shape}")

        n = len(query_embs)  # Use all query embeddings

        # Initialize arrays to store results
        top_similarities = np.zeros((n, top_k))
        top_indices = np.zeros((n, top_k), dtype=int)

        # Process one query vector at a time to minimize memory usage
        for i in tqdm(range(n), desc=f"Computing top {top_k} similarities for {n} queries..."):
            # Compute similarities between current query and all reference embeddings
            query = query_embs[i:i+1]  # Keep 2D shape for broadcasting
            similarities = cosine_similarity(ref_embs, query).flatten()
            
            # Get top k similarities and indices
            idx = np.argpartition(similarities, -top_k)[-top_k:]
            idx = idx[np.argsort(-similarities[idx])]  # Sort top k
            
            top_indices[i] = idx
            top_similarities[i] = similarities[idx]

        # Save results as a single matrix
        results = np.stack([top_indices, top_similarities], axis=1)  # Shape: (n, 2, top_k)
        np.save(ref_pth.with_suffix('.benchmark.npy'), results)
        print(f"Saved benchmark results to {ref_pth.with_suffix('.benchmark.npy')}")

        print(f"\nResults for {ref_pth}:")
        print("Shape of query embeddings:", query_embs.shape)
        print("Shape of reference embeddings:", ref_embs.shape) 
        print("Shape of top similarities per vector:", top_similarities.shape)
        print("Shape of top indices per vector:", top_indices.shape)


if __name__ == "__main__":
    main()
