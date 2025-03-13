import argparse
import psutil
import time
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matchms.similarity import BaseEmbeddingSimilarity


def monitor_memory(interval=0.01):
    """Continuously monitor peak memory usage of the process."""
    process = psutil.Process()
    peak_memory = 0
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        peak_memory = max(peak_memory, mem)
        time.sleep(interval)  # Adjust interval for precision
    peak_memory_usage[0] = peak_memory  # Store result


def measure_peak_memory_and_time(func, *args, **kwargs):
    """Measure peak memory usage and execution time during function execution."""
    global stop_event, peak_memory_usage
    peak_memory_usage = [0]
    stop_event = threading.Event()

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    start_time = time.time()
    result = func(*args, **kwargs)  # Run the function
    execution_time = time.time() - start_time

    stop_event.set()
    monitor_thread.join()

    return {
        'peak_memory [MB]': peak_memory_usage[0],
        'execution_time [s]': execution_time
    }


def benchmark(ann_backend: str, dataset_name: str):

    embs_path = f"data/{dataset_name}.npy"
    benchmark_path = f"data/{dataset_name}.benchmark.npy"
    embs_query_path = f"data/MassSpecGym_DreaMS_rand1k.npy"
    res = {"index_backend": ann_backend, "dataset_name": dataset_name}

    emb_sim = BaseEmbeddingSimilarity(similarity="cosine")

    # Benchmark index construction
    res.update(
        measure_peak_memory_and_time(emb_sim.build_ann_index, embeddings_path=embs_path, index_backend=ann_backend)
    )
    
    # Benchmark index search accuracy
    embs = np.load(embs_path)
    embs_query = np.load(embs_query_path)
    benchmark_data = np.load(benchmark_path)
    gt_indices = benchmark_data[:, 0, :].astype(int)
    gt_similarities = benchmark_data[:, 1, :]

    # Search for each query embedding and compute recalls
    k_values = [1, 10]
    recalls = {k: [] for k in k_values}
    query_times = []
    
    for i in tqdm(range(len(gt_indices)), desc="Benchmarking index search accuracy"):
        query_emb = embs_query[i:i+1]

        # Measure query time
        start_time = time.time()
        nn_indices, nn_similarities = emb_sim.get_anns(query_emb, k=max(k_values))
        query_time = time.time() - start_time
        query_times.append(query_time)

        # Calculate recalls at different k
        for k in k_values:
            recall = np.sum(np.isin(nn_indices[:k], gt_indices[i][:k])) / k
            recalls[k].append(recall)

    # Compute mean and std of recalls
    for k in k_values:
        res[f"Recall @ {k} mean"] = np.mean(recalls[k])
        res[f"Recall @ {k} std"] = np.std(recalls[k])

    # Add query time statistics
    res["Query time mean [s]"] = np.mean(query_times)
    res["Query time std [s]"] = np.std(query_times)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark ANN index construction and search.')
    parser.add_argument('--ann_backend', type=str, default='pynndescent',
                      help='Backend to use for approximate nearest neighbors. Should be compatible with matchms.')
    parser.add_argument('--dataset_name', type=str, default='GeMS_A1_DreaMS_rand50k',
                      help='Name of dataset to benchmark. One of GeMS_A1_DreaMS_rand50k, GeMS_A1_DreaMS_rand500k, '
                      'GeMS_A1_DreaMS_rand5M.')
    
    args = parser.parse_args()
    
    res = benchmark(ann_backend=args.ann_backend, dataset_name=args.dataset_name)
    print("\nBenchmark results:")
    for key, value in res.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
            
    # Create CSV filename from backend and dataset
    csv_path = Path("results") / f"{args.ann_backend}_{args.dataset_name}.csv"
    pd.DataFrame([res]).to_csv(csv_path, index=False)
