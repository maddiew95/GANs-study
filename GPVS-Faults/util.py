from sklearn.preprocessing import StandardScaler
import numpy as np, torch, time

COLS = slice(1, 14)
LABEL = -1

def zscore(data, scaler=None):
    data_norm = data.copy()
    if scaler is None:
        scaler = StandardScaler()
        data_norm[:, COLS] = scaler.fit_transform(data[:, COLS])
    else:
        data_norm[:, COLS] = scaler.transform(data[:, COLS])
    return data_norm, scaler

# Remove rows where any feature value falls outside
# [Q1 - 1.5*IQR, Q3 + 1.5*IQR], with bounds computed per class.
# per Li et al's work
# Applied only class 0 (normal)
def outlier_remove(data, verbose=True):
    labels = data[:, LABEL]
    unique_classes = np.unique(labels)
    keep_mask = np.ones(len(data), dtype=bool)
    log = {}
    
    for kelas in unique_classes:
        class_mask = labels == kelas
        n_before = int(class_mask.sum())
        
        # Outlier removal only applies to F0 (normal operation)
        if kelas != 0:
            log[int(kelas)] = {
                'rows_before': n_before,
                'rows_removed': 0,
                'pct_removed': 0.0,
            }
            continue
        
        class_features = data[class_mask][:, COLS]
        
        Q1 = np.percentile(class_features, 25, axis=0)
        Q3 = np.percentile(class_features, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        within = np.all(
            (class_features >= lower) & (class_features <= upper),
            axis=1
        )
        
        class_within_global = np.zeros(len(data), dtype=bool)
        class_within_global[class_mask] = within
        keep_mask &= (~class_mask | class_within_global)
        
        n_removed = int(n_before - within.sum())
        log[int(kelas)] = {
            'rows_before': n_before,
            'rows_removed': n_removed,
            'pct_removed': round(100 * n_removed / n_before, 2),
        }
    
    cleaned = data[keep_mask]
    
    if verbose:
        n_total_removed = len(data) - len(cleaned)
        pct_total = 100 * n_total_removed / len(data)
        print(f"Outlier removal (F0-only): {n_total_removed}/{len(data)} rows removed ({pct_total:.2f}%)")
        for kelas, stats in log.items():
            print(f"  Class {kelas}: {stats['rows_removed']:>4} / "
                  f"{stats['rows_before']:>5} removed ({stats['pct_removed']:>5.2f}%)")
    
    return cleaned, log

def preprocess_scenario(train, val, test, verbose=False):
    # Step 1: outlier removal on training only
    # train_outlier_removed, outlier_log = outlier_remove(train, verbose=verbose)
    
    # Step 2: Z-score — fit on train, apply to all
    X_train_norm, scaler = zscore(train)
    X_val_norm,   _      = zscore(val,  scaler=scaler)
    X_test_norm,  _      = zscore(test, scaler=scaler)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler

def count_parameters(model):
    """Count trainable parameters. Returns (total, by_type_dict)."""
    total = 0
    by_type = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            kind = name.split(".")[0]        # e.g. 'conv', 'lstm', 'fc'
            by_type[kind] = by_type.get(kind, 0) + n
    return total, by_type


def reset_gpu_peak_memory(device):
    """Call before training to get a clean peak-memory measurement."""
    if "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_peak_memory_mb(device):
    """Returns peak CUDA memory in MB since last reset. 0 if on CPU."""
    if "cuda" in str(device):
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


class Timer:
    """Context manager for wall-clock timing with CUDA sync."""
    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device is not None and "cuda" in str(self.device):
            torch.cuda.synchronize(self.device)
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        if self.device is not None and "cuda" in str(self.device):
            torch.cuda.synchronize(self.device)
        self.elapsed = time.time() - self.t0


def measure_inference_time(predict_fn, X_test, device, n_warmup=5, n_runs=20):
    """
    Measure mean inference time per sample.
    predict_fn: a callable that takes X (tensor) and returns predictions.
    """
    # warmup runs -- first few calls are always slow due to kernel compilation
    for _ in range(n_warmup):
        _ = predict_fn(X_test)

    # actual measurement
    times = []
    for _ in range(n_runs):
        with Timer(device) as t:
            _ = predict_fn(X_test)
        times.append(t.elapsed)

    mean_total = sum(times) / len(times)
    return {
        "total_sec":     mean_total,
        "per_sample_ms": 1000 * mean_total / len(X_test),
        "n_samples":     len(X_test),
    }