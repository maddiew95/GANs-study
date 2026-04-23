from sklearn.preprocessing import StandardScaler
import numpy as np

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
def outlier_remove(data, verbose=True):
    labels = data[:, LABEL]
    unique_classes = np.unique(labels)
    keep_mask = np.ones(len(data), dtype=bool)
    log = {}
    
    for kelas in unique_classes:
        class_mask = labels == kelas
        class_features = data[class_mask][:, COLS]
        
        # Per-feature IQR bounds for this class
        Q1 = np.percentile(class_features, 25, axis=0)
        Q3 = np.percentile(class_features, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Row passes if ALL features within bounds
        within = np.all(
            (class_features >= lower) & (class_features <= upper),
            axis=1
        )
        
        # Build a global-length within-bounds array for this class
        class_within_global = np.zeros(len(data), dtype=bool)
        class_within_global[class_mask] = within
        
        # Keep a row if it's not in this class OR it passes bounds
        keep_mask &= (~class_mask | class_within_global)
        
        n_before = int(class_mask.sum())
        n_removed = int(n_before - within.sum())
        log[int(kelas)] = {
            'rows_before': n_before,
            'rows_removed': n_removed,
            'pct_removed': round(100 * n_removed / n_before, 2)
        }
    
    cleaned = data[keep_mask]
    
    if verbose:
        print(f"Total before: {len(data)}")
        print(f"Total after:  {len(cleaned)}")
        print(f"Total removed: {len(data) - len(cleaned)} "
              f"({100 * (len(data) - len(cleaned)) / len(data):.2f}%)")
        print("\nPer-class breakdown:")
        for kelas, stats in log.items():
            print(f"  Class {kelas}: {stats['rows_removed']:>5} / "
                  f"{stats['rows_before']:>5} removed "
                  f"({stats['pct_removed']:.2f}%)")
    return cleaned, log

def preprocess_scenario(X_train, X_val, X_test, verbose=False):
    # Step 1: outlier removal on training only
    X_train, outlier_log = outlier_remove(X_train, verbose=verbose)
    
    # Step 2: Z-score — fit on train, apply to all
    X_train_norm, scaler = zscore(X_train)
    X_val_norm,   _      = zscore(X_val,  scaler=scaler)
    X_test_norm,  _      = zscore(X_test, scaler=scaler)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler, outlier_log