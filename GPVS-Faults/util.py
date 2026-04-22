from sklearn.preprocessing import StandardScaler
import numpy as np

cols = slice(1, 14)
label_col = -1

def zscore(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        data_norm[:, cols] = scaler.fit_transform(data[:, cols])
    else:
        data_norm[:, cols] = scaler.transform(data[:, cols])
    return data_norm, scaler

def outlier_remove(data, verbose=True):
    """
    Remove rows where any feature value falls outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR], with bounds computed per class.
    """
    labels = data[:, label_col]
    unique_classes = np.unique(labels)
    keep_mask = np.ones(len(data), dtype=bool)
    log = {}
    
    for cls in unique_classes:
        class_mask = labels == cls
        class_features = data[class_mask][:, cols]
        
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
        log[int(cls)] = {
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
        for cls, stats in log.items():
            print(f"  Class {cls}: {stats['rows_removed']:>5} / "
                  f"{stats['rows_before']:>5} removed "
                  f"({stats['pct_removed']:.2f}%)")
    return cleaned, log

def preprocess_scenario(X_train, X_val, X_test):
    # Outlier removal on training only
    X_train, _ = outlier_remove(X_train)

    # Z-score — fit on train, apply to all
    X_train_norm, scaler = zscore(X_train)
    X_val_norm,   _      = zscore(X_val, scaler=scaler)
    X_test_norm,  _      = zscore(X_test, scaler=scaler)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler