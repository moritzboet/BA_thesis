import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_dataset(file_path):
    """
    Loads a CICIDS-2017 CSV file, cleans column spaces, strips out non-numeric
    metadata identifiers, purges text-based infinity fields, and drops invalid rows.
    """
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # 1. Clean header names by stripping out hidden spaces
    df.columns = df.columns.str.strip()
    
    if 'Label' not in df.columns:
        raise ValueError("Critical Error: 'Label' column not found in the dataset.")
        
    # 2. Exclude network metadata identifiers to prevent data leakage (cheating)
    metadata_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    columns_to_drop = [col for col in metadata_cols if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"[-] Dropped identifier columns: {columns_to_drop}")
    
    # 3. Intercept literal text strings of "Infinity" before casting types
    # This prevents pandas from forcing numeric columns into object types
    df.replace(["Infinity", "infinity", "Inf", "-Infinity", "-infinity"], np.nan, inplace=True)
    
    # 4. Force convert all features (except the target Label) into numerical floats
    for col in df.columns:
        if col != 'Label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Catch any remaining mathematical inf entries just in case
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 5. Purge rows containing invalid cells (NaN / Inf)
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    final_rows = df.shape[0]
    
    if initial_rows != final_rows:
        print(f"[!] Purged {initial_rows - final_rows} malformed rows containing Infinity or empty cells.")
        
    return df

def prepare_splits(df, sample_size=None, random_state=42):
    """
    Applies the binary label shortcut, separates features from targets,
    optionally applies stratified sampling, executes an 80/20 split, 
    and handles data-leakage safe scaling.
    """
    df_processed = df.copy()
    
    # 1. Binary Label Shortcut: If it's BENIGN, map to 0. Everything else is an attack (1).
    df_processed['Label'] = df_processed['Label'].astype(str).str.strip().str.upper()
    df_processed['Label'] = df_processed['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Separate features (X) from target ground truth (y)
    X = df_processed.drop(columns=['Label'])
    y = df_processed['Label']
    
    # 2. Strategic Sampling: Keep class ratios identical but shrink total rows 
    # to protect against post-hoc explainers slowing down your computer later.
    if sample_size and sample_size < len(df_processed):
        print(f"[*] Applying strategic downsampling to a subset of {sample_size} rows...")
        X, _, y, _ = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state
        )
        
    # 3. Isolated Train/Test Split (80% Train, 20% Evaluation Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )
    
    print(f"[*] Preprocessing complete. Pipeline breakdown:")
    print(f"    -> Training set size: {X_train.shape[0]} rows")
    print(f"    -> Evaluation test set size: {X_test.shape[0]} rows")
    print(f"    -> Feature metrics tracking: {X_train.shape[1]} individual features")
    
    # 4. Bounded Min-Max Scaling [0, 1] — Completely isolated to training split data
    scaler = MinMaxScaler()
    
    # Fit the mathematics ONLY on training data, then transform both splits
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X_train.columns.tolist()

if __name__ == "__main__":
    print("[+] Preprocessing module compiled successfully.")