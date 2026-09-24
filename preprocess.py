import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_dataset(file_path):
    """Loads a CICIDS-2017 CSV file, cleans column spaces, strips out non-numeric

    metadata identifiers, removes empty CSV padding rows, handles infinite flow
    rates, and logs exact dropped row counts.
    """
    print(f"[*] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(
            "Critical Error: 'Label' column not found in the dataset."
        )

    # 1. Purge completely empty padding lines (trailing CSV artifacts)
    initial_raw_rows = len(df)
    empty_rows_mask = df.isna().all(axis=1)
    df = df[~empty_rows_mask].copy()
    purged_empty_rows = initial_raw_rows - len(df)

    # 2. Exclude network metadata identifiers
    metadata_cols = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp",
    ]
    columns_to_drop = [col for col in metadata_cols if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    # 3. Numeric conversion & Infinity handling
    for col in df.columns:
        if col != "Label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Remove invalid rows
    valid_rows_before = len(df)
    df.dropna(inplace=True)
    purged_invalid_rows = valid_rows_before - len(df)

    print(
        f"[*] Dataset Ingestion & Cleaning Audit:\n"
        f"    -> Raw Rows Ingested:          {initial_raw_rows:,}\n"
        f"    -> Empty CSV Padding Purged:   {purged_empty_rows:,}\n"
        f"    -> Valid Network Flows:        {valid_rows_before:,}\n"
        f"    -> Invalid Flows Dropped (Inf/NaN): {purged_invalid_rows:,}\n"
        f"    -> Final Usable Flows:         {len(df):,}"
    )

    return df

def prepare_splits(df, random_state=42):
    """
    Standardized 70/15/15 Train/Validation/Test split protocol matching
    all experimental notebooks.
    
    1. Maps binary labels (0 = BENIGN, 1 = Attack).
    2. Performs stratified two-stage split:
       - 70% Train, 30% Temp
       - 15% Validation, 15% Test (from Temp)
    3. Fits MinMaxScaler strictly on the Training split to prevent leakage.
    """
    df_processed = df.copy()
    
    # 1. Binary Label Standardisation
    df_processed['Label'] = df_processed['Label'].astype(str).str.strip().str.upper()
    df_processed['Label'] = df_processed['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    X = df_processed.drop(columns=['Label'])
    y = df_processed['Label'].values
    
    # 2. Stage A: 70% Train, 30% Temp (Stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=random_state
    )
    
    # 2. Stage B: 15% Validation, 15% Test (Stratified)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )
    
    # 3. Leakage-Free Feature Scaling: Fit strictly on X_train only
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    feature_names = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    
    print(f"[*] Preprocessing complete. Synchronized 70/15/15 pipeline breakdown:")
    print(f"    -> Training set size:   {X_train_scaled.shape[0]} rows ({X_train_scaled.shape[1]} features)")
    print(f"    -> Validation set size: {X_val_scaled.shape[0]} rows")
    print(f"    -> Test set size:       {X_test_scaled.shape[0]} rows")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names

if __name__ == "__main__":
    print("[+] Preprocessing module compiled successfully.")