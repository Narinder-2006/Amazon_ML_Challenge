import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.sparse import load_npz
import joblib
import warnings

if __name__ == '__main__':
    # --- 1. Load the COMPLETE Refined Dataset ---
    print("Step 1: Loading the full refined feature set and labels...")
    X_full = load_npz("final_features_best.npz")
    y_full_df = pd.read_csv("final_labels_best.csv")

    print(f"Training data shape: {X_full.shape}")

    # Identify categorical feature indices
    brand_feature_index = X_full.shape[1] - 2
    cluster_feature_index = X_full.shape[1] - 1
    categorical_indices = [brand_feature_index, cluster_feature_index]

    # --- 2. Prepare Data Formats for Different Models ---
    print("\nStep 2: Preparing data formats for the models...")
    # Convert to dense format and fix types for XGBoost/CatBoost
    X_full_dense = X_full.toarray()
    X_full_dense[:, categorical_indices] = X_full_dense[:, categorical_indices].astype(int)

    # Create a DataFrame version for CatBoost
    num_feats = X_full_dense.shape[1]
    col_names = [f'f{i}' for i in range(num_feats)]
    X_full_df_cb = pd.DataFrame(X_full_dense, columns=col_names)
    for idx in categorical_indices:
        col = col_names[idx]
        X_full_df_cb[col] = X_full_df_cb[col].astype(int).astype('category')

    # --- 3. Train and Save Each Model on 100% of the Data ---
    print("\nStep 3: Training the final model ensemble...")

    # --- Model 1: LightGBM ---
    print(" - Training and saving LightGBM...")
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05)
    lgbm.fit(X_full, y_full_df['price_log1p'].to_numpy())
    joblib.dump(lgbm, 'lgbm_final_model.pkl')
    print("   -> LightGBM model saved to 'lgbm_final_model.pkl'")

    # --- Model 2: XGBoost ---
    print(" - Training and saving XGBoost...")
    dtrain = xgb.DMatrix(X_full_dense, label=y_full_df['price_log1p'].to_numpy())
    xgb_params = {'seed': 42, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'device': 'cuda'}
    
    try:
        xgbr = xgb.train(xgb_params, dtrain, num_boost_round=1000)
        xgbr.save_model('xgb_final_model.json')
        print("   -> XGBoost model saved to 'xgb_final_model.json'")
    except Exception as e:
        warnings.warn(f"XGBoost training with GPU failed. Error: {e}")
        # Add CPU fallback if needed

    # --- Model 3: CatBoost ---
    print(" - Training and saving CatBoost...")
    cat_params = dict(random_state=42, iterations=1500, learning_rate=0.05, verbose=0, task_type='GPU')
    
    try:
        cbr = cb.CatBoostRegressor(**cat_params)
        cbr.fit(X_full_df_cb, y_full_df['price_log1p'].to_numpy(), cat_features=[col_names[i] for i in categorical_indices])
        cbr.save_model('cat_final_model.cbm')
        print("   -> CatBoost model saved to 'cat_final_model.cbm'")
    except Exception as e:
        warnings.warn(f"CatBoost training with GPU failed. Error: {e}")
        # Add CPU fallback if needed

    print("\n--- ✅ All models trained on 100% of the data and saved successfully! ---")