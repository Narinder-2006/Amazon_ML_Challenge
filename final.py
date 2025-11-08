import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.sparse import load_npz
import joblib

if __name__ == '__main__':
    # --- Load Trained Models and Test Data ---
    print("Step 1: Loading trained models and the final test feature set...")
    lgbm_model = joblib.load('lgbm_final_model.pkl')
    xgb_model = xgb.Booster()
    xgb_model.load_model('xgb_final_model.json')
    cat_model = cb.CatBoostRegressor()
    cat_model.load_model('cat_final_model.cbm')

    X_test = load_npz("final_features_test.npz")
    df_ids = pd.read_csv("final_test_ids.csv")

    # --- Prepare Data Formats for Prediction ---
    # Convert to dense (CatBoost needs DataFrame with categorical dtypes)
    X_test_dense = X_test.toarray()

    # Determine column names: use deterministic f0..fN-1 so they match training-time names
    num_feats = X_test_dense.shape[1]
    col_names = [f'f{i}' for i in range(num_feats)]

    # Build a DataFrame for CatBoost and LightGBM to preserve feature names and categorical dtypes
    X_test_df = pd.DataFrame(X_test_dense, columns=col_names)

    # Identify categorical columns (last two features by convention) and cast them
    categorical_indices = [num_feats - 2, num_feats - 1]
    cat_cols = [col_names[i] for i in categorical_indices]
    X_test_df[cat_cols] = X_test_df[cat_cols].astype(int).astype('category')

    # Prepare XGBoost DMatrix from numpy values
    dtest = xgb.DMatrix(X_test_df.values)

    # --- Make Predictions with Each Model ---
    print("\nStep 2: Making predictions with the ensemble...")
    # Use DataFrame for LightGBM and CatBoost so feature names and categorical dtypes are preserved
    lgbm_preds_log = lgbm_model.predict(X_test_df)
    xgb_preds_log = xgb_model.predict(dtest)
    cat_preds_log = cat_model.predict(X_test_df)
    
    # --- Ensemble and Create Submission File ---
    print("\nStep 3: Ensembling predictions and creating the submission file...")
    ensemble_preds_log = (lgbm_preds_log + xgb_preds_log + cat_preds_log) / 3.0
    
    # Reverse the log transformation to get the final price
    final_prices = np.expm1(ensemble_preds_log)
    
    # Ensure all prices are positive
    final_prices[final_prices < 0] = 0.01 

    # --- Create the final submission DataFrame ---
    submission_df = pd.DataFrame({
        'sample_id': df_ids['sample_id'],
        'price': final_prices
    })
    
    submission_df.to_csv('submission.csv', index=False)
    
    print("\n--- ✅ SUBMISSION FILE CREATED ---")
    print("Successfully saved predictions to 'submission.csv'")
    print(submission_df.head())