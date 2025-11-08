import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, save_npz
from sentence_transformers import SentenceTransformer

# =========================================
# 1. All your helper functions go here
# (smape, extract_ipq, extract_brand, extract_keywords)
# =========================================
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

def extract_ipq(text):
    text = text.lower()
    ipq = 1.0
    pack_match = re.search(r'(?:pack of|set of|of)\s*(\d+)|(\d+)\s*count', text)
    if pack_match:
        ipq = float(pack_match.group(1) or pack_match.group(2))
    return ipq

def extract_brand(text):
    match = re.search(r'Item Name:\s*([^,]+)', text, re.IGNORECASE)
    if match:
        potential_brand = match.group(1).strip()
        words = potential_brand.split()
        if len(words) > 1 and words[1][0].isupper():
            return f"{words[0]} {words[1]}"
        return words[0]
    return "Unknown"

def extract_keywords(df, keywords):
    df_keywords = pd.DataFrame()
    for keyword in keywords:
        col_name = f"is_{keyword.replace('-', '_').replace(' ', '_')}"
        df_keywords[col_name] = df['catalog_content'].str.contains(keyword, case=False, regex=False).astype(int)
    return df_keywords

# =========================================
# 2. Main Execution Block
# =========================================
if __name__ == '__main__':
    # --- Load Data ---
    print("Step 1: Loading datasets...")
    df_train = pd.read_csv("dataset/train.csv")
    df_image_features = pd.read_parquet("vit_image_features.parquet")
    df_train = pd.merge(df_train, df_image_features, on='sample_id', how='left')
    df_train['catalog_content'] = df_train['catalog_content'].fillna('')
    df_train['price_log1p'] = np.log1p(df_train['price'])
    # --- Feature Engineering ---
    print("Step 2: Engineering all features...")
    # (This includes all your text and image feature engineering logic)
    df_train['brand'] = df_train['catalog_content'].apply(extract_brand)
    df_train['ipq'] = df_train['catalog_content'].apply(extract_ipq)
    KEYWORD_LIST = ['organic', 'gluten-free', 'sugar free', 'premium', 'gourmet', 'natural', 'made in usa']
    keyword_features_df = extract_keywords(df_train, KEYWORD_LIST)
    df_train['brand_id'], brand_map = pd.factorize(df_train['brand'])
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    text_embeddings = text_model.encode(df_train['catalog_content'].tolist(), batch_size=128, show_progress_bar=True)

    # --- Combine ALL Features ---
    print("Step 3: Combining all features into a final matrix...")
    vit_feature_cols = [col for col in df_train.columns if col.startswith('vit_emb_')]
    image_features_sparse = csr_matrix(df_train[vit_feature_cols].fillna(0).values)
    text_embeddings_sparse = csr_matrix(text_embeddings)
    keyword_features_sparse = csr_matrix(keyword_features_df.values)
    ipq_features_sparse = csr_matrix(df_train['ipq'].values.reshape(-1, 1))
    brand_features_sparse = csr_matrix(df_train['brand_id'].values.reshape(-1, 1))

    final_features = hstack([
        text_embeddings_sparse,
        image_features_sparse,
        keyword_features_sparse,
        ipq_features_sparse,
        brand_features_sparse
    ])
    
    # --- ✅ Step 4: Save the Final Dataset to Disk ---
    print("\nStep 4: Saving the final dataset...")
    
    # Save the huge sparse matrix in a highly efficient format
    save_npz("final_features.npz", final_features)
    
    # Save the labels and IDs in a small, separate CSV
    df_train[['sample_id', 'price', 'price_log1p']].to_csv("final_labels.csv", index=False)
    
    print("✅ Successfully saved 'final_features.npz' and 'final_labels.csv'")