# 🧠 ML Price Prediction Pipeline (Text + Image + Metadata)

## 📘 Overview
This project implements an **end-to-end Machine Learning pipeline** for **price prediction** using a combination of:
- Product **text descriptions** (`catalog_content`)
- **Visual features** extracted from images (ViT embeddings)
- **Keyword, brand, and metadata features**

The system integrates **text embedding models**, **image features**, and multiple **boosting algorithms** (LightGBM, XGBoost, CatBoost) to build a strong ensemble for price estimation.

---

## 📂 Project Structure

├── dataset.py # Data preprocessing, feature extraction, and dataset preparation
├── deloy.py # Training and saving final models (LightGBM, XGBoost, CatBoost)
├── final.py # Loading trained models and generating ensemble predictions
├── vit_image_features.parquet# Precomputed ViT embeddings (input)
├── final_features.npz # Sparse matrix of engineered features (generated)
├── final_labels.csv # Target variable and sample IDs (generated)
└── submission.csv # Final predictions output (generated)

---

## ⚙️ Workflow

### 🧩 Step 1: Feature Engineering (`dataset.py`)
- Loads raw dataset (`train.csv`) and ViT image embeddings.
- Extracts:
  - **Brand** name from text
  - **Item per quantity (IPQ)** using regex
  - **Keyword indicators** (organic, gluten-free, etc.)
- Generates **SentenceTransformer** text embeddings (`all-MiniLM-L6-v2`).
- Combines text, image, keyword, and metadata into a **sparse feature matrix**.
- Saves:
final_features.npz
final_labels.csv

---

### 🧠 Step 2: Model Training (`deloy.py`)
Trains and saves three models using the full feature set:
- **LightGBM**
- **XGBoost**
- **CatBoost**

Each model is trained on the log-transformed target (`price_log1p`) for numerical stability.

Output models:
lgbm_final_model.pkl
xgb_final_model.json
cat_final_model.cbm

---

### 🚀 Step 3: Inference & Ensemble Prediction (`final.py`)
- Loads trained models and `final_features_test.npz`.
- Makes predictions with all three models.
- Computes ensemble prediction (mean of all model outputs).
- Reverses log transform to get final prices.
- Exports:
submission.csv

---

## 🧮 Evaluation Metric
Uses **SMAPE (Symmetric Mean Absolute Percentage Error)** defined as:

\[
SMAPE = \frac{100}{N}\sum \frac{|y_{pred} - y_{true}|}{(|y_{true}| + |y_{pred}|)/2}
\]

Implementation available in `dataset.py`.

---

## 🧰 Dependencies

Install all required libraries:
```bash
pip install pandas numpy lightgbm xgboost catboost sentence-transformers scipy joblib
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
🧭 Running the Pipeline
Step 1 — Generate Features
python dataset.py

Step 2 — Train Models
python deloy.py

Step 3 — Predict and Generate Submission
python final.py

💡 Highlights

Combines text embeddings + image embeddings + categorical & numeric features

Uses three powerful gradient boosting models for ensemble robustness

Employs log-transform regression for stable target distribution

Modular pipeline — each script can be re-run independently

Highly optimized for both GPU and CPU systems

👤 Author

(Narinder Partap Singh)
B.Tech CSE, NIT Jalandhar
📧 narinderpartapsinghasr@gmail.com

🎯 Passionate about Data Science, Deep Learning & AI Research
