import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load and Merge Data
print("Loading data...")
train_trans = pd.read_csv("data/raw/train_transaction.csv")
train_id = pd.read_csv("data/raw/train_identity.csv")

df = train_trans.merge(train_id, on="TransactionID", how="left")
print("Merged Shape:", df.shape)

# Drop Highly Missing Features
missing_percent = df.isnull().mean() * 100
cols_to_drop = missing_percent[missing_percent > 90].index
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns with >90% missing values.")

# Feature Engineering
df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000)

# Separate Features and Target
target = "isFraud"
X = df.drop(columns=[target, "TransactionID"])
y = df[target]

# Handle Missing Values
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

cat_cols = X.select_dtypes(include=['object', 'str']).columns
X[cat_cols] = X[cat_cols].fillna("Missing")

# Encode Categorical Features
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save Processed Data
os.makedirs("data/processed", exist_ok=True)
X_train.to_parquet("data/processed/X_train.parquet", index=False)
X_test.to_parquet("data/processed/X_test.parquet", index=False)
pd.DataFrame(y_train).to_parquet("data/processed/y_train.parquet", index=False)
pd.DataFrame(y_test).to_parquet("data/processed/y_test.parquet", index=False)

joblib.dump(scaler, "data/processed/scaler.pkl")
feature_metadata = {'num_cols': num_cols.tolist(), 'cat_cols': cat_cols.tolist()}
joblib.dump(feature_metadata, "data/processed/feature_metadata.pkl")

print("Preprocessing complete and parquet files saved.")
