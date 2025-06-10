import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, normaltest
from sklearn.cluster import KMeans

# Create output directory
output_dir = os.path.join(os.getcwd(), "output_correlation_filtered")
os.makedirs(output_dir, exist_ok=True)

def process_selected_file(file_path, corr_threshold=0.9, n_clusters=3):
    # Load data
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)

    # Automatically select feature columns (assumes Group and imageName are first two)
    feature_cols = df.columns[2:]
    X = df[feature_cols].select_dtypes(include=[np.number]).dropna(axis=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check global normality
    is_normal = all(normaltest(df[col].dropna())[1] >= 0.05 for col in feature_cols)

    # Choose correlation function
    corr_func = pearsonr if is_normal else spearmanr

    # Compute correlation matrix
    corr_matrix = np.zeros((len(feature_cols), len(feature_cols)))
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            valid_idx = df[[col1, col2]].dropna().index
            if len(valid_idx) < 2:
                corr_matrix[i, j] = 0
                continue
            corr_val, _ = corr_func(df.loc[valid_idx, col1], df.loc[valid_idx, col2])
            corr_matrix[i, j] = corr_val

    corr_df = pd.DataFrame(corr_matrix, columns=feature_cols, index=feature_cols)

    # Reorder features using KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(corr_df.values.T)
    sorted_idx = np.argsort(cluster_labels)
    corr_df = corr_df.iloc[sorted_idx, :].iloc[:, sorted_idx]

    # Identify and remove highly correlated features
    to_remove = set()
    for i in range(len(corr_df.columns)):
        for j in range(i):
            if abs(corr_df.iloc[i, j]) > corr_threshold:
                to_remove.add(corr_df.columns[i])
                break

    selected_features = [col for col in feature_cols if col not in to_remove]
    selected_df = df[['Group', 'imageName'] + selected_features]

    # Save filtered feature matrix
    output_file = f"filtered_features_{os.path.basename(file_path)}"
    selected_df.to_excel(os.path.join(output_dir, output_file), index=False)

    # Plot heatmap of selected features
    try:
        plt.figure(figsize=(12, 12))
        corr_selected = selected_df[selected_features].corr()
        sns.heatmap(corr_selected, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True,
                    cbar_kws={"shrink": 0.8, "aspect": 30})
        plt.title("Correlation Heatmap of Selected Features")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_base = os.path.splitext(os.path.basename(file_path))[0]
        plt.savefig(os.path.join(output_dir, f"heatmap_{heatmap_base}.png"), dpi=500)
        plt.savefig(os.path.join(output_dir, f"heatmap_{heatmap_base}.pdf"), dpi=500)
        plt.close()
    except Exception as e:
        print(f"Heatmap generation failed: {e}")

    print(f"✅ Processed: {file_path}")

# Main batch processor
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data")  # Default to ./data folder
    if not os.path.exists(data_dir):
        print("❌ Data directory not found.")
    else:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if 'train' in file.lower() and file.endswith(('.xlsx', '.csv')):
                    file_path = os.path.join(root, file)
                    process_selected_file(file_path)
