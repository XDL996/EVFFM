import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

def process_file(file_path, output_dir):
    # Load data
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

    # Assumes first column is target, second is ID, features start from column 3
    y = df.iloc[:, 0]
    X = df.iloc[:, 2:].select_dtypes(include=[np.number])  # Avoid non-numeric columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Basic Lasso
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coef_series = pd.Series(lasso.coef_, index=X.columns)
    selected_features = coef_series[coef_series != 0].index.tolist()

    # Save selected coefficients
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    coef_series[coef_series != 0].to_excel(
        os.path.join(output_dir, f"{file_base}_lasso_coefficients.xlsx")
    )

    # Evaluation
    y_pred = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ [{file_base}] MSE (Test): {mse:.4f} | Selected: {len(selected_features)} features")

    # LassoCV: cross-validated selection of alpha
    alphas = np.logspace(-4, 0, 50)
    lasso_cv = LassoCV(alphas=alphas, cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=42), random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)

    # Compute λ_min and λ_1se
    mse_path = lasso_cv.mse_path_.mean(axis=1)
    mse_std = lasso_cv.mse_path_.std(axis=1)
    best_alpha_index = np.argmin(mse_path)
    one_se_index = np.where(mse_path <= mse_path[best_alpha_index] + mse_std[best_alpha_index])[0][0]
    alpha_min = lasso_cv.alphas_[best_alpha_index]
    alpha_1se = lasso_cv.alphas_[one_se_index]

    print(f"   λ_min: {alpha_min:.5f}, λ_1se: {alpha_1se:.5f}")

    # Fit with α_min and α_1se
    selected_min = X.columns[Lasso(alpha=alpha_min).fit(X_train_scaled, y_train).coef_ != 0]
    selected_1se = X.columns[Lasso(alpha=alpha_1se).fit(X_train_scaled, y_train).coef_ != 0]

    # Save selected data matrix
    selected_data = df.iloc[:, [0, 1]].join(df[selected_features])
    selected_data.to_excel(os.path.join(output_dir, f"{file_base}_selected_features.xlsx"), index=False)

    # Save MSE vs Alpha plot
    plt.figure()
    plt.errorbar(lasso_cv.alphas_, mse_path, yerr=mse_std, fmt='o', capsize=2)
    plt.axvline(alpha_min, color='black', linestyle='--', label=r"$\lambda_{min}$")
    plt.axvline(alpha_1se, color='blue', linestyle='--', label=r"$\lambda_{1se}$")
    plt.xscale('log')
    plt.xlabel("Alpha (λ)")
    plt.ylabel("Mean Squared Error")
    plt.title("LassoCV: MSE vs Alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_base}_MSE_vs_Alpha.png"), dpi=300)
    plt.close()

    # Save Lasso path
    coefs = []
    for a in alphas:
        model = Lasso(alpha=a, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        coefs.append(model.coef_)

    plt.figure()
    plt.plot(np.log10(alphas), coefs)
    plt.xlabel("log10(Alpha)")
    plt.ylabel("Coefficients")
    plt.title("Lasso Coefficient Paths")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_base}_Lasso_Paths.png"), dpi=300)
    plt.close()

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if "train" in fname.lower() and fname.endswith(('.csv', '.xlsx')):
            process_file(os.path.join(input_dir, fname), output_dir)

if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "data")      # Replace with your data directory
    output_dir = os.path.join(os.getcwd(), "lasso_output")
    batch_process(input_dir, output_dir)
