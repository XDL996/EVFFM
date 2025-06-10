import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, ttest_ind
from sklearn.preprocessing import StandardScaler

# Plotting configuration
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")


def perform_ttest(group0, group1):
    """Perform Levene test + independent two-sample T-test"""
    if len(group0) < 2 or len(group1) < 2:
        return None, None
    try:
        _, p_levene = levene(group0, group1)
    except:
        return None, None

    equal_var = p_levene >= 0.05
    try:
        t_stat, p_val = ttest_ind(group0, group1, equal_var=equal_var)
        return t_stat, p_val
    except:
        return None, None


def process_file(file_path, top_n=50):
    """Main processing function for one file"""
    output_dir = "./results"  # Output directory
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load data
    try:
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        if 'Group' not in df.columns:
            print(f"⚠️ Skipping {file_name}: missing 'Group' column.")
            return
    except Exception as e:
        print(f"❌ Failed to read {file_name}: {e}")
        return

    # Check binary group labels
    groups = df.groupby('Group')
    available_groups = list(groups.groups.keys())
    if 0 not in available_groups or 1 not in available_groups:
        print(f"⚠️ Skipping {file_name}: required Group 0/1 not found. Available: {available_groups}")
        return

    # Select feature columns (assume Group and ID come first)
    feature_cols = df.columns[df.columns.get_loc('Group') + 2:]
    results = []

    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').dropna()
            group0_data = groups.get_group(0)[col].dropna()
            group1_data = groups.get_group(1)[col].dropna()
            t_stat, p_val = perform_ttest(group0_data, group1_data)
            if t_stat is None or p_val is None:
                continue
            mean_std_0 = f"{group0_data.mean():.2f}±{group0_data.std():.2f}"
            mean_std_1 = f"{group1_data.mean():.2f}±{group1_data.std():.2f}"
            results.append([col, mean_std_0, mean_std_1, t_stat, p_val])
        except Exception as e:
            print(f"Error processing feature {col}: {e}")

    # Save t-test results
    result_df = pd.DataFrame(results, columns=['Variable', 'Group0_Mean±Std', 'Group1_Mean±Std', 'T', 'P'])
    result_path = os.path.join(output_dir, f'{file_name}_ttest_results.xlsx')
    result_df.to_excel(result_path, index=False)
    print(f"✅ T-test results saved: {result_path}")

    # Filter significant features
    significant_df = result_df[result_df['P'] < 0.05]
    if significant_df.empty:
        print(f"⚠️ No significant features found in {file_name}.")
        return

    # Save filtered feature matrix
    significant_cols = significant_df['Variable'].tolist()
    required_columns = ['Group', 'imageName']  # Adjust based on actual data
    merged_columns = [col for col in required_columns if col in df.columns]
    try:
        merged_df = df[merged_columns + significant_cols]
    except KeyError as e:
        print(f"❌ Missing required column(s): {e}")
        return

    merged_path = os.path.join(output_dir, f'{file_name}_filtered_features.xlsx')
    merged_df.to_excel(merged_path, index=False)
    print(f"✅ Filtered feature matrix saved: {merged_path}")

    # Heatmap visualization
    try:
        heatmap_data = df[['Group'] + significant_cols].dropna()
        features_only = heatmap_data[significant_cols]

        scaler = StandardScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features_only),
            columns=significant_cols,
            index=heatmap_data.index
        )
        scaled_features['Group'] = heatmap_data['Group'].values
        scaled_features = scaled_features.sort_values(by='Group')

        group_labels = scaled_features['Group']
        scaled_features = scaled_features.drop(columns=['Group'])

        top_features = significant_df.sort_values('P')['Variable'].tolist()[:top_n]
        scaled_top = scaled_features[top_features]

        lut = {0: '#4daf4a', 1: '#e41a1c'}  # green vs red
        col_colors = group_labels.map(lut)
        figsize_width = min(0.5 * len(top_features) + 3, 20)

        cluster_grid = sns.clustermap(
            scaled_top.T,
            cmap='vlag',
            col_colors=col_colors,
            figsize=(figsize_width, 10),
            cbar_kws={'label': 'Z-score'},
            xticklabels=False,
            yticklabels=True,
            col_cluster=False,
            row_cluster=True
        )
        cluster_grid.fig.suptitle(f'Top-{top_n} Significant Features (p < 0.05)', fontsize=14, y=1.03)

        heatmap_path = os.path.join(plot_dir, f"{file_name}_top{top_n}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Heatmap saved: {heatmap_path}")

    except Exception as e:
        print(f"❌ Heatmap generation failed: {e}")


if __name__ == "__main__":
    root_dir = "./input_files"  # Root directory for input files

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if ('train' in file.lower()) and file.endswith(('.xlsx', '.csv')):
                file_path = os.path.join(root, file)
                print(f"\n{'=' * 40}")
                print(f"Processing: {file_path}")

                try:
                    process_file(file_path, top_n=30)
                except Exception as e:
                    print(f"❌ Critical error while processing {file}: {e}")

                print(f"{'=' * 40}\n")
