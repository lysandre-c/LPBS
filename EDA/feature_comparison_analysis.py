import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import argparse

def setup_plotting_style():
    """Set up the style for all plots"""
    plt.style.use('ggplot')
    sns.set_theme(font_scale=1.2)
    sns.set_style("whitegrid")

def load_data(data_type):
    """
    Load the data based on the data type.
    
    Args:
        data_type (str): Either 'full' or 'segments'
        
    Returns:
        tuple: (DataFrame, plot directory path)
    """
    if data_type == 'full':
        data_file = 'feature_data/full_features.csv'
        plot_dir = 'EDA/full_data/feature_comparison_plots'
    else:
        data_file = 'feature_data/segments_features.csv'
        plot_dir = 'EDA/segments_data/feature_comparison_plots'
    
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Loading {data_type} data from {data_file}...")
    df = pd.read_csv(data_file)
    return df, plot_dir

def get_numerical_features(df):
    """Get list of numerical features excluding metadata columns"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return [f for f in numerical_features if f not in ['label', 'death_index']]

def create_violin_plots(df, numerical_features, plot_dir):
    """Create violin plots for feature distributions"""
    print("Creating violin plots for feature distributions...")
    label_names = {0: "Healthy", 1: "Unhealthy"}
    df_labeled = df.copy()
    df_labeled['label_name'] = df_labeled['label'].map(label_names)
    
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='label_name', y=feature, data=df_labeled, 
                      inner='quartile', hue='label_name', palette='Set2', legend=False)
        plt.title(f'Distribution of {feature} by Group')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{feature}_violin.png')
        plt.close()

def bin_feature_equal_width(series, bins=5):
    """Bin a feature into equal-width bins"""
    return pd.cut(series, bins=bins)

def bin_feature_equal_freq(series, bins=5):
    """Bin a feature into equal-frequency bins (quantiles)"""
    return pd.qcut(series, q=bins, duplicates='drop')

def create_grouped_bar_plots(df, plot_dir):
    """Create grouped bar plots for selected features"""
    print("Creating grouped bar plots for selected features...")
    selected_features = ['mean_speed', 'mean_turning_angle', 'fraction_efficient_movement', 
                        'fraction_paused', 'mean_roaming_score']
    label_names = {0: "Healthy", 1: "Unhealthy"}
    df_labeled = df.copy()
    df_labeled['label_name'] = df_labeled['label'].map(label_names)
    
    for feature in selected_features:
        # Equal-width binning
        plt.figure(figsize=(12, 6))
        df_labeled[f'{feature}_binned_width'] = bin_feature_equal_width(df_labeled[feature])
        cross_tab_width = pd.crosstab(df_labeled[f'{feature}_binned_width'], 
                                    df_labeled['label_name'], normalize='index')
        cross_tab_width.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
        plt.title(f'Proportion of Labels by {feature} Range (Equal-Width Bins)')
        plt.xlabel(feature)
        plt.ylabel('Proportion')
        plt.legend(title='Label')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{feature}_proportion_equal_width.png')
        plt.close()
        
        # Equal-frequency binning
        plt.figure(figsize=(12, 6))
        try:
            df_labeled[f'{feature}_binned_freq'] = bin_feature_equal_freq(df_labeled[feature])
            cross_tab_freq = pd.crosstab(df_labeled[f'{feature}_binned_freq'], 
                                       df_labeled['label_name'], normalize='index')
            cross_tab_freq.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
            plt.title(f'Proportion of Labels by {feature} Range (Equal-Frequency Bins)')
            plt.xlabel(feature)
            plt.ylabel('Proportion')
            plt.legend(title='Label')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{feature}_proportion_equal_freq.png')
            plt.close()
        except ValueError as e:
            print(f"Could not create equal-frequency bins for {feature}: {e}")
            plt.close()

def perform_pca_analysis(df, numerical_features, plot_dir):
    """Perform PCA analysis and create visualization"""
    print("Performing PCA analysis...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_features])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['label'] = df['label']
    pca_df['label_name'] = pca_df['label'].map({0: "Healthy", 1: "Unhealthy"})
    
    plt.figure(figsize=(10, 8))
    for label, group in pca_df.groupby('label_name'):
        plt.scatter(group['PC1'], group['PC2'], label=label, alpha=0.7)
    plt.title('PCA of Features by Label')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/pca_visualization.png')
    plt.close()

def calculate_feature_importance(df, numerical_features):
    """Calculate feature importance based on group differences"""
    print("Calculating feature importance based on group differences...")
    feature_differences = []
    
    for feature in numerical_features:
        means_by_group = df.groupby('label')[feature].mean().to_dict()
        
        if len(means_by_group) >= 2:
            abs_diff = abs(means_by_group[0] - means_by_group[1])
            mean_value = df[feature].mean()
            relative_diff = abs_diff / mean_value if mean_value != 0 else 0
            
            group0 = df[df['label'] == 0][feature]
            group1 = df[df['label'] == 1][feature]
            
            n1, n2 = len(group0), len(group1)
            s1, s2 = group0.std(), group1.std()
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            
            cohens_d = abs_diff / pooled_std if pooled_std != 0 else 0
            
            feature_differences.append({
                'feature': feature,
                'absolute_difference': abs_diff,
                'relative_difference': relative_diff,
                'cohens_d': cohens_d
            })
    
    diff_df = pd.DataFrame(feature_differences)
    return diff_df.sort_values('cohens_d', ascending=False)

def plot_feature_importance(diff_df, plot_dir):
    """Plot feature importance visualization"""
    plt.figure(figsize=(12, 8))
    top_features_df = diff_df.head(15)
    sns.barplot(x='cohens_d', y='feature', data=top_features_df, 
                hue='feature', palette='viridis', legend=False)
    plt.title("Top Features by Effect Size (Cohen's d)")
    plt.xlabel("Effect Size")
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/top_features_effect_size.png')
    plt.close()

def create_radar_chart(df, diff_df, plot_dir):
    """Create radar chart for comparing top features"""
    print("Creating radar chart for comparing top features...")
    top_features = diff_df.head(8)['feature'].tolist()
    top_features_data = df[top_features + ['label']]
    
    # Normalize features
    normalized_data = top_features_data.copy()
    for feature in top_features:
        min_val = normalized_data[feature].min()
        max_val = normalized_data[feature].max()
        if max_val > min_val:
            normalized_data[feature] = (normalized_data[feature] - min_val) / (max_val - min_val)
        else:
            normalized_data[feature] = 0
    
    means = normalized_data.groupby('label')[top_features].mean()
    categories = top_features
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=12)
    
    for label_value, color in zip([0, 1], ['blue', 'red']):
        values = means.loc[label_value].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                label="Healthy" if label_value == 0 else "Unhealthy", color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Feature Comparison by Group (Normalized Values)', size=15)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/radar_chart.png')
    plt.close()

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Feature comparison analysis for worm movement data')
    parser.add_argument('--data_type', type=str, choices=['full', 'segments'], default='full',
                        help='Type of data to analyze: full or segments')
    args = parser.parse_args()
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load data
    df, plot_dir = load_data(args.data_type)
    
    # Get numerical features
    numerical_features = get_numerical_features(df)
    
    # Create various plots and analyses
    create_violin_plots(df, numerical_features, plot_dir)
    create_grouped_bar_plots(df, plot_dir)
    perform_pca_analysis(df, numerical_features, plot_dir)
    
    # Feature importance analysis
    diff_df = calculate_feature_importance(df, numerical_features)
    print("\nTop features by effect size (Cohen's d):")
    print(diff_df.head(10))
    
    plot_feature_importance(diff_df, plot_dir)
    create_radar_chart(df, diff_df, plot_dir)
    
    print(f"\nAnalysis complete for {args.data_type} data. Results saved to '{plot_dir}' directory.")

if __name__ == "__main__":
    main() 