import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
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
        plot_dir = 'EDA/full_data/correlation_plots'
    else:
        data_file = 'feature_data/segments_features.csv'
        plot_dir = 'EDA/segments_data/correlation_plots'
    
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Loading {data_type} data from {data_file}...")
    df = pd.read_csv(data_file)
    return df, plot_dir

def get_numerical_features(df):
    """Get list of numerical features excluding metadata columns"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return [f for f in numerical_features if f not in ['label', 'death_index']]

def has_variation(series):
    """Return True if the series has variation (not constant)"""
    return series.nunique() > 1 and not np.isclose(series.std(), 0)

def filter_constant_features(df, numerical_features):
    """Filter out constant features from the list"""
    print("Checking for constant features...")
    valid_features = []
    for feature in numerical_features:
        if has_variation(df[feature]):
            valid_features.append(feature)
        else:
            print(f"  Warning: Feature '{feature}' is constant and will be excluded from correlation analysis")
    return valid_features

def calculate_correlations(df, numerical_features):
    """Calculate various correlation metrics for each feature with the label"""
    print(f"Calculating correlations for {len(numerical_features)} valid features...")
    correlations = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
        
        for feature in numerical_features:
            try:
                pearson_corr = df[feature].corr(df['label'])
                spearman_corr = df[feature].corr(df['label'], method='spearman')
                point_biserial = stats.pointbiserialr(df['label'], df[feature])
                
                correlations.append({
                    'feature': feature,
                    'pearson_corr': pearson_corr,
                    'spearman_corr': spearman_corr,
                    'point_biserial_corr': point_biserial.correlation,
                    'point_biserial_pvalue': point_biserial.pvalue
                })
            except Exception as e:
                print(f"  Error calculating correlation for feature '{feature}': {e}")
    
    corr_df = pd.DataFrame(correlations)
    corr_df['abs_pearson'] = corr_df['pearson_corr'].abs()
    return corr_df.sort_values('abs_pearson', ascending=False)

def create_correlation_heatmap(df, corr_df, plot_dir):
    """Create correlation heatmap for top features"""
    plt.figure(figsize=(12, 8))
    top_features = corr_df['feature'].head(10).tolist()
    valid_top_features = [f for f in top_features if has_variation(df[f])]
    correlation_matrix = df[valid_top_features + ['label']].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=True, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix of Top 10 Features with Label')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/top_features_correlation.png')
    plt.close()

def create_correlation_barplot(corr_df, plot_dir):
    """Create bar plot of correlations"""
    plt.figure(figsize=(15, 8))
    sns.barplot(x='feature', y='pearson_corr', data=corr_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Pearson Correlation of Features with Label')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/feature_correlations.png')
    plt.close()

def create_scatter_plots(df, corr_df, plot_dir):
    """Create scatter plots for top 5 features"""
    top_5_features = corr_df['feature'].head(5).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_5_features):
        sns.scatterplot(data=df, x=feature, y='label', ax=axes[i])
        axes[i].set_title(f'{feature}\nCorrelation: {corr_df.loc[corr_df["feature"] == feature, "pearson_corr"].values[0]:.3f}')
        axes[i].set_ylim(-0.1, 1.1)
    
    axes[-1].remove()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/top_features_scatter.png')
    plt.close()

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Correlation analysis for worm movement data')
    parser.add_argument('--data_type', type=str, choices=['full', 'segments'], default='full',
                        help='Type of data to analyze: full or segments')
    args = parser.parse_args()
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load data
    df, plot_dir = load_data(args.data_type)
    
    # Get and filter numerical features
    numerical_features = get_numerical_features(df)
    numerical_features = filter_constant_features(df, numerical_features)
    
    # Calculate correlations
    corr_df = calculate_correlations(df, numerical_features)
    
    # Print correlation results
    print("\n=== Feature Correlations with Label ===")
    print("\nTop 10 features by absolute Pearson correlation:")
    print(corr_df[['feature', 'pearson_corr', 'spearman_corr', 'point_biserial_corr', 'point_biserial_pvalue']].head(10))
    
    # Create visualizations
    create_correlation_heatmap(df, corr_df, plot_dir)
    create_correlation_barplot(corr_df, plot_dir)
    create_scatter_plots(df, corr_df, plot_dir)
    
    print(f"\nCorrelation analysis complete for {args.data_type} data. Results saved to '{plot_dir}' directory.")

if __name__ == "__main__":
    main()