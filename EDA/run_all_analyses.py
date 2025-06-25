import os
import sys
import time
from feature_comparison_analysis import main as run_feature_comparison
from eda_extracted_features import main as run_eda_features
from correlation_analysis import main as run_correlation_analysis

def run_analysis(analysis_func, data_type):
    """
    Run an analysis function with specified data type.
    
    Args:
        analysis_func (function): The analysis function to run
        data_type (str): Type of data to analyze ('full' or 'segments')
    """
    analysis_name = analysis_func.__module__
    print(f"\n{'='*80}")
    print(f"Running {analysis_name} on {data_type} data")
    print(f"{'='*80}")
    
    # Save original command line arguments
    original_argv = sys.argv.copy()
    
    try:
        # Set command line arguments for the analysis
        sys.argv = [analysis_name, f"--data_type={data_type}"]
        
        # Run the analysis
        analysis_func()
        print(f"\nSuccessfully completed {analysis_name} on {data_type} data")
    except Exception as e:
        print(f"\nError running {analysis_name} on {data_type} data: {str(e)}")
    finally:
        # Restore original command line arguments
        sys.argv = original_argv

def main():
    """Run all analyses on both data types"""
    # Define the analysis functions to run
    analyses = [
        run_feature_comparison,
        run_eda_features,
        run_correlation_analysis
    ]
    
    data_types = ["full", "segments"]
    
    # Record start time
    start_time = time.time()
    
    # Run each analysis with each data type
    for analysis in analyses:
        for data_type in data_types:
            run_analysis(analysis, data_type)
    
    # Calculate and display summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Analysis Summary:")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per analysis: {total_time/len(analyses)/len(data_types):.2f} seconds")

if __name__ == "__main__":
    main()