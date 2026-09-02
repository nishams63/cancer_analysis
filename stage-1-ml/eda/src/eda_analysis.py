import json
from pathlib import Path
from src.data_loader import load_data
from src.stats_utils import profile_dataset, analyze_numerical_features, get_feature_associations, identify_outliers_iqr, analyze_rare_categories
from src.visualization import (plot_target_distribution, plot_numerical_distributions, 
                              plot_numerical_boxplots, plot_categorical_analysis, plot_correlation_matrix)
from src.biomarker_analysis import analyze_biomarkers
from src.leakage_analysis import check_target_leakage

def main():
    print("=== Stage 1: Automated EDA ===")
    # 1. Load dataset
    df = load_data()
    target = 'toxicity_risk'
    
    if target not in df.columns:
        print(f"Error: Target variable '{target}' not found in dataset.")
        return
        
    print("\n--- 2. Profiling Dataset ---")
    profile = profile_dataset(df)
    print(f"Numerical columns: {len(profile['numerical_cols'])}")
    print(f"Categorical columns: {len(profile['categorical_cols'])}")
    
    print("\n--- 3. Target Variable Analysis ---")
    plot_target_distribution(df, target)
    print("Generated: toxicity_risk_distribution.png")
    
    print("\n--- 4. Numerical Feature Analysis ---")
    num_cols = [c for c in profile['numerical_cols'] if c != target]
    if num_cols:
        plot_numerical_distributions(df, num_cols)
        plot_numerical_boxplots(df, num_cols, target)
        print("Generated: numerical_distributions.png, numerical_boxplots.png")
        
    print("\n--- 5. Categorical Feature Analysis ---")
    cat_cols = [c for c in profile['categorical_cols'] if c != target]
    if cat_cols:
        plot_categorical_analysis(df, cat_cols, target)
        print("Generated: categorical_analysis.png")
        
    print("\n--- 6. Biomarker Analysis ---")
    analyze_biomarkers(df, target)
    print("Generated biomarker visualizations.")
    
    print("\n--- 7. Feature Associations (Leaderboard) ---")
    associations = get_feature_associations(df, target)
    print(associations.head(10).to_string())
    
    print("\n--- 8. Correlation Analysis ---")
    if len(num_cols) > 1:
        plot_correlation_matrix(df, num_cols)
        print("Generated: correlation_matrix.png")
        
    print("\n--- 9. Leakage Analysis ---")
    leakage_df = check_target_leakage(df, target)
    suspicious = leakage_df[leakage_df['Status'] != 'SAFE']
    if not suspicious.empty:
        print("Potential leakage candidates found:")
        print(suspicious.to_string())
    else:
        print("No obvious leakage found.")
        
    print("EDA completed successfully.")

if __name__ == "__main__":
    main()
