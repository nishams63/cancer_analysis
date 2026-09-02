import pandas as pd
import numpy as np
from scipy import stats

def profile_dataset(df: pd.DataFrame) -> dict:
    """Profiles the dataset for basic quality checks."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = [col for col in categorical_cols if 'date' in col.lower()]
    categorical_cols = [c for c in categorical_cols if c not in date_cols]
    
    missing = df.isnull().sum()
    unique = df.nunique()
    
    profile = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'date_cols': date_cols,
        'missing': missing[missing > 0].to_dict(),
        'unique_counts': unique.to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'constant_cols': [col for col in df.columns if unique[col] == 1],
    }
    return profile

def analyze_numerical_features(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Calculates statistical summary for numerical features."""
    summary = df[num_cols].describe().T
    summary['IQR'] = summary['75%'] - summary['25%']
    summary['median'] = df[num_cols].median()
    return summary

def get_feature_associations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Calculates feature associations with the target (leaderboard)."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if target in categorical_cols:
        categorical_cols.remove(target)
    if target in numerical_cols:
        numerical_cols.remove(target)
        
    results = []
    
    # Numerical features vs categorical target (ANOVA F-value)
    for col in numerical_cols:
        groups = [df[col][df[target] == c].dropna() for c in df[target].unique()]
        try:
            f_val, p_val = stats.f_oneway(*groups)
            results.append({'Feature': col, 'Type': 'Numerical', 'Method': 'ANOVA F', 'Score': f_val, 'P-Value': p_val})
        except:
            pass
            
    # Categorical features vs categorical target (Chi-square)
    for col in categorical_cols:
        try:
            contingency = pd.crosstab(df[col], df[target])
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            results.append({'Feature': col, 'Type': 'Categorical', 'Method': 'Chi-Square', 'Score': chi2, 'P-Value': p_val})
        except:
            pass
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('P-Value', ascending=True)
    return res_df

def identify_outliers_iqr(df: pd.DataFrame, num_cols: list) -> dict:
    """Identifies outliers based on IQR."""
    outliers = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if count > 0:
            outliers[col] = {'count': count, 'percentage': count / len(df) * 100}
    return outliers

def analyze_rare_categories(df: pd.DataFrame, cat_cols: list, threshold: float = 0.01) -> dict:
    """Identifies categories with low frequency."""
    rare_cats = {}
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold]
        if not rare.empty:
            rare_cats[col] = rare.to_dict()
    return rare_cats
