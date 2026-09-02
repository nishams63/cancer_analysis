import pandas as pd
from scipy import stats

def check_target_leakage(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Analyzes columns for potential target leakage."""
    leakage_candidates = []
    
    # Check all columns except target
    for col in df.columns:
        if col == target:
            continue
            
        status = 'SAFE'
        reason = ''
        
        # Check categorical columns
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Check for high association
            try:
                contingency = pd.crosstab(df[col], df[target])
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                
                # Check for perfect predictors (or near perfect)
                max_pred = contingency.max(axis=1) / contingency.sum(axis=1)
                if (max_pred > 0.95).any():
                    status = 'POTENTIAL LEAKAGE'
                    reason = 'Near perfect predictor for some classes'
                elif p_val < 1e-10:
                    status = 'INVESTIGATE'
                    reason = 'Extremely high association (p < 1e-10)'
            except:
                status = 'UNKNOWN'
                reason = 'Could not compute association'
                
        # Check numerical columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            try:
                groups = [df[col][df[target] == c].dropna() for c in df[target].unique()]
                f_val, p_val = stats.f_oneway(*groups)
                
                if p_val < 1e-10:
                    status = 'INVESTIGATE'
                    reason = 'Extremely high association (p < 1e-10)'
            except:
                status = 'UNKNOWN'
                reason = 'Could not compute ANOVA'
                
        # Heuristic checks based on names
        leakage_keywords = ['outcome', 'grade', 'status', 'response']
        if any(kw in col.lower() for kw in leakage_keywords) and status != 'POTENTIAL LEAKAGE':
            status = 'INVESTIGATE'
            reason += ' | Suspicious column name'
            
        leakage_candidates.append({
            'Feature': col,
            'Status': status,
            'Reason': reason.strip(' | ')
        })
        
    return pd.DataFrame(leakage_candidates)
