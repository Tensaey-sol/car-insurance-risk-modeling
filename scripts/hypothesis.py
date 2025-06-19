import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests

def prepare_hypothesis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and prepares the dataset for hypothesis testing.
    Adds 'has_claim' and 'margin' columns.
    """
    df = df.copy()
    df = df[(df['TotalPremium'] > 0) & (df['TotalClaims'] >= 0)]
    df['has_claim'] = (df['TotalClaims'] > 0).astype(int)
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    df = df[df['Gender'].isin(['Male', 'Female'])]
    print(f"Filtered Data: {df.shape}")
    return df

def calculate_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculate claim frequency, claim severity, and margin by group.
    """
    metrics = df.groupby(group_col).agg({
        'has_claim': 'mean',
        'TotalClaims': lambda x: x[x > 0].mean() if x[x > 0].count() > 0 else np.nan,
        'margin': 'mean'
    }).rename(columns={
        'has_claim': 'claim_frequency',
        'TotalClaims': 'claim_severity',
        'margin': 'margin'
    })
    return metrics

def chi_squared_test(df: pd.DataFrame, group_col, group_a, group_b):
    df = df[df[group_col].isin([group_a, group_b])]
    contingency = pd.crosstab(df[group_col], df['has_claim'])
    if contingency.shape[1] < 2 or contingency.min().min() < 5:
        return None
    chi2, p, _, _ = chi2_contingency(contingency)
    return {
        'test': 'Chi-Squared', 'metric': 'claim_frequency',
        'group_a': group_a, 'group_b': group_b, 'p_value': p
    }

def t_test(df: pd.DataFrame, group_col, group_a, group_b, metric):
    df = df[df[group_col].isin([group_a, group_b])]
    a_data = df[df[group_col] == group_a]
    b_data = df[df[group_col] == group_b]
    a_vals = a_data[a_data['TotalClaims'] > 0]['TotalClaims'] if metric == 'claim_severity' else a_data['margin']
    b_vals = b_data[b_data['TotalClaims'] > 0]['TotalClaims'] if metric == 'claim_severity' else b_data['margin']
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    t_stat, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy='omit')
    return {
        'test': 'T-Test', 'metric': metric,
        'group_a': group_a, 'group_b': group_b, 'p_value': p
    }

def check_group_equivalence(df: pd.DataFrame, group_col, group_a, group_b, check_cols):
    """
    Check if group A and B are statistically similar across specified control columns.
    """
    results = []
    df = df[df[group_col].isin([group_a, group_b])]
    for col in check_cols:
        if df[col].dtype in ['object', 'category']:
            contingency = pd.crosstab(df[group_col], df[col])
            if contingency.min().min() >= 5:
                _, p, _, _ = chi2_contingency(contingency)
                results.append({'column': col, 'test': 'Chi-Squared', 'p_value': p})
        else:
            a_vals = df[df[group_col] == group_a][col].dropna()
            b_vals = df[df[group_col] == group_b][col].dropna()
            if len(a_vals) >= 2 and len(b_vals) >= 2:
                _, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy='omit')
                results.append({'column': col, 'test': 'T-Test', 'p_value': p})
    return pd.DataFrame(results)

def run_all_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all required hypothesis tests and return result DataFrame with adjusted p-values.
    """
    results = []
    print("Running Hypothesis Tests...")

    # Province
    results.append(chi_squared_test(df, 'Province', 'Gauteng', 'KwaZulu-Natal'))
    results.append(t_test(df, 'Province', 'Gauteng', 'KwaZulu-Natal', 'claim_severity'))

    # MainCrestaZone High vs Low
    if 'MainCrestaZone' in df.columns:
        zone_metrics = calculate_metrics(df, 'MainCrestaZone')
        top_zones = zone_metrics['claim_frequency'].nlargest(3).index
        bottom_zones = zone_metrics['claim_frequency'].nsmallest(3).index
        df = df.copy()
        df['ZoneRisk'] = df['MainCrestaZone'].apply(
            lambda z: 'High' if z in top_zones else 'Low' if z in bottom_zones else np.nan
        )
        df_zone = df[df['ZoneRisk'].notna()]
        results.append(chi_squared_test(df_zone, 'ZoneRisk', 'High', 'Low'))
        results.append(t_test(df_zone, 'ZoneRisk', 'High', 'Low', 'claim_severity'))
        results.append(t_test(df_zone, 'ZoneRisk', 'High', 'Low', 'margin'))

    # Gender
    if set(df['Gender'].dropna().unique()).issuperset({'Male', 'Female'}):
        results.append(chi_squared_test(df, 'Gender', 'Female', 'Male'))
        results.append(t_test(df, 'Gender', 'Female', 'Male', 'claim_severity'))

    # Filter valid results and adjust p-values
    results = [r for r in results if r is not None]
    if results:
        p_vals = [r['p_value'] for r in results]
        _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        for r, p in zip(results, p_adj):
            r['p_value_adjusted'] = p

    return pd.DataFrame(results)