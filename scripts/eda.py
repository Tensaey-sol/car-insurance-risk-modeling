import pandas as pd

def summarize_data(df: pd.DataFrame) -> None:
    """
    Print info and descriptive stats of the dataset.
    """
    print("Data Info:")
    print(df.info())
    print("\nDescriptive Stats:")
    print(df.describe())

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Return columns with missing values.
    """
    missing = df.isnull().sum()
    return missing[missing > 0]

def clean_insurance_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleans the insurance dataset by:
    - Converting date fields
    - Dropping columns with over 90% missing
    - Removing rows with negative premiums or claims
    - Handling zero-premium division in loss ratio
    - Reporting missingness summary
    """
    df_cleaned = df.copy()

    # 1. Convert TransactionMonth to datetime
    df_cleaned['TransactionMonth'] = pd.to_datetime(df_cleaned['TransactionMonth'], errors='coerce')

    # 2. Drop columns with over 90% missing
    missing_threshold = 0.90
    missing_frac = df_cleaned.isnull().mean()
    drop_cols = missing_frac[missing_frac > missing_threshold].index.tolist()
    df_cleaned.drop(columns=drop_cols, inplace=True)

    # 3. Remove rows with negative premiums or claims
    df_cleaned = df_cleaned[df_cleaned['TotalPremium'] >= 0]
    df_cleaned = df_cleaned[df_cleaned['TotalClaims'] >= 0]

    # 4. Recalculate Loss Ratio safely (avoid /0)
    df_cleaned['LossRatio'] = df_cleaned.apply(
        lambda row: row['TotalClaims'] / row['TotalPremium'] if row['TotalPremium'] > 0 else 0,
        axis=1
    )

    # 5. Report missing values (optional)
    if verbose:
        print("Columns dropped due to missingness > 90%:\n", drop_cols)
        print("\nRemaining missing value counts:\n", df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])

    return df_cleaned


def calculate_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Loss Ratio column to DataFrame.
    """
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    return df

def loss_ratio_by_group(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    """
    Calculate loss ratio aggregated by a specific column (e.g., Province, Gender).
    """
    group_stats = df.groupby(groupby_col)[['TotalClaims', 'TotalPremium']].sum()
    group_stats['LossRatio'] = group_stats['TotalClaims'] / group_stats['TotalPremium']
    return group_stats.sort_values('LossRatio', ascending=False)

def claims_by_make_model(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """
    Returns top and bottom N vehicle makes/models by TotalClaims.
    """
    claim_summary = df.groupby(['make', 'Model'])['TotalClaims'].sum().reset_index()
    top = claim_summary.sort_values('TotalClaims', ascending=False).head(top_n)
    bottom = claim_summary.sort_values('TotalClaims').head(top_n)
    return top, bottom