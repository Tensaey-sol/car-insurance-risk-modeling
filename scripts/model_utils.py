import pandas as pd
from sklearn.impute import SimpleImputer

def prepare_model_data(df: pd.DataFrame):
    df = df.copy()
    
    # Basic filtering
    df = df[(df['TotalPremium'] > 0) & (df['TotalClaims'] >= 0)]

    # Add target columns
    df['has_claim'] = (df['TotalClaims'] > 0).astype(int)
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    
    # Drop high-missing/uninformative cols
    drop_cols = ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth',
                 'VehicleIntroDate', 'CapitalOutstanding', 'CrossBorder',
                 'NumberOfVehiclesInFleet']  # extend as needed
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Identify categorical and numerical features
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('TotalClaims')
    num_cols.remove('TotalPremium')
    
    # Impute missing numerical values
    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df