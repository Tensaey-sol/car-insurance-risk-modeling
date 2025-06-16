import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_claim_distribution(df: pd.DataFrame):
    sns.histplot(df['TotalClaims'], bins=50)
    plt.title('Distribution of Total Claims')
    plt.show()

def plot_premium_distribution(df: pd.DataFrame):
    sns.histplot(df['TotalPremium'], bins=50)
    plt.title('Distribution of Total Premium')
    plt.show()

def plot_outliers_custom_value(df: pd.DataFrame):
    sns.boxplot(x=df['CustomValueEstimate'])
    plt.title('Outliers in Custom Value Estimate')
    plt.show()

def plot_loss_ratio_by_province(df: pd.DataFrame):
    summary = df.groupby('Province')[['TotalClaims', 'TotalPremium']].sum()
    summary['LossRatio'] = summary['TotalClaims'] / summary['TotalPremium']
    summary['LossRatio'].sort_values().plot(kind='barh', figsize=(10, 6))
    plt.title('Loss Ratio by Province')
    plt.xlabel('Loss Ratio')
    plt.ylabel('Province')
    plt.show()

def plot_claim_trend_over_time(df: pd.DataFrame):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    monthly = df.groupby(df['TransactionMonth'].dt.to_period('M'))['TotalClaims'].sum()
    monthly.plot(figsize=(12, 5))
    plt.title("Total Claims Over Time")
    plt.ylabel("Total Claims")
    plt.xlabel("Month")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate', 'SumInsured']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_scatter_premium_claims_by_zip(df: pd.DataFrame):
    zip_agg = df.groupby('PostalCode')[['TotalClaims', 'TotalPremium']].sum().reset_index()
    sns.scatterplot(data=zip_agg, x='TotalPremium', y='TotalClaims', hue='PostalCode', palette='viridis', legend=False)
    plt.title('Total Claims vs Premium by PostalCode')
    plt.show()
