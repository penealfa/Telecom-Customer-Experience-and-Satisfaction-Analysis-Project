import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA


# def describe_all(df):
#     print("Data Types:\n", df.dtypes)
#     print("\nSummary Statistics:\n", df.describe(include='all'))

# def handle_missing_values(df):
#     print("\nMissing Values:\n", df.isnull().sum())
#     df.fillna(df.mean(numeric_only=True), inplace=True)
#     z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
#     df = df[(z_scores < 3).all(axis=1)]

#     return df
# def variable_transformation(df):
#     df['Total Duration'] = df['Dur. (ms)']
#     df['Duration Decile'] = pd.qcut(df['Total Duration'], q=5, labels=False)

#     # Compute total data (DL+UL) per decile class
#     decile_data = df.groupby('Duration Decile').agg({
#         'Total DL (Bytes)': 'sum',
#         'Total UL (Bytes)': 'sum',
#         'Total Data (Bytes)': 'sum'  # Total Data (DL+UL)
#     }).reset_index()

# def metrics_analysis(df):
#     basic_metrics = df.describe()
#     print("\nBasic Metrics:\n", basic_metrics)

# def univariate_analysis(df):
#     dispersion_params = df.select_dtypes(include=[np.number]).describe()
#     plt.figure(figsize=(18, 12))
#     for i, column in enumerate(dispersion_params.columns, 1):
#         plt.subplot(3, 5, i)
#         sns.histplot(df[column], kde=True)
#         plt.title(f'Histogram of {column}')
#     plt.tight_layout()
#     plt.show()

# def bivariate_analysis(df):
#     plt.figure(figsize=(18, 12))
#     applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
#     for i, app in enumerate(applications, 1):
#         plt.subplot(2, 4, i)
#         sns.scatterplot(x=df[f'{app} DL (Bytes)'] + df[f'{app} UL (Bytes)'], y=df['Total Data (Bytes)'])
#         plt.title(f'{app} Data vs Total Data (Bytes)')
#     plt.tight_layout()
#     plt.show()

# def correlation_analysis(df):
#     correlation_matrix = df[['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
#                          'Google DL (Bytes)', 'Google UL (Bytes)',
#                          'Email DL (Bytes)', 'Email UL (Bytes)',
#                          'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
#                          'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
#                          'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
#                          'Other DL (Bytes)', 'Other UL (Bytes)']].corr()
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#     plt.title('Correlation Matrix of Application Data')
#     plt.show()

# def dim_reduction_using_pca(df):
#     # 9. Dimensionality Reduction using PCA
#     numerical_df = df.select_dtypes(include=[np.number])
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(numerical_df)

#     # Plot PCA results
#     plt.figure(figsize=(10, 6))
#     plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Duration Decile'], cmap='viridis')
#     plt.title('PCA of Numerical Variables')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.colorbar(label='Duration Decile')
#     plt.show()

#     # Summary of PCA
#     explained_variance = pca.explained_variance_ratio_
#     print("\nExplained Variance by PCA Components:\n", explained_variance)

def describe_all(df):
    print("Data Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe(include='all'))

def handle_missing_values(df):
    print("\nMissing Values:\n", df.isnull().sum())
    df.fillna(df.mean(numeric_only=True), inplace=True)
    z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]
    return df

def variable_transformation(df):
    df['Total_Duration'] = df['Dur. (ms)']
    df['Total_Data_Volume'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    df['Duration_Decile'] = pd.qcut(df['Total_Duration'], q=5, labels=False)
    decile_data = df.groupby('Duration_Decile').agg({
        'Total_Data_Volume': 'sum',
        'Total_Duration': 'mean'
    }).reset_index()
    print(decile_data)

def metrics_analysis(df):
    basic_metrics = df.describe()
    print("\nBasic Metrics:\n", basic_metrics)

def univariate_analysis(df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(df['Total_Duration'], bins=20, kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Total Session Duration')
    sns.histplot(df['Total_Data_Volume'], bins=20, kde=True, ax=axs[1])
    axs[1].set_title('Distribution of Total Data Volume')

    plt.tight_layout()
    plt.show()

def non_graphical_univariate_analysis(df):
    dp = df[['Total_Duration', 'Total_Data_Volume', 'Total DL (Bytes)', 'Total UL (Bytes)',
                            'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 
                            'Email DL (Bytes)', 'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
                            'Other DL (Bytes)', 'Other UL (Bytes)']].agg(['var', 'std'])

    print(dp)

def bivariate_analysis(df):
    features = [
        'Social Media DL (Bytes)',
        'Youtube DL (Bytes)',
        'Netflix DL (Bytes)',
        'Google DL (Bytes)',
        'Email DL (Bytes)',
        'Gaming DL (Bytes)'
    ]
    
    for feature in features:
        plt.figure()  # Create a new figure for each plot
        sns.scatterplot(data=df, x=feature, y='Total_Data_Volume')
        plt.title(f'Relationship Between {feature} and Total Data Volume')
        plt.tight_layout()
        plt.show()

def correlation_analysis(df):
    correlation_matrix =  ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 
               'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    corr_matrix = df[correlation_matrix].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix for Application Data Usage')
    plt.show()
    
def dim_reduction_using_pca(df):
    numerical_df = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numerical_df)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Duration_Decile'], cmap='viridis')
    plt.title('PCA of Numerical Variables')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Duration Decile')
    plt.show()

    explained_variance = pca.explained_variance_ratio_
    print("\nExplained Variance by PCA Components:\n", explained_variance)

