import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import zscore
import pandas as pd

def remove_columns(df, columns_to_remove):
    
    # Check if all columns are in the dataframe
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    # Drop the columns
    df = df.drop(columns=columns_to_remove)
    
    print(f"Removed columns: {columns_to_remove}")
    
    return df

def missing_values_table(df):
    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_dtype = df.dtypes
    
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis = 1)
    
    mis_val_table_ren_columns = mis_val_table.rename (
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'DType'}
    )
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
    
    print ("The dataframe has " + str(df.shape[1]) + "columns.\n"
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.\n")
           
    return mis_val_table_ren_columns
    
# Function to drop rows with missing values in identifier columns
def handle_missing_identifiers(df, columns):
    for column in columns:
        if column in df.columns:
            initial_rows = df.shape[0]
            df.dropna(subset=[column], inplace=True)
            final_rows = df.shape[0]
            print(f"Dropped {initial_rows - final_rows} rows due to missing values in '{column}'")
    return df

# Function to fill missing values for numerical columns based on a percentage threshold
def handle_missing_numerical(df, columns, threshold=65, fill_strategy='mean'):
    for column in columns:
        if column in df.columns:
            missing_percent = df[column].isnull().sum() * 100 / len(df)
            if missing_percent > threshold:
                df.drop(column, axis=1, inplace=True)
                print(f"Dropped column '{column}' due to {missing_percent:.1f}% missing values")
            else:
                if fill_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif fill_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                print(f"Filled missing values in '{column}' using {fill_strategy}")
    return df

# Function to fill missing values for categorical columns with 'Unknown'
def handle_missing_categorical(df, columns, fill_value='Unknown'):
    for column in columns:
        if column in df.columns:
            df[column].fillna(fill_value, inplace=True)
            print(f"Missing values in categorical column '{column}' filled with '{fill_value}'")
    return df

def convert_bytes_to_megabytes(df, columns):

    megabyte = 1 * 10e+5
    
    for column in columns:
        if column in df.columns:
            df[column] = df[column] / megabyte
    return df

def fix_outlier(df, columns):
    
    for column in columns:
        if column in df.columns:
            # Calculate 95th percentile and median
            percentile_95 = df[column].quantile(0.95)
            median_value = df[column].median()
            
            # Replace values higher than 95th percentile with median
            df[column] = np.where(df[column] > percentile_95, median_value, df[column])
           # print(f"Outliers in column {column} fixed (values above 95th percentile replaced with median).")
    
    return df

def remove_outliers(df, columns_to_process, z_threshold=3):

    for column in columns_to_process:
        if column in df.columns:
            z_scores = zscore(df[column])
            outlier_column = column + '_Outlier'
            
            # Flag rows as outliers
            df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
            
            # Filter out the rows with outliers
            df = df[df[outlier_column] == 0]
            
            # Drop the outlier flag column
            df = df.drop(columns=[outlier_column], errors='ignore')
            #print(f"Outliers removed from column {column} based on Z-score threshold of {z_threshold}.")
    
    return df