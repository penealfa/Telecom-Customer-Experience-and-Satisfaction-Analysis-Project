import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def top_10_headset(df):
    handset_counts = df['Handset Type'].value_counts()
    top_10_handsets = handset_counts.head(10)
    plt.figure(figsize=(12, 6))
    top_10_handsets.plot(kind='bar')
    plt.title('Top 10 Handset Types')
    plt.xlabel('Handset Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()  
    plt.show()

def top_3_manufacturers(df):
    manufacturers_counts = df['Handset Manufacturer'].value_counts()
    top_3_manufacturers  = manufacturers_counts.head(3)
    plt.figure(figsize=(12, 6))
    top_3_manufacturers.plot(kind='bar')
    plt.title('Top 3 Manufacturers')
    plt.xlabel('Handset Manufacturer')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()  
    plt.show()

def plot_top_handsets_by_manufacturer(df):
    manufacturer_counts = df['Handset Manufacturer'].value_counts()
    top_3_manufacturers = manufacturer_counts.head(3)
    fig, axes = plt.subplots(nrows=len(top_3_manufacturers), ncols=1, figsize=(12, 6 * len(top_3_manufacturers)))
    
    if len(top_3_manufacturers) == 1:
        axes = [axes]
    
    for i, manufacturer in enumerate(top_3_manufacturers.index):
        top_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        top_handsets.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Top 5 Handsets for {manufacturer}')
        axes[i].set_xlabel('Handset Type')
        axes[i].set_ylabel('Count')
        axes[i].set_xticklabels(top_handsets.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def aggregate_xdr(df):
    agg_df = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # Number of xDR sessions
        'Dur. (ms)': 'sum',    # Total session duration
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum',
        'Social Media DL (Bytes)': 'sum',
        'Social Media UL (Bytes)': 'sum',
        'Youtube DL (Bytes)': 'sum',
        'Youtube UL (Bytes)': 'sum',
        'Netflix DL (Bytes)': 'sum',
        'Netflix UL (Bytes)': 'sum',
        'Google DL (Bytes)': 'sum',
        'Google UL (Bytes)': 'sum',
        'Email DL (Bytes)': 'sum',
        'Email UL (Bytes)': 'sum',
        'Gaming DL (Bytes)': 'sum',
        'Gaming UL (Bytes)': 'sum',
        'Other DL (Bytes)': 'sum',
        'Other UL (Bytes)': 'sum'
    }).reset_index()
    agg_df.shape

    

def vis_aggregate(df):
    aggregated_df = aggregate_xdr(df) # Replace with your actual data loading method

    # Bar Chart for Total Data Volume
    plt.figure(figsize=(10, 6))
    sns.barplot(data=aggregated_df.head(20), x='MSISDN/Number', y='Total Data Volume (Bytes)')
    plt.xticks(rotation=90)
    plt.title('Total Data Volume (Bytes) per User')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.tight_layout()
    plt.show()

    # Pie Chart for Application Data Volume Contribution
    data_sums = [
        aggregated_df['Social Media Total (Bytes)'].sum(),
        aggregated_df['Google Total (Bytes)'].sum(),
        aggregated_df['Email Total (Bytes)'].sum(),
        aggregated_df['Youtube Total (Bytes)'].sum(),
        aggregated_df['Netflix Total (Bytes)'].sum(),
        aggregated_df['Gaming Total (Bytes)'].sum(),
        aggregated_df['Other Total (Bytes)'].sum()
    ]
    labels = [
        'Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other'
    ]
    plt.figure(figsize=(8, 8))
    plt.pie(data_sums, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Application Data Volume Contribution')
    plt.show()

    # Scatter Plot for Session Duration vs. Total Data Volume
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=aggregated_df, x='Total Session Duration (ms)', y='Total Data Volume (Bytes)', alpha=0.5)
    plt.title('Session Duration vs. Total Data Volume')
    plt.xlabel('Total Session Duration (ms)')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.show()