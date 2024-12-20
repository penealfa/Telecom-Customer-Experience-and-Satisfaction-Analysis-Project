import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def analyze_customer_metrics(df):
    aggregated_df = (
        df.groupby('MSISDN/Number')
          .agg({
              'Handset Type': 'first',
              'Dur. (ms)': 'sum',
              'Total DL (Bytes)': 'sum',
              'Total UL (Bytes)': 'sum',
              'Bearer Id': 'count'
          })
          .reset_index()
    )
    aggregated_df['Total Traffic (Bytes)'] = (
        aggregated_df['Total DL (Bytes)'] + aggregated_df['Total UL (Bytes)']
    )
    aggregated_df.rename(columns={
        'MSISDN/Number': 'Customer ID',
        'Bearer Id': 'Session Frequency',
        'Dur. (ms)': 'Total Duration (ms)',
        'Total DL (Bytes)': 'Total Download (Bytes)',
        'Total UL (Bytes)': 'Total Upload (Bytes)'
    }, inplace=True)
    metrics = {
        'Total Duration (ms)': 'Top 10 Customers by Total Duration (ms)',
        'Total Traffic (Bytes)': 'Top 10 Customers by Total Traffic (Bytes)',
        'Session Frequency': 'Top 10 Customers by Session Frequency'
    }
    def format_large_numbers(df, columns):
        for col in columns:
            df[col] = df[col].apply(lambda x: f"{x:,.0f}")
        return df
    for metric, title in metrics.items():
        top_10_customers = aggregated_df.nlargest(10, metric)
        if metric in ['Total Duration (ms)', 'Total Traffic (Bytes)']:
            top_10_customers = format_large_numbers(top_10_customers, [metric])
        elif metric == 'Session Frequency':
            top_10_customers['Session Frequency'] = top_10_customers['Session Frequency'].apply(lambda x: f"{x:,.0f}")

        print(f"\n{title}:")
        print(top_10_customers[['Customer ID', metric]].to_string(index=False))
    return aggregated_df

def cluster_engagement(aggregated_df, n_clusters=3):
    # Step 2: Normalize engagement metrics
    def normalize_metrics(df):
        scaler = StandardScaler()
        metrics_to_normalize = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
        normalized_df = pd.DataFrame(scaler.fit_transform(df[metrics_to_normalize]), columns=metrics_to_normalize)
        return normalized_df

    normalized_metrics = normalize_metrics(aggregated_df)

    # Step 3: Apply k-means clustering
    def apply_kmeans(normalized_df, n_clusters=n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(normalized_df)
        return kmeans.labels_

    aggregated_df['Engagement Cluster'] = apply_kmeans(normalized_metrics)

    # Analyze the clusters
    cluster_summary = aggregated_df.groupby('Engagement Cluster').agg({
        'Total Duration (ms)': 'mean',
        'Total Traffic (Bytes)': 'mean',
        'Session Frequency': 'sum',
        'Customer ID': 'count'
    }).rename(columns={'Customer ID': 'Number of Users'}).reset_index()

    print("Cluster Summary:")
    print(cluster_summary.head())

    # Plotting the cluster summary as a grouped bar chart
    metrics = ['Total Duration (ms)', 'Total Traffic (Bytes)', 'Session Frequency', 'Number of Users']
    x = np.arange(len(cluster_summary))
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, cluster_summary[metric], width, label=metric)

    ax.set_xlabel('Engagement Cluster')
    ax.set_ylabel('Values')
    ax.set_title('Cluster Summary')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(cluster_summary['Engagement Cluster'])
    ax.legend()

    plt.tight_layout()
    plt.show()

    return aggregated_df

def aggregate_user_traffic_per_application(df):
    applications = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
                'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                'Google DL (Bytes)', 'Google UL (Bytes)', 
                'Email DL (Bytes)', 'Email UL (Bytes)', 
                'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
                'Other DL (Bytes)', 'Other UL (Bytes)']
    top_users_per_app = {}

    for app in applications:
        app_traffic = df.groupby('MSISDN/Number').agg({
            f'{app}': 'sum',
            f'{app}': 'sum'
        }).reset_index()
        app_traffic['Total Data (Bytes)'] = app_traffic[f'{app}'] + app_traffic[f'{app}']
        top_users = app_traffic.nlargest(10, 'Total Data (Bytes)')
        top_users_per_app[app] = top_users
    
    return top_users_per_app

def top_3_applications(df):
    applications = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
                'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                'Google DL (Bytes)', 'Google UL (Bytes)', 
                'Email DL (Bytes)', 'Email UL (Bytes)', 
                'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
                'Other DL (Bytes)', 'Other UL (Bytes)']
    app_traffic = {}

    for app in applications:
        total_traffic = df[[f'{app}', f'{app}']].sum().sum()
        app_traffic[app] = total_traffic

    app_traffic_df = pd.DataFrame(list(app_traffic.items()), columns=['Application', 'Total Data (Bytes)'])
    top_3_apps = app_traffic_df.sort_values(by='Total Data (Bytes)', ascending=False).head(3)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_3_apps['Application'], top_3_apps['Total Data (Bytes)'], color='skyblue')
    plt.title('Top 3 Applications by Total Data (Bytes)')
    plt.xlabel('Application')
    plt.ylabel('Total Data (Bytes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_cluster_metrics(df):
    cluster_metrics = df.groupby('Engagement Cluster').agg({
        'Session Frequency': ['min', 'max', 'mean', 'sum'],
        'Total Duration (ms)': ['min', 'max', 'mean', 'sum'],
        'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
    })
    return cluster_metrics

def normalize_metrics(df):
    scaler = StandardScaler()
    metrics_to_normalize = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
    normalized_df = pd.DataFrame(scaler.fit_transform(df[metrics_to_normalize]), columns=metrics_to_normalize)
    return normalized_df

def aggregate_metrics(df):
    aggregated_df = df.groupby('MSISDN/Number').agg({
        'Handset Type': 'first',
        'Dur. (ms)': 'sum',  # Total session duration
        'Total DL (Bytes)': 'sum',  # Total download data
        'Total UL (Bytes)': 'sum',  # Total upload data
        'Bearer Id': 'count'  # Session frequency
    }).reset_index()
    
    # Add total traffic column (DL + UL)
    aggregated_df['Total Traffic (Bytes)'] = aggregated_df['Total DL (Bytes)'] + aggregated_df['Total UL (Bytes)']
    
    # Rename columns for better readability
    aggregated_df.rename(columns={
        'MSISDN/Number': 'Customer ID',
        'Bearer Id': 'Session Frequency',
        'Dur. (ms)': 'Total Duration (ms)',
        'Total DL (Bytes)': 'Total Download (Bytes)',
        'Total UL (Bytes)': 'Total Upload (Bytes)',
        'Total Traffic (Bytes)': 'Total Traffic (Bytes)'
    }, inplace=True)
    
    return aggregated_df

def elbow_method(normalized_df):
    inertia = []
    k_range = range(1, 5)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_df)
        inertia.append(kmeans.inertia_)
    
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.show()