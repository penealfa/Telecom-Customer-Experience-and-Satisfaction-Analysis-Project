from scripts.load import load_data_from_postgres
from scripts.clean_dataset import handle_missing_identifiers,handle_missing_categorical,handle_missing_numerical,remove_columns,convert_bytes_to_megabytes,remove_outliers,fix_outlier


def get_data():
    
    query = "SELECT * FROM xdr_data;"  # Replace with your actual table name
    df = load_data_from_postgres(query)
    columns_to_remove = ['Start','Start ms','End','End ms','Dur. (ms).1','UL TP < 10 Kbps (%)',
                     '50 Kbps < UL TP < 300 Kbps (%)','UL TP > 300 Kbps (%)','10 Kbps < UL TP < 50 Kbps (%)',
                     'Nb of sec with Vol UL < 1250B','DL TP > 1 Mbps (%)','50 Kbps < DL TP < 250 Kbps (%)',
                     '250 Kbps < DL TP < 1 Mbps (%)','DL TP < 50 Kbps (%)','Nb of sec with 125000B < Vol DL',
                     'Nb of sec with 1250B < Vol UL < 6250B','Nb of sec with 31250B < Vol DL < 125000B','Nb of sec with 37500B < Vol UL',
                     'Nb of sec with 6250B < Vol DL < 31250B','Nb of sec with 6250B < Vol UL < 37500B','Nb of sec with Vol DL < 6250B',
                     'Nb of sec with Vol UL < 1250B','x','Last Location Name']
    byte_columns= ['HTTP UL (Bytes)','HTTP DL (Bytes)','Social Media DL (Bytes)','Social Media UL (Bytes)',
                      'Youtube DL (Bytes)','Youtube UL (Bytes)','Netflix DL (Bytes)','Netflix UL (Bytes)',
                      'Google DL (Bytes)','Google UL (Bytes)','Email DL (Bytes)','Email UL (Bytes)','Gaming DL (Bytes)',
                      'Gaming UL (Bytes)','Other DL (Bytes)','Other UL (Bytes)','Total DL (Bytes)','Total UL (Bytes)','TCP DL Retrans. Vol (Bytes)','TCP UL Retrans. Vol (Bytes)']
    
    df = remove_columns(df, columns_to_remove)
    identifier_columns = ['IMEI', 'IMSI', 'MSISDN/Number', 'Bearer Id']
    df = handle_missing_identifiers(df, identifier_columns)
    df.shape
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).dropna(axis=1, how='all').columns
    df = handle_missing_numerical(df, numeric_columns)
    df = handle_missing_categorical(df, ['Last Location Name'])
    df = convert_bytes_to_megabytes(df, byte_columns)
    df = fix_outlier(df, numeric_columns)
    df = remove_outliers(df, numeric_columns)
    df.shape
    return df
    