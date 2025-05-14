import pandas as pd
import numpy as np
import scipy.stats as stats
import os
from sklearn.preprocessing import MinMaxScaler
import darts
from darts import TimeSeries, concatenate


def load_and_preprocess_data(location):
    location_params = {
    'Linkou': {'train_end': 3362, 'val_end': 3727},
    'Taipei': {'train_end': 3372, 'val_end': 3737},
    'Kaohsiung': {'train_end': 3208, 'val_end': 3573},
    'Keelung': {'train_end': 3274, 'val_end': 3639},
    'Yunlin': {'train_end': 2557, 'val_end': 2622},
    'Chiayi': {'train_end': 3237, 'val_end': 3602}
    }
    # Construct file path
    file_name = f'EDvisitfile{location}.csv'
    file_path = os.path.join('..', 'DataSet', file_name)
    
    # Load the dataset
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Ensure 'date' column is in DateTime format and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Get training end index based on location
    train_end = location_params[location]['train_end']
    
    # Split the dataset into training and test sets
    train_df = df.iloc[:train_end].copy()  
    test_df = df.iloc[train_end:].copy()

    # Scaling the 'No' column using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_df[['No']])  # Fit the scaler only on the training data
    
    # Transform 'No' column
    train_df['No_scaled'] = scaler.transform(train_df[['No']])
    test_df['No_scaled'] = scaler.transform(test_df[['No']])
    
    # Convert to TimeSeries objects
    train_series_scaled = TimeSeries.from_dataframe(train_df, value_cols='No_scaled')
    test_series_scaled = TimeSeries.from_dataframe(test_df, value_cols='No_scaled')
    combined_series_scaled = concatenate([train_series_scaled, test_series_scaled])
    
    train_series = TimeSeries.from_dataframe(train_df, value_cols='No')
    test_series = TimeSeries.from_dataframe(test_df, value_cols='No')
    combined_series = concatenate([train_series, test_series])

    # Select columns to create multivariate TimeSeries (one-hot encoding)
    columns = ['Dayoff', 'Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun', 'Dayscaled', 
               'NewYear', 'Level_3_Alert', 'Outbreak', 'COVID19', 'Jan', 'Feb', 'Mar', 'Apr', 
               'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_multivariate = df[columns]
    
    # Convert the multivariate DataFrame to TimeSeries
    ED_covariates = TimeSeries.from_dataframe(df_multivariate)
    
    return train_series_scaled, test_series_scaled, combined_series_scaled, train_series,test_series,combined_series, ED_covariates, scaler



def statistics_load_and_preprocess_data(location):
    location_params = {
    'Linkou': {'train_end': 3362, 'val_end': 3727},
    'Taipei': {'train_end': 3372, 'val_end': 3737},
    'Kaohsiung': {'train_end': 3208, 'val_end': 3573},
    'Keelung': {'train_end': 3274, 'val_end': 3639},
    'Yunlin': {'train_end': 2557, 'val_end': 2622},
    'Chiayi': {'train_end': 3237, 'val_end': 3602}
    }
    # Construct file path
    file_name = f'EDvisitfile{location}.csv'
    file_path = os.path.join('..', 'DataSet', file_name)
    
    # Load the dataset
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Ensure 'date' column is in DateTime format and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Get training end index based on location
    train_end = location_params[location]['train_end']
    
    # Split the dataset into training and test sets
    train_df = df.iloc[:train_end].copy()  
    test_df = df.iloc[train_end:].copy()

    # Scaling the 'No' column using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_df[['No']])  # Fit the scaler only on the training data
    
    # Transform 'No' column
    train_df['No_scaled'] = scaler.transform(train_df[['No']])
    test_df['No_scaled'] = scaler.transform(test_df[['No']])
    
    # Convert to TimeSeries objects
    train_series_scaled = TimeSeries.from_dataframe(train_df, value_cols='No_scaled')
    test_series_scaled = TimeSeries.from_dataframe(test_df, value_cols='No_scaled')
    combined_series_scaled = concatenate([train_series_scaled, test_series_scaled])
    
    train_series = TimeSeries.from_dataframe(train_df, value_cols='No')
    test_series = TimeSeries.from_dataframe(test_df, value_cols='No')
    combined_series = concatenate([train_series, test_series])

    # Select columns to create multivariate TimeSeries (one-hot encoding)
    columns = ['Dayoff', 'Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun', 'Dayscaled', 
               'NewYear', 'Jan', 'Feb', 'Mar', 'Apr',
               'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_multivariate = df[columns]
    
    # Convert the multivariate DataFrame to TimeSeries
    ED_covariates = TimeSeries.from_dataframe(df_multivariate)
    
    return train_series_scaled, test_series_scaled, combined_series_scaled, train_series,test_series,combined_series, ED_covariates, scaler