import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def merge_and_enrich(base_df, stores_df, features_df):
    df = base_df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    features_cp = features_df.copy()
    features_cp['Date'] = pd.to_datetime(features_cp['Date'])
    
    features_cp = features_cp.drop(columns=['IsHoliday'], errors='ignore')
    df = df.merge(stores_df, on='Store', how='left')
    df = df.merge(features_cp, on=['Store', 'Date'], how='left')
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)
    
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    
    return df

def get_model_ready_data(train_raw, stores, features, split_date='2012-01-01'):
    """
    Cleans, splits, and packages the data along with its preprocessor object.
    Returns: X_train, y_train, X_val, y_val, is_holiday_val, preprocessor
    """
    df_clean = merge_and_enrich(train_raw, stores, features)
    
    num_features = ['Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 
                    'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 
                    'Year', 'Month', 'Week', 'Day', 'IsHoliday']
    cat_features = ['Type']
    features_list = num_features + cat_features
    
    train_mask = df_clean['Date'] < split_date
    val_mask = df_clean['Date'] >= split_date
    
    X_train = df_clean[train_mask][features_list]
    y_train = df_clean[train_mask]['Weekly_Sales']
    
    X_val = df_clean[val_mask][features_list]
    y_val = df_clean[val_mask]['Weekly_Sales']
    is_holiday_val = df_clean[val_mask]['IsHoliday']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ('num', 'passthrough', num_features)
        ]
    )
    
    return X_train, y_train, X_val, y_val, is_holiday_val, preprocessor
