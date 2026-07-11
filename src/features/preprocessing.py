import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def merge_and_enrich(train_raw, stores, features_raw):
    df = train_raw.merge(stores, on="Store", how="left")
    df = df.merge(features_raw, on=["Store", "Date", "IsHoliday"], how="left")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_model_ready_data(train_raw, stores, features_raw, split_date="2012-01-01"):
    df = merge_and_enrich(train_raw, stores, features_raw)
    
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Year'] = df['Date'].dt.year
    
    df['Is_Black_Friday'] = ((df['Date'].dt.month == 11) & (df['Date'].dt.day >= 23) & (df['Date'].dt.day <= 29)).astype(int)
    df['Pre_Christmas'] = (df['WeekOfYear'] == 51).astype(int)
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    
    df['Store_Dept'] = df['Store'].astype(str) + "_" + df['Dept'].astype(str)
    
    df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    df['Fuel_Price_Delta'] = df.groupby(['Store', 'Dept'])['Fuel_Price'].diff().fillna(0)
    df['CPI_Delta'] = df.groupby(['Store', 'Dept'])['CPI'].diff().fillna(0)
    df['Unemployment_Delta'] = df.groupby(['Store', 'Dept'])['Unemployment'].diff().fillna(0)
    
    train_mask = df['Date'] < pd.to_datetime(split_date)
    val_mask = df['Date'] >= pd.to_datetime(split_date)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    y_train = train_df['Weekly_Sales']
    y_val = val_df['Weekly_Sales']
    is_holiday_val = val_df['IsHoliday'].values
    
    categorical_features = ['Type']
    numeric_features = [
        'Store', 'Dept',
        'Temperature', 'Fuel_Price_Delta', 'CPI_Delta', 'Unemployment_Delta',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
        'Size', 'WeekOfYear', 'Is_Black_Friday', 'Pre_Christmas', 'IsHoliday'
    ]
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ], remainder='drop')
    
    # Return RAW feature frames; the model Pipeline applies `preprocessor` itself.
    # (Returning pre-transformed arrays caused a double-transform when wrapped in a Pipeline.)
    X_train = train_df[numeric_features + categorical_features]
    X_val = val_df[numeric_features + categorical_features]

    return X_train, y_train, X_val, y_val, is_holiday_val, preprocessor
