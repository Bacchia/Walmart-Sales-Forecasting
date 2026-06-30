import xgboost as xgb
from sklearn.pipeline import Pipeline

def create_xgboost_pipeline(preprocessor, model_params):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(**model_params))
    ])
