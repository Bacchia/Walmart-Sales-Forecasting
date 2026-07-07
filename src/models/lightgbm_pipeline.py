import lightgbm as lgb
from sklearn.pipeline import Pipeline

def create_lightgbm_pipeline(preprocessor, model_params):

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**model_params))
    ])
