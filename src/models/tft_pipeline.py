from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MAE


def build_tft(horizon, params, hist_exog=None, stat_exog=None, freq="W-FRI"):
    # Build a NeuralForecast pipeline with one TFT model from the config params.
    # Pass hist_exog / stat_exog to use exogenous features; leave None for univariate.
    model = TFT(
        h=horizon,
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        n_head=params["n_head"],
        max_steps=params["max_steps"],
        scaler_type=params["scaler_type"],
        start_padding_enabled=params["start_padding_enabled"],
        random_seed=params["random_seed"],
        hist_exog_list=hist_exog,
        stat_exog_list=stat_exog,
        loss=MAE(),
        alias="TFT",
    )
    return NeuralForecast(models=[model], freq=freq)
