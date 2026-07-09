from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE


def build_nbeats(horizon, params, freq="W-FRI"):
    # Build a NeuralForecast pipeline with one NBEATS model from the config params.
    model = NBEATS(
        h=horizon,
        input_size=params["input_size"],
        max_steps=params["max_steps"],
        stack_types=params["stack_types"],
        n_blocks=params["n_blocks"],
        scaler_type=params["scaler_type"],
        start_padding_enabled=params["start_padding_enabled"],
        random_seed=params["random_seed"],
        loss=MAE(),
        alias="NBEATS",
    )
    return NeuralForecast(models=[model], freq=freq)
