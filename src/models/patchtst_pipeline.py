from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE


def build_patchtst(horizon, params, freq="W-FRI"):
    # Build a NeuralForecast pipeline with one PatchTST model from the config params.
    model = PatchTST(
        h=horizon,
        input_size=params["input_size"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        encoder_layers=params["encoder_layers"],
        n_heads=params["n_heads"],
        hidden_size=params["hidden_size"],
        linear_hidden_size=params["linear_hidden_size"],
        max_steps=params["max_steps"],
        start_padding_enabled=params["start_padding_enabled"],
        random_seed=params["random_seed"],
        loss=MAE(),
        alias="PatchTST",
    )
    return NeuralForecast(models=[model], freq=freq)
