from prophet import Prophet


def build_prophet(params):
    # Build a Prophet model for one weekly series (yearly seasonality only).
    return Prophet(
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=False,
        daily_seasonality=False,
    )
