from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_sarima(y_series, params):
    # Fit a SARIMA model on one weekly series (seasonal period 52 = yearly).
    model = SARIMAX(
        y_series,
        order=tuple(params["order"]),
        seasonal_order=tuple(params["seasonal_order"]),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False, maxiter=50)
