import pandas as pd

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.arima import AutoARIMA


def crossval_window_size(y, X, forecaster, train_window_size_list, forecasting_horizon):
    cv_results_list = []

    for train_window_size in train_window_size_list:
        cv = SlidingWindowSplitter(window_length=train_window_size, fh=list(range(1,forecasting_horizon+1)), step_length=forecasting_horizon)
        # refit strategy makes a new model on each iteration, not an update
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", return_data=True, error_score="raise")
        cv_results_list.append(cv_results)

    return cv_results_list


def crossval_plot_series(y, cv_results_list):
    for cv_results in cv_results_list:
        windows_nb = len(cv_results["y_pred"].tolist())

        fig, ax = plot_series(
            y,
            *cv_results["y_pred"].tolist(),
            markers=["o"] + ["" for x in range(windows_nb)]
        )


def relation_between_response_and_predictor(y, X, diff_lag):
    transformer = Differencer(lags=diff_lag, na_handling="drop_na")

    y_transform = transformer.fit_transform(y)
    x_transform = transformer.fit_transform(X)
    plot_series(y_transform)
    if isinstance(X, pd.Series):
        plot_series(x_transform)
    if isinstance(X, pd.DataFrame):
        if X.shape[1] <= 1:
            plot_series(x_transform)

    forecaster = AutoARIMA(suppress_warnings=True)
    forecaster.fit(y=y_transform, X=x_transform)

    return forecaster.get_fitted_params()