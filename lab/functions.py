import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ShuffleSplit

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import make_reduction

SEED = 0

##########################################################################
#                          Outlier management                            #
##########################################################################

def remove_outliers_isolation_forests(data, contamination=0.05):
    outlier_detector = IsolationForest(contamination=contamination)
    for column in data:
        outlier_detector.fit(data[[column]].values)
        outlier_prediction = outlier_detector.predict(data[[column]].values)
        data.loc[outlier_prediction == -1, column] = np.nan
        data[[column]] = data[[column]].interpolate().fillna(method='ffill').fillna(method='bfill')

    return data


##########################################################################
#                     Predictor importance analysis                      #
##########################################################################




##########################################################################
#                             Price forecast                             #
##########################################################################

def crossval_window_size(y, X, forecaster, train_window_size_list, forecasting_horizon):
    cv_results_list = []

    for train_window_size in train_window_size_list:
        cv = SlidingWindowSplitter(window_length=train_window_size, fh=list(range(1,forecasting_horizon+1)), step_length=forecasting_horizon)
        # refit strategy makes a new model on each iteration, not an update
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", return_data=True, error_score="raise")
        cv_results_list.append(cv_results)

    return cv_results_list


def crossval_window_size_make_reduction(y, X, regressor, train_window_size_list, forecasting_horizon):
    cv_results_list = []

    for train_window_size in train_window_size_list:
        cv = SlidingWindowSplitter(window_length=train_window_size, fh=list(range(1,forecasting_horizon+1)), step_length=forecasting_horizon)
        # refit strategy makes a new model on each iteration, not an update
        forecaster = make_reduction(regressor, strategy="direct", window_length=train_window_size, windows_identical=True)
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


if __name__ == '__main__':
    from etl.esios.provider import ESIOSDataProvider

    esios_provider = ESIOSDataProvider()
    esios_tickers = esios_provider.get_tickers()

    esios_df = esios_provider.get_all_series(freq="H", start_index="2023-01-01 00:00", end_index="2023-03-31 23:59")

    contamination = 0.05
    #esios_df = remove_outliers_isolation_forests(esios_df, contamination)

    esios_spot = esios_df["PRECIO_MERCADO_SPOT_DIARIO"]
    esios_demand = esios_df["DEMANDA_REAL"]
    X = esios_df.drop(['PRECIO_MERCADO_SPOT_DIARIO', 'GENERACIÓN_MEDIDA_TOTAL'], axis=1)

    print(esios_spot.head(10))

    #compute_ml_ts_data(esios_spot,X,1,1)