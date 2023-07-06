import os
import pandas as pd
import numpy as np
import shap

from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ShuffleSplit

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError, mean_absolute_scaled_error, mean_absolute_error
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import make_reduction

SEED = 0

##########################################################################
#                          Outlier management                            #
##########################################################################

def remove_outliers_isolation_forests(data, contamination=0.01):
    outlier_detector = IsolationForest(contamination=contamination)
    for column in data:
        outlier_detector.fit(data[[column]].values)
        outlier_prediction = outlier_detector.predict(data[[column]].values)
        data.loc[outlier_prediction == -1, column] = np.nan
        data[[column]] = data[[column]].interpolate().fillna(method='ffill').fillna(method='bfill')

    return data


##########################################################################
#                           ML table computation                         #
##########################################################################

def compute_lag(y, lags):
    name = y.columns[0]

    for lag in lags:
        kwargs = {name+"_shift"+str(lag): y[name].shift(lag)}
        y = y.assign(**kwargs)
        # y = pd.concat([y, y[name].shift(lag)], axis=1)

    y = y.drop(name, axis=1)
    return y


def compute_lag_fh_df(y, lags, fh):
    return pd.concat([y, compute_lag(y, lags).shift(fh-1)], axis=1)


def add_date_features(y, X, lags, fh, date_features):
    y = pd.DataFrame(y)

    if "year" in date_features:
        X['year'] = y.index.year
    if "month" in date_features:
        X['month'] = y.index.month
    if "day" in date_features:
        X['day'] = y.index.day
    if "hour":
        X['hour'] = y.index.hour
    if "day_of_week" in date_features:
        X['day_of_week'] = y.index.day_of_week
    if "day_of_year" in date_features:
        X['day_of_year'] = y.index.day_of_year
    if "week_of_year" in date_features:
        X['week_of_year'] = y.index.weekofyear

    df = pd.concat([compute_lag_fh_df(y, lags, fh), X], axis=1)
    df = df.copy()
    return df.dropna()


##########################################################################
#                             Price forecast                             #
##########################################################################

def crossval_model_sktime(y, X, regressor_list, regressor_str_list, window_length, max_lag, step_size, fh, save_path, save_name):
    cv = SlidingWindowSplitter(fh, window_length, step_size)
    #cv = ExpandingWindowSplitter(fh, window_length, step_size)

    cv_results_list = []
    for regressor, regressor_str in zip(regressor_list, regressor_str_list):
        print(f"Training setup: regressor {regressor_str}")

        # refit strategy makes a new model on each iteration, not an update
        forecaster = make_reduction(regressor, strategy="direct", window_length=max_lag, windows_identical=True)
        # cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError(), backend="dask", compute=True)
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError())
        cv_results_list.append(cv_results)

    cv_results_df = pd.DataFrame(columns=list(cv_results_list[0].columns))
    for element, regressor_str in zip(cv_results_list, regressor_str_list):
        element["model"] = regressor_str
        cv_results_df = pd.concat([cv_results_df, element], axis=0, ignore_index=True)

    os.makedirs(save_path, exist_ok=True)
    cv_results_df.to_pickle(save_path + save_name)
    return cv_results_df


def crossval_summary_sktime(cv_results_df):
    cv_results_df_summary = pd.DataFrame(columns=['model', 'mae', 'mase', 'fit_time'])
    for model in cv_results_df["model"].unique():
        cv_results_model = cv_results_df[cv_results_df["model"] == model]

        mae = 0
        for test, pred in zip(cv_results_model["y_test"], cv_results_model["y_pred"]):
            mae += mean_absolute_error(test, pred)
        row = {'model': model,
               'mae': mae,
               'mase': cv_results_model["test_MeanAbsoluteScaledError"].mean(),
               'fit_time': cv_results_model["fit_time"].sum()
               }
        row_df = pd.DataFrame([row])
        cv_results_df_summary = pd.concat([cv_results_df_summary, row_df], axis=0, ignore_index=True)

    cv_results_df_summary = cv_results_df_summary.sort_values(by="mase")
    return cv_results_df_summary


def crossval_plot_series(y, cv_results_df):
    for model in cv_results_df["model"].unique():
        cv_results_model = cv_results_df[cv_results_df["model"]==model]
        windows_nb = len(cv_results_model["y_pred"].tolist())

        fig, ax = plot_series(
            y,
            *cv_results_model["y_pred"].tolist(),
            markers=["o"] + ["" for x in range(windows_nb)]
        )

        ax.plot


def crossval_arima_sktime(y, X, window_lengths, step_size, fh, save_path, save_name):

    cv_results_list = []
    for window_length in window_lengths:
        cv = SlidingWindowSplitter(fh, window_length, step_size)

        forecaster_str = f"AutoARIMA_{window_length}"
        print(f"Training setup: regressor {forecaster_str}")

        forecaster = AutoARIMA(maxiter=200, random_state=0)
        cv_results = evaluate(forecaster, y=y, X=X, cv=cv, strategy="refit", error_score="raise", return_data=True, scoring=MeanAbsoluteScaledError())
        cv_results_list.append(cv_results)

    cv_results_df = pd.DataFrame(columns=list(cv_results_list[0].columns))
    for result, window_length in zip(cv_results_list, window_lengths):
        forecaster_str = f"AutoARIMA_{window_length}"
        result["model"] = forecaster_str
        cv_results_df = pd.concat([cv_results_df, result], axis=0, ignore_index=True)

    os.makedirs(save_path, exist_ok=True)
    cv_results_df.to_pickle(save_path + save_name)
    return cv_results_df


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


##########################################################################
#                         Predictor's influence                          #
##########################################################################

def predictors_influence_study(esios_spot, X, best_model, lags, date_features, forecasting_horizon, version="first", seed=0):
    if version=="first":
        df_date_features = add_date_features(esios_spot, X, lags, 1, date_features)
    elif version=="last":
        df_date_features = add_date_features(esios_spot, X, lags, forecasting_horizon, date_features)
    else:
        raise NotImplementedError

    if version=="first":
        df_date_features = df_date_features.iloc[:-forecasting_horizon, :]

    y_date_features = df_date_features.iloc[:, 0]
    X_date_features = df_date_features.iloc[:, 1:]

    y_date_features_train, y_date_features_test, X_date_features_train, X_date_features_test = temporal_train_test_split(
    y_date_features, X_date_features, test_size=0.15)

    best_model.fit(X_date_features_train, y_date_features_train)
    y_pred = best_model.predict(X_date_features_test)
    print("Prediction MASE:", mean_absolute_scaled_error(y_date_features_test, y_pred, y_train=y_date_features_train))
    explainer = shap.Explainer(best_model.predict, X_date_features_test, seed=seed)
    shap_values = explainer(X_date_features_test)

    return shap_values


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